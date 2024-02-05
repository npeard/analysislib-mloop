from mloop.controllers import GaussianProcessController, ControllerInterrupt
import mloop.utilities as mlu
import traceback

class LoopController(GaussianProcessController):
    """
    This controller is modified to enable queue based interaction with the learner, so 
    several requests for a loss value can be made prior to actually getting a response
    """
    def __init__(self, interface, *args, **kwargs):
        
        if "training_type" in kwargs:
            self.log.info(f"training_type was provided as '{kwargs['training_type']}', but note that only 'random' is supported")
        kwargs["training_type"] = "random"

        super(LoopController, self).__init__(interface, *args, **kwargs)

        self.log.info("Starting LoopController")

        self.num_buffered_runs = interface.num_buffered_runs

    def _get_cost_and_in_dict(self):
        '''
        Get cost, uncertainty, parameters, bad and extra data from experiment.

        This method stores results in lists and also puts data in the
        appropriate 'current' variables. This method doesn't return anything and
        instead stores all of its results in the internal storage arrays and the
        'current' variables.
        
        If the interface encounters an error, it will pass the error to the
        controller here so that the error can be re-raised in the controller's
        thread (note that the interface runs in a separate thread).
        '''
        while True:
            try:
                in_dict = self.costs_in_queue.get(True, self.controller_wait)
            except mlu.empty_exception:
                # Check for an error from the interface.
                try:
                    err = self.interface_error_queue.get_nowait()
                except mlu.empty_exception:
                    # The interface didn't send an error, so go back to waiting
                    # for results.

                    # IBS: check if we have enough buffered runs.  If not
                    # break out of this loop so we can request more.
                    if self.last_out_params.qsize() < self.num_buffered_runs:
                        self.log.debug("Found no costs in queue, and am also requesting another shot")

                        return
                else:
                    # Log and re-raise the error sent by the interface.
                    msg = 'The interface raised an error with traceback:\n'
                    msg = msg + '\n'.join(
                        traceback.format_tb(err.__traceback__),
                    )
                    self.log.error(msg)
                    raise err
            else:
                # Got a cost dict, so exit this while loop.
                break

        self.num_in_costs += 1
        self.num_last_best_cost += 1

        if not ('cost' in in_dict) and (not ('bad' in in_dict) or not in_dict['bad']):
            self.log.error('You must provide at least the key cost or the key bad with True.')
            raise ValueError
        try:
            self.curr_cost = float(in_dict.pop('cost',float('nan')))
            self.curr_uncer = float(in_dict.pop('uncer',0))
            self.curr_bad = bool(in_dict.pop('bad',False))
            self.curr_extras = in_dict
        except ValueError:
            self.log.error('One of the values you provided in the cost dict could not be converted into the right type.')
            raise
        if self.curr_bad and ('cost' in in_dict):
            self.log.warning('The cost provided with the bad run will be saved, but not used by the learners.')

        self.in_costs.append(self.curr_cost)
        self.in_uncers.append(self.curr_uncer)
        self.in_bads.append(self.curr_bad)
        self.in_extras.append(self.curr_extras)
        self.curr_params = self.last_out_params.get() # IBS: change to queue here
        if self.curr_cost < self.best_cost: 
            self.best_cost = self.curr_cost
            self.best_uncer = self.curr_uncer
            self.best_index =  self.num_in_costs - 1  # -1 for zero-indexing.
            self.best_params = self.curr_params
            self.best_in_extras = self.curr_extras
            self.num_last_best_cost = 0
        if self.curr_bad:
            self.log.info('bad run')
        else:
            self.log.info('cost ' + str(self.curr_cost) + ' +/- ' + str(self.curr_uncer))

        # IBS: composing base Controller class with MachineLearnerController manually
        self.log.debug('sending data to machine learner')
        self.ml_learner_costs_queue.put(
            (self.curr_params, self.curr_cost, self.curr_uncer, self.curr_bad)
        )


    def _optimization_routine(self):
        '''
        Overrides _optimization_routine. Uses the parent routine for the training runs. Implements a customized _optimization_routine when running the machine learning learner.
        '''
        #Run the training runs using the standard optimization routine.
        self.log.debug('Starting training optimization.')
        self.log.info(f'Run (training): {self.num_in_costs +1} of {self.num_training_runs}')

        next_params = self._first_params()
        self._put_params_and_out_dict(next_params,param_type=self.learner.OUT_TYPE)
        self.save_archive()
        self._get_cost_and_in_dict()

        while (self.num_in_costs < self.num_training_runs) and self.check_end_conditions():
            self.log.info(f'Run (training): {self.num_in_costs +1} of {self.num_training_runs}')
            next_params = self._next_params()
            self._put_params_and_out_dict(next_params, param_type=self.learner.OUT_TYPE)
            self.save_archive()
            self._get_cost_and_in_dict()

        if self.check_end_conditions():
            #Start last training run
            self.log.info(f'Run (training): {self.num_in_costs +1} of {self.num_training_runs}')
            next_params = self._next_params()
            self._put_params_and_out_dict(next_params, param_type=self.learner.OUT_TYPE)

            self.log.debug('Starting ML optimization.')
            # This may be a race. Although the cost etc. is put in the queue to
            # the learner before the new_params_event is set, it's not clear if
            # python guarantees that the other process will see the item in the
            # queue before the event is set. To work around this,
            # learners.MachineLearner.get_params_and_costs() blocks with a
            # timeout while waiting for an item in the queue.
            self._get_cost_and_in_dict()
            self.save_archive()
            self.new_params_event.set()
            self.log.debug('End training runs.')

            ml_count = 0

        while self.check_end_conditions():
            run_num = self.num_in_costs + 1
            if ml_count==self.generation_num or (self.no_delay and self.ml_learner_params_queue.empty()):
                self.log.info(f'Run (trainer): {run_num}')
                next_params = self._next_params()
                self._put_params_and_out_dict(next_params, param_type=self.learner.OUT_TYPE)
            else:
                self.log.info(f'Run (machine learner): {run_num}')
                next_params = self.ml_learner_params_queue.get()
                self._put_params_and_out_dict(next_params, param_type=self.ml_learner.OUT_TYPE)
                ml_count += 1

            self.save_archive()
            self._get_cost_and_in_dict()

            if ml_count==self.generation_num:
                self.log.debug(f'Requesting new parameters (machine learner)')
                self.new_params_event.set()
                ml_count = 0


    def optimize(self):
        '''
        Optimize the experiment. This code learner and interface processes/threads are launched and appropriately ended.
        Starts both threads and catches kill signals and shuts down appropriately.
        '''
        try:
            self.log.info('Optimization started.')
            self._start_up()
            self._optimization_routine()
            self.log.info('Controller finished. Closing down M-LOOP. Please wait a moment...')
        except ControllerInterrupt:
            self.log.warning('Controller ended by interruption.')
        except (KeyboardInterrupt,SystemExit):
            self.log.warning('!!! Do not give the interrupt signal again !!! \n M-LOOP stopped with keyboard interrupt or system exit. Please wait at least 1 minute for the threads to safely shut down. \n ')
            self.log.warning('Closing down controller.')
        except Exception as err:
            msg = f"Unexpected controller {err=}, {type(err)=} ; Starting shut down."
            self.log.warning(msg)
            self.halt_reasons.append(msg)
            self._shut_down()
            self.log.warning('Safely shut down. Below are results found before exception.')
            self.print_results()
            return

        self._shut_down()
        self.print_results()
        self.log.info('M-LOOP Done.')
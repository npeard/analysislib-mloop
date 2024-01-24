import lyse
from . import mloop_config
from runmanager.remote import set_globals, engage
from mloop.interfaces import Interface
from mloop.controllers import GaussianProcessController
import mloop.utilities as mlu
import logging
import queue


logger = logging.getLogger('analysislib_mloop')


def set_globals_mloop(mloop_session=None, mloop_iteration=None):
    """Set globals named 'mloop_session' and 'mloop_iteration'
    based on the current . Defaults are None, which will ideally
    remain that way unless there is an active optimisation underway.
    """
    if mloop_iteration and mloop_session is None:
        globals = {'mloop_iteration': mloop_iteration}
    else:
        globals = {'mloop_session': mloop_session, 'mloop_iteration': mloop_iteration}
    try:
        set_globals(globals)
        logger.debug('mloop_iteration and/or mloop_session set.')
    except ValueError:
        logger.debug('Failed to set mloop_iteration and/or mloop_session.')

"""
 ######   #######  ##    ## ######## ########   #######  ##       ##       ######## ########
##    ## ##     ## ###   ##    ##    ##     ## ##     ## ##       ##       ##       ##     ##
##       ##     ## ####  ##    ##    ##     ## ##     ## ##       ##       ##       ##     ##
##       ##     ## ## ## ##    ##    ########  ##     ## ##       ##       ######   ########
##       ##     ## ##  ####    ##    ##   ##   ##     ## ##       ##       ##       ##   ##
##    ## ##     ## ##   ###    ##    ##    ##  ##     ## ##       ##       ##       ##    ##
 ######   #######  ##    ##    ##    ##     ##  #######  ######## ######## ######## ##     ##
"""

class LoopController(GaussianProcessController):
    """
    This controller is modified to enable queue based interaction with the learner, so 
    several requests for a loss value can be made prior to actually getting a response
    """
    def __init__(self, interface, *args, **kwargs):
        
        super(LoopController, self).__init__(interface, *args, **kwargs)
        formatter = logging.Formatter(
            '%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s: %(message)s'
        )
        [h.setFormatter(formatter) for h in self.log.handlers]
        self.log.info("Starting LoopController")

        self.num_buffered_runs = interface.num_buffered_runs

        # This is the biggest change
        self.last_out_params = queue.Queue()
        self.cost_obtained = False

    def _put_params_and_out_dict(self, params, param_type=None, **kwargs):
        '''
        Send parameters to queue with optional additional keyword arguments.

        This method also saves sent variables in appropriate storage arrays.

        Args:
            params (array): Array of values to be experimentally tested.
            param_type (Optional, str): The learner type which generated the
                parameter values. Because some learners use other learners as
                trainers, the parameter type can be different for different
                iterations during a given optimization. This value will be
                stored in `self.out_type` and in the `out_type` list in the
                controller archive. If `None`, then it will be set to
                `self.learner.OUT_TYPE`. Default `None`.
        Keyword Args:
            **kwargs: Any additional keyword arguments will be stored in
                `self.out_extras` and in the `out_extras` list in the controller
                archive.
        '''
        # Set default values if needed.
        if param_type is None:
            param_type = self.learner.OUT_TYPE

        # Do one last check to ensure parameter values are within the allowed
        # limits before sending those values to the interface.
        params = self._enforce_boundaries(params)

        # Send the parameters to the interface and update various attributes.
        out_dict = {'params':params}
        out_dict.update(kwargs)
        self.params_out_queue.put(out_dict)
        self.num_out_params += 1
        self.last_out_params.put(params) # IBS: the only change here
        self.out_params.append(params)
        self.out_extras.append(kwargs)
        self.out_type.append(param_type)
        self.log.info('params ' + str(params))

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
                        self.cost_obtained = False
                        self.log.debug("Looked for costs in queue, and am requesting another shot")

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
                self.cost_obtained = True

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
        self.log.debug('sending data to learner')
        self.ml_learner_costs_queue.put((self.curr_params,
                                    self.curr_cost,
                                    self.curr_uncer,
                                    self.curr_bad))

"""
##        #######   #######   #######  ########  ######## ########
##       ##     ## ##     ## ##     ## ##     ## ##       ##     ##
##       ##     ## ##     ## ##     ## ##     ## ##       ##     ##
##       ##     ## ##     ## ##     ## ########  ######   ########
##       ##     ## ##     ## ##     ## ##        ##       ##   ##
##       ##     ## ##     ## ##     ## ##        ##       ##    ##
########  #######   #######   #######  ##        ######## ##     ##
"""

class LoopInterface(Interface):
    def __init__(self, config_file):
        
        # Retrieve configuration from file or generate from defaults
        self.config = mloop_config.get(config_file)

        # this will be the number of runs to pre-submit to blacs via runmanager.  
        # The point of this is to keep the shot queue non-empty and avoid delays
        self.num_buffered_runs = int(self.config.get("num_buffered_runs", 0))

        # Pass config arguments to parent class's __init__() so that any
        # relevant specified options are applied appropriately.
        super(LoopInterface, self).__init__(**self.config)
        formatter = logging.Formatter(
            '%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s: %(message)s'
        )
        [h.setFormatter(formatter) for h in self.log.handlers] # doesn't work

        self.num_in_costs = 0



    def run(self):
        '''
        The run sequence for the interface.

        Overloaded to enable pre-submission of shots
        '''

        self.log.debug('Entering main loop of interface.')
        while not self.end_event.is_set():
            # Wait for the next set of parameter values to test.
            try:
                print("Trying for params_dict")
                params_dict = self.params_out_queue.get(
                    True,
                    self.interface_wait,
                )
                print("Got params_dict", params_dict)
            except mlu.empty_exception:
                continue

            # Try to run self.get_next_cost_dict(), passing any errors on to the
            # controller. Note that the interface and controller run in separate
            # threads which is why the error has to be passed through a queue
            # rather than just raised here. If it were raised here, then the
            # interface thread would crash and the controller thread would get
            # stuck indefinitely waiting for results from the interface.

            # only get the cost after the first self.num_buffered_runs
            get_cost = (self.num_in_costs >= self.num_buffered_runs)
            try:
                print("Trying for cost_dict")
                cost_dict = self.get_next_cost_dict(params_dict, get_cost=get_cost)
                print("Got cost_dict", params_dict)
            except Exception as err:
                # Send the error to the controller and set the end event to shut
                # down the interface. Setting the end event here and now
                # prevents the interface from running another iteration when
                # there are more items in self.params_out_queue already.
                self.interface_error_queue.put(err)
                self.end_event.set()
            else:
                # Send the results back to the controller.
                if get_cost:
                    self.costs_in_queue.put(cost_dict)
                else:
                    self.log.debug('Shot submitted but cost not recorded.')

                    
        self.log.debug('Interface ended')

    # Method called by M-LOOP upon each new iteration to determine the cost
    # associated with a given point in the search space
    def get_next_cost_dict(self, params_dict, get_cost=True):
        self.num_in_costs += 1
        # Store current parameters to later verify reported cost corresponds to these
        # or so mloop_multishot.py can fake a cost if mock = True
        logger.debug('Storing requested parameters in lyse.routine_storage.')
        globals_dict = mloop_config.prepare_globals(
                self.config['runmanager_globals'],
                dict(zip(self.config['mloop_params'].keys(), params_dict['params']))
        )

        lyse.routine_storage.params.put(globals_dict) 

        if not self.config['mock']:
            logger.info('Requesting next shot from experiment interface...')
            logger.debug('Setting optimization parameter values.')
            set_globals(globals_dict)
            logger.debug('Setting mloop_iteration...')
            set_globals_mloop(mloop_iteration=self.num_in_costs)
            logger.debug('Calling engage().')
            engage()

        # Only proceed once per lyse call ONCE we have num_buffered_runs queued up.
        if get_cost:
            logger.info('Waiting for next cost from lyse queue...')
            cost_dict = lyse.routine_storage.queue.get()
            logger.debug('Got cost_dict from lyse queue: {cost}'.format(cost=cost_dict))
        else:
            logger.info('Not waiting for lyse queue...')
            cost_dict = {}

        return cost_dict

"""
##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##
"""

def main(config):
    # Create M-LOOP optmiser interface with desired parameters
    interface = LoopInterface(config)

    # Instantiate experiment controller
    controller = LoopController(interface, **interface.config)

    # Define the M-LOOP session ID and initialise the mloop_iteration
    set_globals_mloop(controller.start_datetime.strftime('%Y%m%dT%H%M%S'), 0)

    # Run the optimiser using the constructed interface
    controller.optimize()

    # Reset the M-LOOP session and index to None
    logger.info('Optimisation ended.')
    set_globals_mloop()

    # Set the optimisation globals to their best results
    logger.info('Setting best parameters in runmanager.')
    globals_dict = mloop_config.prepare_globals(
            interface.config['runmanager_globals'],
            dict(zip(interface.config['mloop_params'].keys(), controller.best_params))
    )
    set_globals(globals_dict)

    # Return the results in a dictionary
    opt_results = {}
    opt_results['best_params'] = controller.best_params
    opt_results['best_cost'] = controller.best_cost
    opt_results['best_uncer'] = controller.best_uncer
    opt_results['best_index'] = controller.best_index
    return opt_results

import lyse
from . import mloop_config
from . import mloop_controller
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


class LoopInterface(Interface):
    def __init__(self, config_file):
        
        # Retrieve configuration from file or generate from defaults
        self.config = mloop_config.get(config_file)

        # the number of runs to pre-submit to blacs via runmanager.  
        self.num_buffered_runs = int(self.config.get("num_buffered_runs", 0))

        # Pass config arguments to parent class's __init__() so that any
        # relevant specified options are applied appropriately.
        super(LoopInterface, self).__init__(**self.config)

        formatter = logging.Formatter('%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s: %(message)s')
        for i in range(len(self.log.handlers)):
            self.log.handlers[i].setFormatter(formatter)

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
                self.log.debug("Trying for params_dict")
                params_dict = self.params_out_queue.get(
                    True,
                    self.interface_wait,
                )
                self.log.debug("Got params_dict", params_dict)
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
                self.log.debug("Trying for cost_dict")
                cost_dict = self.get_next_cost_dict(params_dict, get_cost=get_cost)
                self.log.debug("Got cost_dict", params_dict)
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

                    
        self.log.debug('Interface ended normally')

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
    controller = mloop_controller.LoopController(interface, **interface.config)

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

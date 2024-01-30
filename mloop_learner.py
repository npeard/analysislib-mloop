from mloop.learners import Learner
import mloop.utilities as mlu

import threading
import queue
import time
import numpy as np

class SimpleRandomLearner(Learner, threading.Thread):
    '''
    Random learner. Simply generates new parameters randomly with a uniform distribution over the boundaries. Learner is perhaps a misnomer 
    for this class.  Unlike the M-loop version, this complexly ignores the cost sent to it.  Its only job is to make sure that its queue is non-empty.

    Args:
        **kwargs (Optional dict): Other values to be passed to Learner.

    Keyword Args:
        min_boundary (Optional [array]): If set to None, overrides default learner values and sets it to a set of value 0. Default None.
        max_boundary (Optional [array]): If set to None overides default learner values and sets it to an array of value 1. Default None.
        first_params (Optional [array]): The first parameters to test. If None will just randomly sample the initial condition.
        trust_region (Optional [float or array]): The trust region defines the maximum distance the learner will travel from the current best set of parameters. If None, the learner will search everywhere. If a float, this number must be between 0 and 1 and defines maximum distance the learner will venture as a percentage of the boundaries. If it is an array, it must have the same size as the number of parameters and the numbers define the maximum absolute distance that can be moved along each direction.
    '''

    def __init__(self,
                 trust_region=None,
                 first_params=None,
                 **kwargs):

        super(SimpleRandomLearner,self).__init__(**kwargs)

        if ((np.all(np.isfinite(self.min_boundary))&np.all(np.isfinite(self.max_boundary)))==False):
            msg = 'Minimum and/or maximum boundaries are NaN or inf. Must both be finite for random learner. Min boundary:' + repr(self.min_boundary) +'. Max boundary:' + repr(self.max_boundary)
            self.log.error(msg)
            raise ValueError(msg)
        if first_params is None:
            self.first_params = None
            self.log.debug("First parameters not provided.")
        else:
            self.first_params = np.array(first_params, dtype=float)

            if not self.check_num_params(self.first_params):
                msg = 'first_params has the wrong number of parameters:' + repr(self.first_params)
                self.log.error(msg)
                raise ValueError(msg)
            elif not self.check_in_boundary(self.first_params):
                msg = 'first_params is not in the boundary:' + repr(self.first_params)
                self.log.error(msg)
                raise ValueError(msg)
            else:
                msg = 'first_params are:' + repr(self.first_params)
                self.log.debug(msg)


        # Keep track of best parameters to implement trust region.
        self.best_cost = float('inf')
        self.best_params = None

        self._set_trust_region(trust_region)

        new_values_dict = {
            'archive_type': 'random_learner',
            'trust_region': self.trust_region,
            'has_trust_region': self.has_trust_region,
        }
        self.archive_dict.update(new_values_dict)

        self.log.debug('Simple random learner init completed.')

    def run(self):
        '''
        Puts the next parameters on the queue which are randomly picked from a uniform distribution between the minimum and maximum boundaries when a cost is added to the cost queue.
        '''
        if self.first_params is None:
            self.log.debug('Starting Simple Random Learner with random starting parameters')
            next_params = mlu.rng.uniform(self.min_boundary, self.max_boundary)
        else:
            self.log.debug('Starting Simple Random Learner with provided starting parameters')
            next_params = self.first_params

        while not self.end_event.is_set():

            # Wait until the queue is empty and send a new element promptly.
            while not self.params_out_queue.empty():
                time.sleep(self.learner_wait)

            self.params_out_queue.put(next_params)

            # Clear the costs in queue
            # TODO: update this slightly so we can have a "best values" and therefore
            # implement a trust region.
            try:
                while True:
                    message = self.costs_in_queue.get_nowait()
                    if not all(elem is None for elem in message):
                        self.log.debug(f'Learner got message: {message}')

                        params, cost, uncer, bad = self._parse_cost_message(message)
                        self._update_run_data_attributes(params, cost, uncer, bad)

                        # Update best parameters if necessary.
                        if self.best_cost is None or cost < self.best_cost:
                            self.best_cost = cost
                            self.best_params = self.all_params[-1]
                    else:
                        self.log.info(f'Learner got INVALID message: {message}')

            except queue.Empty:
                pass

            if self.has_trust_region and (self.best_cost != float('inf')) and (self.best_params is not None):
                temp_min = np.maximum(self.min_boundary, self.best_params - self.trust_region)
                temp_max = np.minimum(self.max_boundary, self.best_params + self.trust_region)
                next_params = mlu.rng.uniform(temp_min, temp_max)
            else:
                next_params = mlu.rng.uniform(
                    self.min_boundary,
                    self.max_boundary,
                )

        self._shut_down()
        self.log.debug('Ended Simple Random Learner')
from mloop.learners import Learner
import mloop.utilities as mlu

import threading
import queue
import time
import numpy as np

class RandomLearner(Learner, threading.Thread):
    '''
    Random learner. Simply generates new parameters randomly with a uniform distribution over the boundaries. Learner is perhaps a misnomer 
    for this class.  Its primary job is to make sure that its queue is non-empty.

    Args:
        **kwargs (Optional dict): Other values to be passed to Learner.

    Keyword Args:
        min_boundary (Optional [array]): If set to None, overrides default learner values and sets it to a set of value 0. Default None.
        max_boundary (Optional [array]): If set to None overrides default learner values and sets it to an array of value 1. Default None.
        first_params (Optional [array]): The first parameters to test. If None will just randomly sample the initial condition.
        trust_region (Optional [float or array]): The trust region defines the maximum distance the learner will travel from the 
            parameters defined by trust_range. If None, the learner will search everywhere. If a float, this number must be between 
            0 and 1 and defines maximum distance the learner will venture as a percentage of the boundaries. 
            If it is an array, it must have the same size as the number of parameters and the numbers define 
            the maximum absolute distance that can be moved along each direction.
        trust_range (Optional [array]): define the range over which the learner will apply the trust region.  
            For example trust_range=[0.4, 0.9] will identify the parameters with costs that are between 0.4 and 0.9 of the range from the 
            best and worst.  One of these will be randomly selected and we will then apply a standard trust region about this.  If the range is empty
            we simply select the best.  Setting trust_range = [1,1] will select the best set of parameters.
        trust_gaussian (Optional [bool]): Draw from a gaussian distribution with width defined by trust_region.
        explore_fraction (Optional [float]): fraction of sequences dedicated to exploring. Default 0.0 .
    '''

    def __init__(self,
                 trust_region=None,
                 trust_gaussian=False,
                 trust_range=[0.1, 0.25],
                 first_params=None,
                 explore_fraction=0.0,
                 **kwargs):

        super(RandomLearner, self).__init__(**kwargs)

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


        # Keep track of best and worst parameters to implement trust region.
        self.best_cost = float('inf')
        self.worst_cost = float('-inf')
        self.best_params = None
        self.trust_params =  np.empty( (0,0) )

        self._set_trust_region(trust_region)
        self.trust_gaussian = bool(trust_gaussian)
        self.explore_fraction = float(explore_fraction)

        if len(trust_range) != 2:
            msg = "trust_range must be a 1D vector-like object with length 2"
            self.log.error(msg)
            raise ValueError(msg)
        elif (not 0 <= trust_range[0] <= 1) or (not 0 <= trust_range[1] <= 1):
            msg =f"trust_range parameters must be in the range [0,1]: {trust_range}"
            self.log.error(msg)
            raise ValueError(msg)
        else:
            self.trust_range = sorted(trust_range)

        new_values_dict = {
            'archive_type': 'random_learner',
            'trust_region': self.trust_region,
            'has_trust_region': self.has_trust_region,
        }
        self.archive_dict.update(new_values_dict)

        self.log.debug('Simple random learner init completed.')

    def _next_params(self, new_data):

        explore = self.explore_fraction > mlu.rng.uniform(0.0, 1.0)

        if explore or (self.best_cost == float('inf')) or (self.worst_cost != float('-inf')):
            next_params = mlu.rng.uniform(
                self.min_boundary,
                self.max_boundary,
            )   
        else:
            # If we got new data update the array trust_params to explore near 
            if new_data:
                self.log.debug(f'Learner has trust range: {self.trust_range}')

                # Note that this is minimization problem so self.best_cost < self.worst_cost
                max_trust = self.worst_cost + (self.best_cost - self.worst_cost) * self.trust_range[0]
                min_trust = self.worst_cost + (self.best_cost - self.worst_cost) * self.trust_range[1]
                
                self.trust_params = self.all_params[np.logical_and(min_trust <= self.all_costs, self.all_costs <= max_trust)]
                self.log.debug(f"Have identified {self.trust_params.shape[0]} trust_params for trust_region ({min_trust},{max_trust})")


            if self.has_trust_region and (self.best_params is not None):

                if self.trust_params.shape[0] == 0:
                    self.log.debug("Using best_params for trust_region")
                    nearby_params = self.best_params
                else:
                    self.log.debug("Using trust_params for trust_region")
                    item = mlu.rng.integers(self.trust_params.shape[0])
                    nearby_params = self.trust_params[item,:]

                if self.trust_gaussian:
                    next_params = mlu.rng.normal(nearby_params, self.trust_region)
                    next_params = np.maximum(self.min_boundary, next_params)
                    next_params = np.minimum(self.max_boundary, next_params)
                else:
                    temp_min = np.maximum(self.min_boundary, nearby_params - self.trust_region)
                    temp_max = np.minimum(self.max_boundary, nearby_params + self.trust_region)
                    next_params = mlu.rng.uniform(temp_min, temp_max)
            else:
                next_params = mlu.rng.uniform(
                    self.min_boundary,
                    self.max_boundary,
                )     
        return next_params

    def _clear_cost_queue(self):
        """
        read all data in costs_in_queue and return if valid data was acquired.
        """
        new_data = False
        try:
            while True:
                message = self.costs_in_queue.get_nowait()

                # Check to make sure that the message is valid.
                # TODO: improve _parse_cost_message to deal with invalid messages gracefully instead?
                if not any(elem is None for elem in message):
                    self.log.debug(f'RandomLearner got message: {message}')
                    new_data = True

                    params, cost, uncer, bad = self._parse_cost_message(message)
                    self._update_run_data_attributes(params, cost, uncer, bad)

                    # Update best and worst parameters and cost if necessary.
                    if (self.best_cost is None) or (cost < self.best_cost):
                        self.best_cost = cost
                        self.best_params = self.all_params[-1]

                    if (self.worst_cost is None) or (cost > self.worst_cost):
                        self.worst_cost = cost

                    self.log.debug(f'RandomLearner has best and worst costs: {(self.best_cost, self.worst_cost)}')

                else:
                    self.log.info(f'RandomLearner got INVALID message: {message}')

        except queue.Empty:
            pass
        
        return new_data

    def run(self):
        '''
        Puts the next parameters on the queue which are randomly selected from a uniform distribution between 
        the minimum and maximum boundaries when a cost is added to the cost queue.
        '''
        
        if self.first_params is None:
            self.log.debug('Starting RandomLearner with random starting parameters')
            next_params =  self._next_params(False)
        else:
            self.log.debug('Starting RandomLearner with provided starting parameters')
            next_params = self.first_params

        while not self.end_event.is_set():
            # Wait until the queue is empty and send a new element promptly.
            while not self.params_out_queue.empty():
                time.sleep(self.learner_wait)

            self.params_out_queue.put(next_params)

            # Fully read all costs in the queue
            new_data = self._clear_cost_queue()
            
            next_params = self._next_params(new_data)
        else:
            # make sure that all data is cleared from the queue at graceful termination of while loop
            self._clear_cost_queue()

        self._shut_down()
        self.log.debug('Ended Simple Random Learner')
from mloop.learners import Learner
import mloop.utilities as mlu
import threading
import logging
import queue

logger = logging.getLogger('analysislib_mloop')

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

        super(RandomLearner,self).__init__(**kwargs)

        if ((np.all(np.isfinite(self.min_boundary))&np.all(np.isfinite(self.max_boundary)))==False):
            msg = 'Minimum and/or maximum boundaries are NaN or inf. Must both be finite for random learner. Min boundary:' + repr(self.min_boundary) +'. Max boundary:' + repr(self.max_boundary)
            self.log.error(msg)
            raise ValueError(msg)
        if first_params is None:
            self.first_params = None
        else:
            self.first_params = np.array(first_params, dtype=float)
            if not self.check_num_params(self.first_params):
                msg = 'first_params has the wrong number of parameters:' + repr(self.first_params)
                self.log.error(msg)
                raise ValueError(msg)
            if not self.check_in_boundary(self.first_params):
                msg = 'first_params is not in the boundary:' + repr(self.first_params)
                self.log.error(msg)
                raise ValueError(msg)

        # Keep track of best parameters to implement trust region.
        self.best_cost = None
        self.best_parameters = None

        self._set_trust_region(trust_region)

        new_values_dict = {
            'archive_type': 'random_learner',
            'trust_region': self.trust_region,
            'has_trust_region': self.has_trust_region,
        }
        self.archive_dict.update(new_values_dict)

        self.log.debug('Random learner init completed.')

    def run(self):
        '''
        Puts the next parameters on the queue which are randomly picked from a uniform distribution between the minimum and maximum boundaries when a cost is added to the cost queue.
        '''
        self.log.debug('Starting Simple Random Learner')
        if self.first_params is None:
            next_params = self.min_boundary + nr.rand(self.num_params) * self.diff_boundary
        else:
            next_params = self.first_params

        while not self.end_event.is_set():

            if self.has_trust_region:
                temp_min = np.maximum(self.min_boundary, self.best_params - self.trust_region)
                temp_max = np.minimum(self.max_boundary, self.best_params + self.trust_region)
                next_params = temp_min + nr.rand(self.num_params) * (temp_max - temp_min)
            else:
                next_params =  self.min_boundary + nr.rand(self.num_params) * self.diff_boundary

            # Wait until the queue is empty and send a new element promptly.
            self.params_out_queue.join()
            self.params_out_queue.put(next_params)

            # Clear the costs in queue
            with self.costs_in_queue.mutex:
                self.costs_in_queue.clear()


        self._shut_down()
        self.log.debug('Ended Simple Random Learner')
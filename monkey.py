import os
import sys
import mloop.utilities as mlu

from . import mloop_learner

import logging
logger = logging.getLogger('analysislib_mloop')

import mloop.learners as mll
logger.debug("Monkey patching mll.RandomLearner")
mll.RandomLearner = mloop_learner.RandomLearner
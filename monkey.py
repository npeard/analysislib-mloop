import os
import sys
import mloop.utilities as mlu

from . import mloop_learner

import logging
logger = logging.getLogger('analysislib_mloop')

def _config_logger(log_filename = mlu.default_log_filename,
                  file_log_level=logging.DEBUG,
                  file_log_string='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
                  console_log_level=logging.INFO,
                  console_log_string='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
                  start_datetime=None,
                  **kwargs):
    '''
    Configure and the root logger.
    
    Keyword Args:
        log_filename (Optional [string]) : Filename prefix for log. Default M-LOOP run . If None, no file handler is created
        file_log_level (Optional[int]) : Level of log output for file, default is logging.DEBUG = 10
        console_log_level (Optional[int]) :Level of log output for console, default is logging.INFO = 20
        start_datetime (Optional datetime.datetime): The date and time to use in
            the filename suffix, represented as an instance of the datetime
            class defined in the datetime module. If set to None, then this
            function will use the result returned by datetime.datetime.now().
            Default None.
    
    Returns:
        dictionary: Dict with extra keywords not used by the logging configuration.
    '''    
    log = logging.getLogger('mloop')
    
    if len(log.handlers) == 0:
        log.setLevel(min(file_log_level,console_log_level))
        if log_filename is not None:
            filename_suffix = mlu.generate_filename_suffix('log', start_datetime)
            full_filename = log_filename + filename_suffix
            filename_with_path = os.path.join(mlu.log_foldername, full_filename)
            # Create folder if it doesn't exist, accounting for any parts of the
            # path that may have been included in log_filename.
            actual_log_foldername = os.path.dirname(filename_with_path)
            if not os.path.exists(actual_log_foldername):
                os.makedirs(actual_log_foldername)
            fh = logging.FileHandler(filename_with_path)
            fh.setLevel(file_log_level)
            fh.setFormatter(logging.Formatter(file_log_string))
            log.addHandler(fh)
        ch = logging.StreamHandler(stream = sys.stdout)
        ch.setLevel(console_log_level)
        ch.setFormatter(logging.Formatter(console_log_string))
        log.addHandler(ch)
        log.debug('M-LOOP Logger configured.')
    
    return kwargs

logger.debug("Monkey patching mlu._config_logger")
mlu._config_logger = _config_logger


import mloop.learners as mll
logger.debug("Monkey patching mll.RandomLearner")
mll.RandomLearner = mloop_learner.SimpleRandomLearner
# logger.py
# manage the log for console printing and logging file

import logging
import datetime

class logger(object):
    # Input:
    # root_dir: root folder to save the logging file
    # time: time stamp from upper level classes
    # level: set the level for console stream printing
    def __init__(self, root_dir, time=None, level=logging.DEBUG):
        # Log name and path
        if time is not None:
            time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        else:
            time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_path = root_dir + "/" + time_stamp + ".log"
        log_name = time_stamp
        # get logger object
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level=logging.DEBUG)
        # file stream handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        # console stream handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # logging format
        # formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        # add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    # write log to console and file
    def print(self, text, level=logging.INFO):
        if level == logging.DEBUG:
            self.logger.debug(text)
        elif level == logging.INFO:
            self.logger.info(text)
        elif level == logging.warning:
            self.logger.warning(text)
        elif level == logging.error:
            self.logger.error(text)
        elif level == logging.critical:
            self.logger.critical(text)
import logging
import logging.config
import re
import os

class NoHttpRequestsFilter(logging.Filter):
    """
    Custom filter to exclude log records containing specific patterns.
    """
    def __init__(self, pattern):
        super().__init__()
        self.pattern = re.compile(pattern)
    
    def filter(self, record):
        return not self.pattern.search(record.getMessage())
    

# Custom filter to trim the last 3 characters from the filename
class FilenameTrimFilter(logging.Filter):
    def filter(self, record):
        # Trim the last three characters of the filename
        if record.filename:
            record.filename_trimmed = record.filename[:-3]
        else:
            record.filename_trimmed = record.filename
        return True



def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] [Thread: %(threadName)s] %(message)s',
                'datefmt': '%H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'HandleResponsesFormatter': {
                'format': '\033[1m[%(levelname)s] (%(asctime)s)\033[0m - [%(filename_trimmed)s: %(funcName)s] - [Process: \033[1m%(processName)s\033[0m] - [Thread: \033[1m%(threadName)s\033[0m] : %(message)s',
                'datefmt': '%H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'HandleResponsesHandler': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'HandleResponsesFormatter',
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': 'logs/app.log',
                'mode': 'a',
                'formatter': 'detailed',
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
            'HandleResponses': {
                'handlers': ['HandleResponsesHandler'],
                'level': 'INFO',
                'propagate': False
            },
            'module_b': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False
            },
        },
        'filters': {
            'filename_trim_filter': {
                '()': FilenameTrimFilter,
            }
        }
    }

    # Add the custom filter to each handler that uses the formatter with trimmed filename
    for handler in logging_config['handlers'].values():
        handler['filters'] = ['filename_trim_filter']
        
     # Check if logging is already configured
    if not logging.getLogger().hasHandlers():
        logging.config.dictConfig(logging_config)
    
    logging.config.dictConfig(logging_config)
    
    # Disable HTTP request logging
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

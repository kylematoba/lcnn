import logging
import time


def get_standard_streamhandler(include_time: bool=False):
    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    if include_time:
        logging_fmt = '%(asctime)s %(message)s'
    else:
        logging_fmt = '%(message)s'

    formatter = logging.Formatter(logging_fmt)
    formatter.converter = time.gmtime

    standard_streamhandler = logging.StreamHandler()
    standard_streamhandler.setFormatter(formatter)
    standard_streamhandler.setLevel(logging.INFO)
    return standard_streamhandler


if __name__ == "__main__":
    # standard_streamhandler = get_standard_streamhandler()
    # Logging commands
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_standard_streamhandler())

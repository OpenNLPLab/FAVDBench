import logging


class MyLogger:

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.setup_logging()
        # build logger
        self.logger = logging.getLogger(name)

    def setup_logging(self, resume=False):
        root_logger = logging.getLogger()

        ch = logging.StreamHandler()
        fh = logging.FileHandler(filename=self.path,
                                 mode='a' if resume else 'w')

        root_logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        root_logger.addHandler(ch)
        root_logger.addHandler(fh)

    def info(self, string):
        # if is_main_process():
        self.logger.info(string)

from tqdm.auto import tqdm

import logging


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


handler = TqdmLoggingHandler()
handler.setFormatter(
    logging.Formatter(
        "\033[36m%(asctime)s.%(msecs)03d\033[m "
        "\033[35m%(name)s\033[m "
        "%(message)s",
        "%H:%M:%S",
    )
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False

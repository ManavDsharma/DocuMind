import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging format and level.

    Called once at startup. Using a shared config (instead of ad-hoc
    print statements) means every module's logs are consistently
    formatted and can be redirected/filtered in production.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
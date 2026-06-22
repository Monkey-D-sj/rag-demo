import logging

from rag.common.logging import setup_logging


def main() -> None:
    setup_logging()
    logging.getLogger(__name__).info("rag-demo 启动")


if __name__ == "__main__":
    main()

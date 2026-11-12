"""Точка входа пакета: `python -m zakupki run --data ...`."""

import sys

from zakupki.cli import main

if __name__ == "__main__":
    sys.exit(main())

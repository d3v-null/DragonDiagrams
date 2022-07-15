import imp
import inspect
import os
import sys
import unittest
import logging
from math import sqrt, tan, pi

import numpy as np

THIS_FILE = inspect.stack()[-2].filename
THIS_DIR = os.path.dirname(THIS_FILE)

try:
    PATH = sys.path[:]
    sys.path.insert(0, THIS_DIR)
    import __init__
    imp.reload(__init__)
    from __init__ import TOOLS_DIR, debugTestRunner
    sys.path.insert(0, TOOLS_DIR)
    import common  # type: ignore
    imp.reload(common)
    from common import setup_logger  # type: ignore
finally:
    sys.path = PATH


class TestDragonDiagram(unittest.TestCase):
    def test_foo(self):
        pass


if __name__ == "__main__":
    setup_logger(stream_log_level=logging.DEBUG)
    for suite in [
        unittest.TestLoader().loadTestsFromTestCase(TestDragonDiagram),
    ]:
        debugTestRunner(verbosity=10).run(suite)

import unittest
import shutil


def run_tests(start_dir):
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir)
    runner = unittest.TextTestRunner()
    runner.run(suite)

import sys
run_tests(start_dir=sys.path[0])

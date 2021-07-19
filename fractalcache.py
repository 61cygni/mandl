# --
# File: fractalcache.py
# Sun Jul 18 18:02:48 PDT 2021 
#
# --

import os


FRACTL_CACHE_DIR = "./.fractal_cache/"


class FractalCache:

    def __init__(self, dir = FRACTL_CACHE_DIR):
        self.dir = FRACTL_CACHE_DIR

    def setup(self):
        print("+ using cache %s" % (self.dir))

        if not os.path.isdir(self.dir):
            print("  -> directory not found ...  creating ")
            os.mkdir(self.dir)


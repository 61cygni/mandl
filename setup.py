#!/usr/bin/env python

"""
setup.py  to build mandelbrot code with cython
"""
from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy 


extensions = [
    Extension('cmandelbrot',   ['cmandelbrot.pyx'], include_dirs = [numpy.get_include()]),
    Extension('csmooth',       ['csmooth.pyx'], include_dirs = [numpy.get_include()]),
    Extension('cjulia',        ['cjulia.pyx'], include_dirs = [numpy.get_include()]),
    Extension('hpcsmooth',     ['hpcsmooth.pyx'], include_dirs = [numpy.get_include()]),
    ]

setup(
    ext_modules = cythonize(extensions)
    )

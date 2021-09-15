
"""
Wrapping of custom arb-native impelementation of math functions for fractals.
"""

import sys,os

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy
from numpy.distutils.system_info import default_include_dirs, default_lib_dirs

#flint_include_dirs = [
#print(f"numpy.get_include(): \"{numpy.get_include()}\"")
#print(f"defaults: \"{default_include_dirs}\"")


#default_include_dirs = default_include_dirs + [numpy.get_include()]

mpfr_include_dirs = default_include_dirs.copy()
#arb_include_dirs.append(".")

#for curr_dir in default_include_dirs:
#    arb_include_dirs.append(curr_dir)
#    flint_include_dirs.append(os.path.join(curr_dir, "arb"))

#print(f"arb_include_dirs: \"{arb_include_dirs}\"")

mpfr_library_dirs = default_lib_dirs.copy()
mpfr_library_dirs.append(".") # Trick to make an in-place library be includable

#print(f"arb_library_dirs: \"{arb_library_dirs}\"")

extensions = [
    # The custom fractal library file name is "libmpfrfractalmath.a", 
    # so it's referred to as "mpfrfractalmath" here.
    # The combination of the custom fractal library, and it's cython
    # wrapper, is caled "fractalmath_mpfr", and can be imported by that name
    # in a python script.
    Extension('mpfr_fractalmath', ['mpfr_fractalmath.pyx'], libraries=["mpfr", "mpfrfractalmath"], library_dirs=mpfr_library_dirs, include_dirs=mpfr_include_dirs)
    ]

setup(
    ext_modules = cythonize(extensions, language_level="3")
    )

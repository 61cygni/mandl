
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

arb_include_dirs = default_include_dirs.copy()
#arb_include_dirs.append(".")

#for curr_dir in default_include_dirs:
#    arb_include_dirs.append(curr_dir)
#    flint_include_dirs.append(os.path.join(curr_dir, "arb"))

#print(f"arb_include_dirs: \"{arb_include_dirs}\"")

arb_library_dirs = default_lib_dirs.copy()
arb_library_dirs.append(".") # Trick to make an in-place library be includable

#print(f"arb_library_dirs: \"{arb_library_dirs}\"")

extensions = [
#    Extension('flint_fractal', ['flint_fractal.pyx'], libraries=["flintfractalmath"], library_dirs=['.'], include_dirs=['.'])
    #Extension('flint_fractal', ['flint_fractal.pyx'], libraries=["arb", "flintfractalmath"], library_dirs=flint_library_dirs, include_dirs=flint_include_dirs)

    # The custom fractal library file name is "libarbfractalmath.a", 
    # so it's referred to as "arbfractalmath" here.
    # The combination of the custom fractal library, and it's cython
    # wrapper, is caled "fractalmath_arb", and can be imported by that name
    # in a python script.
    Extension('arb_fractalmath', ['arb_fractalmath.pyx'], libraries=["arb", "arbfractalmath"], library_dirs=arb_library_dirs, include_dirs=arb_include_dirs)
    ]

setup(
    ext_modules = cythonize(extensions, language_level="3")
    )

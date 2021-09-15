# Lazily using flint's arb-from-string function here, though flint
# isn't really required... Should fix that...
# ACTUALLY, maybe flint is required, as an input parameter to start value?
import flint

# Python usage:
# import flint
# import fractalmath_arb
# flintComplexValue = flint.arb(-1.76938, .00423)
# (answer, last_z, remaining_precision) = fractalmath_arb.mandelbrot_steps(flintComplexValue, 2.0, 255);

# Organization Discussion
#
# arbfractalmath.h and arbfractalmath.h
# - compile into arbfractalmath.a
#
# arb_fractalmath.pyx
# - compiles into fractalmath_arb.so module via setup.py, invoked from makefile
# arb_fractalmath.pyx (defines python interface for arbfractalmath.a)
#  - needs to accept python-flint arguments
#  - needs to invoke arb_fractalmath's arb_mandelbrot_steps() with appropriate params

cdef extern from "arb_fractal_lib.h":
    void arb_mandelbrot_steps(long *result, char **last_z_real_str, char **last_z_imag_str, long *remaining_precision, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec)

def mandelbrot_steps(s, radius, max_iter):                                     
    cdef long answer = 0
    cdef long remaining_precision = 0

    cdef char *last_z_real_str = NULL
    cdef char *last_z_imag_str = NULL

    # Shuffle to keep temporary strings from becoming parameters
    # Techincally, this captures the 'bytes' from the .encode()
    # call, so the assignment to a cython char* is allowed.
    cdef char *start_real_str = NULL
    cdef char *start_imag_str = NULL
    start_real_py_str = s.real.str(more=True).encode('ascii')
    start_imag_py_str = s.imag.str(more=True).encode('ascii')
    start_real_str = start_real_py_str
    start_imag_str = start_imag_py_str

    arb_mandelbrot_steps(&answer, &last_z_real_str, &last_z_imag_str, &remaining_precision, start_real_str, start_imag_str, float(radius), max_iter, flint.ctx.prec)

    # Extra step for string conversion back from bytes to python string
    answer_real_py_str = last_z_real_str.decode('ascii')
    answer_imag_py_str = last_z_imag_str.decode('ascii')
    last_z = flint.acb(answer_real_py_str, answer_imag_py_str)

    # TODO: Almost certainly need to free last_z_real_str 
    # and last_z_imag_str, right?

    return (answer, last_z, remaining_precision) 



import numpy as np
cimport numpy as np

import decimal

cdef extern from "mpfr_fractal_lib.h":
    void mpfr_mandelbrot_steps(long *result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec)
    void mpfr_julia_steps(long *result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, const char *julia_real_str, const char *julia_imag_str, float radius, const long max_iter, long prec)

    void mpfr_mandelbrot_distance_estimate(long *result, long double *distance_result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec)
    void mpfr_julia_distance_estimate(long *result, long double *distance_result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, const char *julia_real_str, const char *julia_imag_str, float radius, const long max_iter, long prec)

def mandelbrot_steps(string_real, string_imag, escape_radius, max_iterations, precision):
    # Temporaries used for every call
    cdef long answer = 0
    cdef char *last_z_real_str = NULL
    cdef char *last_z_imag_str = NULL

    cdef char *start_real_str = NULL
    cdef char *start_imag_str = NULL

    # Shuffle to keep temporary strings from becoming parameters
    # Techincally, this captures the 'bytes' from the .encode()
    # call, so the assignment to a cython char* is allowed.
    start_real_py_str = string_real.encode('ascii')
    start_imag_py_str = string_imag.encode('ascii')
    start_real_str = start_real_py_str
    start_imag_str = start_imag_py_str

    #print(f"param's start_real_str \"{start_real_str}\"")
    #print(f"param's start_imag_str \"{start_imag_str}\"")
    mpfr_mandelbrot_steps(&answer, &last_z_real_str, &last_z_imag_str, start_real_str, start_imag_str, float(escape_radius), max_iterations, precision)

    # Extra step for string conversion back from bytes to python string
    #print(f"last_z_real_str \"{last_z_real_str}\"")
    #print(f"last_z_imag_str \"{last_z_imag_str}\"")
    last_z_real = last_z_real_str.decode('ascii')
    last_z_imag = last_z_imag_str.decode('ascii')

    # TODO: Almost certainly need to free last_z_real_str 
    # and last_z_imag_str, right?

            #print(f"answer {answer}")

    return (answer, last_z_real, last_z_imag)

def mandelbrot_2d_string_to_string(string_reals, string_imags, escape_radius, max_iterations, precision):
    if string_reals.shape != string_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = string_reals.shape # (rows, columns)

    # Python-create the numpy arrays.  Originally thought we'd then
    # cython-assign them to be used as memoryviews, but it's gnarly
    # for strings, and really doesn't save much overall.
    count_results = np.zeros(answer_shape, dtype=float)
    last_z_reals = np.empty(answer_shape, dtype=object)
    last_z_imags = np.empty(answer_shape, dtype=object)

    # Temporaries used for every call
    cdef long answer = 0
    cdef char *last_z_real_str = NULL
    cdef char *last_z_imag_str = NULL

    cdef char *start_real_str = NULL
    cdef char *start_imag_str = NULL

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            # Shuffle to keep temporary strings from becoming parameters
            # Techincally, this captures the 'bytes' from the .encode()
            # call, so the assignment to a cython char* is allowed.
            start_real_py_str = string_reals[x,y].encode('ascii')
            start_imag_py_str = string_imags[x,y].encode('ascii')
            start_real_str = start_real_py_str
            start_imag_str = start_imag_py_str

            #print(f"param's start_real_str \"{start_real_str}\"")
            #print(f"param's start_imag_str \"{start_imag_str}\"")
            mpfr_mandelbrot_steps(&answer, &last_z_real_str, &last_z_imag_str, start_real_str, start_imag_str, float(escape_radius), max_iterations, precision)

            # Extra step for string conversion back from bytes to python string
            #print(f"last_z_real_str \"{last_z_real_str}\"")
            #print(f"last_z_imag_str \"{last_z_imag_str}\"")
            last_z_reals[x,y] = last_z_real_str.decode('ascii')
            last_z_imags[x,y] = last_z_imag_str.decode('ascii')
 
            count_results[x,y] = answer
            # TODO: Almost certainly need to free last_z_real_str 
            # and last_z_imag_str, right?

            #print(f"answer {answer}")

    return (count_results, last_z_reals, last_z_imags)

def mandelbrot_2d_pydecimal_to_string(decimal_reals, decimal_imags, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    # TODO: Quite possibly want to set decimal's precision here?

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
  
    return mandelbrot_2d_string_to_string(string_reals, string_imags, escape_radius, max_iterations, precision)

def mandelbrot_2d_pydecimal_to_decimal(decimal_reals, decimal_imags, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
 
    (count_results, string_last_z_reals, string_last_z_imags) =  mandelbrot_2d_string_to_string(string_reals, string_imags, escape_radius, max_iterations, precision)

    decimal_last_z_reals = np.empty(answer_shape, dtype=object)
    decimal_last_z_imags = np.empty(answer_shape, dtype=object)
    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            decimal_last_z_reals[x,y] = decimal.Decimal(string_last_z_reals[x,y])
            decimal_last_z_imags[x,y] = decimal.Decimal(string_last_z_imags[x,y])

    return (count_results, decimal_last_z_reals, decimal_last_z_imags)
 
def julia_2d_string_to_string(string_reals, string_imags, string_julia_real, string_julia_imag, escape_radius, max_iterations, precision):
    if string_reals.shape != string_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = string_reals.shape # (rows, columns)

    # Python-create the numpy arrays.  Originally thought we'd then
    # cython-assign them to be used as memoryviews, but it's gnarly
    # for strings, and really doesn't save much overall.
    count_results = np.zeros(answer_shape, dtype=float)
    last_z_reals = np.empty(answer_shape, dtype=object)
    last_z_imags = np.empty(answer_shape, dtype=object)

    # Temporaries used for every call
    cdef long answer = 0
    cdef char *last_z_real_str = NULL
    cdef char *last_z_imag_str = NULL

    cdef char *start_real_str = NULL
    cdef char *start_imag_str = NULL

    cdef char *julia_real_str = NULL
    cdef char *julia_imag_str = NULL
    julia_real_py_str = string_julia_real.encode('ascii')
    julia_imag_py_str = string_julia_imag.encode('ascii')
    julia_real_str = julia_real_py_str
    julia_imag_str = julia_imag_py_str

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            # Shuffle to keep temporary strings from becoming parameters
            # Techincally, this captures the 'bytes' from the .encode()
            # call, so the assignment to a cython char* is allowed.
            start_real_py_str = string_reals[x,y].encode('ascii')
            start_imag_py_str = string_imags[x,y].encode('ascii')
            start_real_str = start_real_py_str
            start_imag_str = start_imag_py_str

            #print(f"param's start_real_str \"{start_real_str}\"")
            #print(f"param's start_imag_str \"{start_imag_str}\"")
            mpfr_julia_steps(&answer, &last_z_real_str, &last_z_imag_str, start_real_str, start_imag_str, julia_real_str, julia_imag_str, float(escape_radius), max_iterations, precision)

            # Extra step for string conversion back from bytes to python string
            #print(f"last_z_real_str \"{last_z_real_str}\"")
            #print(f"last_z_imag_str \"{last_z_imag_str}\"")
            last_z_reals[x,y] = last_z_real_str.decode('ascii')
            last_z_imags[x,y] = last_z_imag_str.decode('ascii')
 
            count_results[x,y] = answer
            # TODO: Almost certainly need to free last_z_real_str 
            # and last_z_imag_str, right?

            #print(f"answer {answer}")

    return (count_results, last_z_reals, last_z_imags)

def julia_2d_pydecimal_to_string(decimal_reals, decimal_imags, decimal_julia_real, decimal_julia_imag, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    # TODO: Quite possibly want to set decimal's precision here?

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
  
    return julia_2d_string_to_string(string_reals, string_imags, str(decimal_julia_real), str(decimal_julia_imag), escape_radius, max_iterations, precision)

def julia_2d_pydecimal_to_decimal(decimal_reals, decimal_imags, decimal_julia_real, decimal_julia_imag, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
 
    (count_results, string_last_z_reals, string_last_z_imags) =  julia_2d_string_to_string(string_reals, string_imags, str(decimal_julia_real), str(decimal_julia_imag), escape_radius, max_iterations, precision)

    decimal_last_z_reals = np.empty(answer_shape, dtype=object)
    decimal_last_z_imags = np.empty(answer_shape, dtype=object)
    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            decimal_last_z_reals[x,y] = decimal.Decimal(string_last_z_reals[x,y])
            decimal_last_z_imags[x,y] = decimal.Decimal(string_last_z_imags[x,y])

    return (count_results, decimal_last_z_reals, decimal_last_z_imags)

def mandelbrot_distance_2d_string_to_string(string_reals, string_imags, escape_radius, max_iterations, precision):
    if string_reals.shape != string_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = string_reals.shape # (rows, columns)

    # Python-create the numpy arrays.  Originally thought we'd then
    # cython-assign them to be used as memoryviews, but it's gnarly
    # for strings, and really doesn't save much overall.
    count_results = np.zeros(answer_shape, dtype=float)
    smooth_results = np.zeros(answer_shape, dtype=float)
    last_z_reals = np.empty(answer_shape, dtype=object)
    last_z_imags = np.empty(answer_shape, dtype=object)

    # Temporaries used for every call
    cdef long answer = 0
    cdef long double smooth_answer = 0.0
    cdef char *last_z_real_str = NULL
    cdef char *last_z_imag_str = NULL

    cdef char *start_real_str = NULL
    cdef char *start_imag_str = NULL

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            # Shuffle to keep temporary strings from becoming parameters
            # Techincally, this captures the 'bytes' from the .encode()
            # call, so the assignment to a cython char* is allowed.
            start_real_py_str = string_reals[x,y].encode('ascii')
            start_imag_py_str = string_imags[x,y].encode('ascii')
            start_real_str = start_real_py_str
            start_imag_str = start_imag_py_str

            #print(f"param's start_real_str \"{start_real_str}\"")
            #print(f"param's start_imag_str \"{start_imag_str}\"")
            mpfr_mandelbrot_distance_estimate(&answer, &smooth_answer, &last_z_real_str, &last_z_imag_str, start_real_str, start_imag_str, float(escape_radius), max_iterations, precision)

            # Extra step for string conversion back from bytes to python string
            #print(f"last_z_real_str \"{last_z_real_str}\"")
            #print(f"last_z_imag_str \"{last_z_imag_str}\"")
            last_z_reals[x,y] = last_z_real_str.decode('ascii')
            last_z_imags[x,y] = last_z_imag_str.decode('ascii')
 
            count_results[x,y] = answer
            smooth_results[x,y] = smooth_answer 
            # TODO: Almost certainly need to free last_z_real_str 
            # and last_z_imag_str, right?

            #print(f"answer {answer}")

    return (count_results, smooth_results, last_z_reals, last_z_imags)

def mandelbrot_distance_2d_pydecimal_to_string(decimal_reals, decimal_imags, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    # TODO: Quite possibly want to set decimal's precision here?

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
  
    return mandelbrot_distance_2d_string_to_string(string_reals, string_imags, escape_radius, max_iterations, precision)

def mandelbrot_distance_2d_pydecimal_to_decimal(decimal_reals, decimal_imags, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
 
    (count_results, smooth_results, string_last_z_reals, string_last_z_imags) =  mandelbrot_distance_2d_string_to_string(string_reals, string_imags, escape_radius, max_iterations, precision)

    decimal_last_z_reals = np.empty(answer_shape, dtype=object)
    decimal_last_z_imags = np.empty(answer_shape, dtype=object)
    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            decimal_last_z_reals[x,y] = decimal.Decimal(string_last_z_reals[x,y])
            decimal_last_z_imags[x,y] = decimal.Decimal(string_last_z_imags[x,y])

    return (count_results, smooth_results, decimal_last_z_reals, decimal_last_z_imags)

def julia_distance_2d_string_to_string(string_reals, string_imags, string_julia_real, string_julia_imag, escape_radius, max_iterations, precision):
    if string_reals.shape != string_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = string_reals.shape # (rows, columns)

    # Python-create the numpy arrays.  Originally thought we'd then
    # cython-assign them to be used as memoryviews, but it's gnarly
    # for strings, and really doesn't save much overall.
    count_results = np.zeros(answer_shape, dtype=float)
    smooth_results = np.zeros(answer_shape, dtype=float)
    last_z_reals = np.empty(answer_shape, dtype=object)
    last_z_imags = np.empty(answer_shape, dtype=object)

    # Temporaries used for every call
    cdef long answer = 0
    cdef long double smooth_answer = 0.0
    cdef char *last_z_real_str = NULL
    cdef char *last_z_imag_str = NULL

    cdef char *start_real_str = NULL
    cdef char *start_imag_str = NULL

    cdef char *julia_real_str = NULL
    cdef char *julia_imag_str = NULL
    julia_real_py_str = string_julia_real.encode('ascii')
    julia_imag_py_str = string_julia_imag.encode('ascii')
    julia_real_str = julia_real_py_str
    julia_imag_str = julia_imag_py_str

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            # Shuffle to keep temporary strings from becoming parameters
            # Techincally, this captures the 'bytes' from the .encode()
            # call, so the assignment to a cython char* is allowed.
            start_real_py_str = string_reals[x,y].encode('ascii')
            start_imag_py_str = string_imags[x,y].encode('ascii')
            start_real_str = start_real_py_str
            start_imag_str = start_imag_py_str

            #print(f"param's start_real_str \"{start_real_str}\"")
            #print(f"param's start_imag_str \"{start_imag_str}\"")
            mpfr_julia_distance_estimate(&answer, &smooth_answer, &last_z_real_str, &last_z_imag_str, start_real_str, start_imag_str, julia_real_str, julia_imag_str, float(escape_radius), max_iterations, precision)

            # Extra step for string conversion back from bytes to python string
            #print(f"last_z_real_str \"{last_z_real_str}\"")
            #print(f"last_z_imag_str \"{last_z_imag_str}\"")
            last_z_reals[x,y] = last_z_real_str.decode('ascii')
            last_z_imags[x,y] = last_z_imag_str.decode('ascii')
 
            count_results[x,y] = answer
            smooth_results[x,y] = smooth_answer
            # TODO: Almost certainly need to free last_z_real_str 
            # and last_z_imag_str, right?

            #print(f"answer {answer}")

    return (count_results, smooth_results, last_z_reals, last_z_imags)

def julia_distance_2d_pydecimal_to_string(decimal_reals, decimal_imags, decimal_julia_real, decimal_julia_imag, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    # TODO: Quite possibly want to set decimal's precision here?

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
  
    return julia_distance_2d_string_to_string(string_reals, string_imags, str(decimal_julia_real), str(decimal_julia_imag), escape_radius, max_iterations, precision)

def julia_distance_2d_pydecimal_to_decimal(decimal_reals, decimal_imags, decimal_julia_real, decimal_julia_imag, escape_radius, max_iterations, precision):
    if decimal_reals.shape != decimal_imags.shape:
        raise ValueError("Array of real string shape doesn't match array of imag string shape.")

    answer_shape = decimal_reals.shape # (rows, columns)
    string_reals = np.empty(answer_shape, dtype=object)
    string_imags = np.empty(answer_shape, dtype=object)

    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            string_reals[x,y] = str(decimal_reals[x,y])
            string_imags[x,y] = str(decimal_imags[x,y])
 
    (count_results, smooth_results, string_last_z_reals, string_last_z_imags) =  julia_2d_string_to_string(string_reals, string_imags, str(decimal_julia_real), str(decimal_julia_imag), escape_radius, max_iterations, precision)

    decimal_last_z_reals = np.empty(answer_shape, dtype=object)
    decimal_last_z_imags = np.empty(answer_shape, dtype=object)
    for y in range(answer_shape[1]):
        for x in range(answer_shape[0]):
            decimal_last_z_reals[x,y] = decimal.Decimal(string_last_z_reals[x,y])
            decimal_last_z_imags[x,y] = decimal.Decimal(string_last_z_imags[x,y])

    return (count_results, smooth_results, decimal_last_z_reals, decimal_last_z_imags)

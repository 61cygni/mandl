
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mpfr.h"

void mpfr_mandelbrot_steps(long *result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec);

void mpfr_julia_steps(long *result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, const char *julia_real_str, const char *julia_imag_str, float radius, const long max_iter, long prec);

// Notably, the type of 'result' is different for distance estimation, 
// because the extra smoothing calculation is handled in-place where the
// derivative is available.
// Also, can't just call 'julia' from 'mandelbrot' for distance estimate,
// because the derivative iteration is subtly different.
void mpfr_mandelbrot_distance_estimate(long *result, long double *distance_result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec);

void mpfr_julia_distance_estimate(long *result, long double *distance_result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, const char *julia_real_str, const char *julia_imag_str, float radius, const long max_iter, long prec);

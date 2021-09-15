
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "arb.h"
#include "acb.h"

void arb_mandelbrot_steps(long *result, char **last_z_real_str, char **last_z_imag_str, long *remaining_precision, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec);


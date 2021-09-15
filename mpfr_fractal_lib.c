
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "mpfr.h"

#define max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})

#define min(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})


void mpfr_mandelbrot_steps(long *result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec)
{
    // Step-wise algorithm modified from: 
    // https://randomascii.wordpress.com/2011/08/13/faster-fractals-through-algebra/
    // Calculates (zreal + zimag)^2 to get
    // zr^2 + 2*zr*zi + zi^2
    // Which is helpful, because for the iteration, zr^2 and zi^2 
    // were already calculated.
    // So subtract them, and the remaining term is just 2*zr*zi.

    // Big blocks of declarations, hang-overs from cython 
    // old-style C declaration rules.
    mpfr_t temp;
    mpfr_t zr_squared;
    mpfr_t zi_squared;
    mpfr_t temp_magnitude;
    mpfr_t temp_sum_a;
    mpfr_t temp_sub_a;
    mpfr_t rad_squared; // Can't mix types, so need an arb for radius

    mpfr_t start_real;
    mpfr_t start_imag;
    mpfr_t last_z_real;
    mpfr_t last_z_imag;
 
    //bool precisionExceeded;
    //long bitsValue;

    //mpfr_exp_t print_exp;
    long print_buffer_size = 65536;
    char print_buffer[print_buffer_size];
    //char last_z_real_buffer[print_buffer_size];
    //char last_z_imag_buffer[print_buffer_size];

    mpfr_set_default_prec(prec);

    mpfr_init(temp);
    mpfr_init(zr_squared);
    mpfr_init(zi_squared);
    mpfr_init(temp_magnitude);
    mpfr_init(temp_sum_a);
    mpfr_init(temp_sub_a);
    mpfr_init(rad_squared);

    mpfr_init2(start_real, prec);
    mpfr_init(start_imag);
    mpfr_init(last_z_real);
    mpfr_init(last_z_imag);
 
    mpfr_set_str(start_real, start_real_str, 10, MPFR_RNDN); // 10 = base, not precision
    mpfr_set_str(start_imag, start_imag_str, 10, MPFR_RNDN);

    //printf("start_real_str \"%s\"\n", start_real_str);
    //printf("start_imag_str \"%s\"\n", start_imag_str);

    //printf("start_real: ");
    //mpfr_out_str(stdout, 10,100,start_real, MPFR_RNDN);
    //printf("\nstart_imag: ");
    //mpfr_out_str(stdout, 10,100,start_imag, MPFR_RNDN);
    //printf("\n");
    //fflush(stdout);


    // Magic conversion for converting prec to dps... roughly.
    // (needed for back-to-string conversions)
    //long digits_precision = max(1, round(prec/3.3219280948873626)-1);
    //int max_exponent_length = 9;

    // r^2
    mpfr_set_d(rad_squared, (double)(radius * radius), MPFR_RNDN); 

//    #print("starting stepping c accuracy:")
//    #bitsValue = acb_rel_accuracy_bits(c)
//    #print("bits")
//    #print(bitsValue)
//
//    # Apperently important not to use an expression as parameter
//    # in place of prec, but to compute the value here.
//    #prec = precin + 10

    /* initialize Z as zero */
 
    mpfr_set_zero(last_z_real,1);
    mpfr_set_zero(last_z_imag,1);
    mpfr_set_zero(zr_squared,1);
    mpfr_set_zero(zi_squared,1); 

    //precisionExceeded = false;

    // Since the zero iteration is just setting the starting values, it's not 
    // counted in the iterations count (so loop until *equal to* max_iter).
    for(long currIter = 0; currIter <= max_iter; currIter++)
    {
        *result = currIter;
        mpfr_add(temp_magnitude, zr_squared, zi_squared, MPFR_RNDN);

        //mpfr_sprintf(print_buffer, "%Re", temp_magnitude);
        //printf("%s\n", print_buffer);
        //fflush(stdout);

        if(mpfr_cmp(temp_magnitude, rad_squared) > 0)
        {
            break;
        } 

        mpfr_add(temp_sum_a, last_z_real, last_z_imag, MPFR_RNDN);
        mpfr_mul(temp, temp_sum_a, temp_sum_a, MPFR_RNDN);
        mpfr_sub(temp_sub_a, temp, zr_squared, MPFR_RNDN);
        mpfr_sub(temp, temp_sub_a, zi_squared, MPFR_RNDN);

        mpfr_add(last_z_imag, temp, start_imag, MPFR_RNDN);

        mpfr_sub(temp_sub_a, zr_squared, zi_squared, MPFR_RNDN);
        mpfr_add(last_z_real, temp_sub_a, start_real, MPFR_RNDN);

        mpfr_mul(zr_squared, last_z_real, last_z_real, MPFR_RNDN);
        mpfr_mul(zi_squared, last_z_imag, last_z_imag, MPFR_RNDN);
    }

    // Print the value to the print buffer, then create a new string
    // of appropriate length from there?
    //mpfr_snprintf(print_buffer, digits_precision + max_exponent_length, "%Re", last_z_real);
    mpfr_sprintf(print_buffer, "%Re", last_z_real);
    *last_z_real_str = strdup(print_buffer);
    //mpfr_snprintf(print_buffer, digits_precision + max_exponent_length, "%Re", last_z_imag);
    mpfr_sprintf(print_buffer, "%Re", last_z_imag);
    *last_z_imag_str = strdup(print_buffer);

    mpfr_clear(temp);
    mpfr_clear(zr_squared);
    mpfr_clear(zi_squared);
    mpfr_clear(temp_magnitude);
    mpfr_clear(temp_sum_a);
    mpfr_clear(temp_sub_a);
    mpfr_clear(rad_squared);

    mpfr_clear(start_real);
    mpfr_clear(start_imag);
    mpfr_clear(last_z_real);
    mpfr_clear(last_z_imag);
}



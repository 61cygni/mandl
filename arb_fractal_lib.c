
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "arb.h"
#include "acb.h"

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

void arb_mandelbrot_steps(long *result, char **last_z_real_str, char **last_z_imag_str, long *remaining_precision, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec)
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
    arb_t temp;
    arb_t zr_squared;
    arb_t zi_squared;
    arb_t temp_magnitude;
    arb_t temp_sum_a;
    arb_t temp_sub_a;
    arb_t rad_squared; // Can't mix types, so need an arb for radius

    arb_t start_real;
    arb_t start_imag;
    arb_t last_z_real;
    arb_t last_z_imag;

    bool precisionExceeded;
    long bitsValue;
    
    arb_init(temp);
    arb_init(zr_squared);
    arb_init(zi_squared);
    arb_init(temp_magnitude);
    arb_init(temp_sum_a);
    arb_init(temp_sub_a);
    arb_init(rad_squared);

    arb_init(start_real);
    arb_init(start_imag);
    arb_init(last_z_real);
    arb_init(last_z_imag);

    arb_set_str(start_real, start_real_str, prec);
    arb_set_str(start_imag, start_imag_str, prec);
    //printf("start_real \"%s\"\n", start_real_str);
    //printf("start_imag \"%s\"\n", start_imag_str);

    // Magic conversion for converting prec to dps... roughly.
    // (needed for back-to-string conversions)
    long digits_precision = max(1, round(prec/3.3219280948873626)-1);

    // r^2
    arb_set_d(rad_squared, (double)(radius * radius)); 

//    #print("starting stepping c accuracy:")
//    #bitsValue = acb_rel_accuracy_bits(c)
//    #print("bits")
//    #print(bitsValue)
//
//    # Apperently important not to use an expression as parameter
//    # in place of prec, but to compute the value here.
//    #prec = precin + 10

    /* initialize Z as zero */
    arb_zero(last_z_real);
    arb_zero(last_z_imag);
    arb_zero(zr_squared);
    arb_zero(zi_squared);

    precisionExceeded = false;

    // Since the zero iteration is just setting the starting values, it's not 
    // counted in the iterations count (so loop until *equal to* max_iter).
    for(long currIter = 0; currIter <= max_iter; currIter++)
    {
        //char * curr_real = arb_get_str(acb_realref(last_z), digits_precision, 0);
        //char * curr_imag = arb_get_str(acb_imagref(last_z), digits_precision, 0);
        //printf("curr real \"%s\"\n", curr_real);
        //printf("curr imag \"%s\"\n\n", curr_imag);

        *result = currIter;
        arb_add(temp_magnitude, zr_squared, zi_squared, prec);
        //#print(arb_get_str(temp_magnitude, 0, 0));

        if(arb_gt(temp_magnitude, rad_squared))
        {
            break;
        } 


        arb_add(temp_sum_a, last_z_real, last_z_imag, prec);
        arb_mul(temp, temp_sum_a, temp_sum_a, prec);
        arb_sub(temp_sub_a, temp, zr_squared, prec);
        arb_sub(temp, temp_sub_a, zi_squared, prec);

        arb_add(last_z_imag, temp, start_imag, prec);

        arb_sub(temp_sub_a, zr_squared, zi_squared, prec);
        arb_add(last_z_real, temp_sub_a, start_real, prec);

        arb_mul(zr_squared, last_z_real, last_z_real, prec);
        arb_mul(zi_squared, last_z_imag, last_z_imag, prec);

//        # This might be it!
//        # Forcibly reset error to zero, when we've exceeded the
//        # available precision
//        #
//        # I'd rather not zero this out, but I guess we might
//        # as well, if it keeps stability of answers longer?
        bitsValue = min(arb_rel_accuracy_bits(last_z_real), arb_rel_accuracy_bits(last_z_imag));
        //bitsValue = acb_rel_accuracy_bits(last_z);
//        #if warningPrinted == False and bitsValue < 1:
//        #    print("Exceeded precision at iteration %d (%d)." % (currIter, bitsValue))
//        #    warningPrinted = True
//        # TODO: should be a 'min_precision' param, not a magic number, right?
        if(bitsValue < 4) 
        {
            precisionExceeded = true;

            mag_zero(arb_radref(last_z_real));
            mag_zero(arb_radref(last_z_imag));
            mag_zero(arb_radref(zr_squared));
            mag_zero(arb_radref(zi_squared));
        }
    }

    if(precisionExceeded == true)
    {
        *remaining_precision = 0;

        mag_zero(arb_radref(last_z_real));
        mag_zero(arb_radref(last_z_imag));
        mag_zero(arb_radref(zr_squared));
        mag_zero(arb_radref(zi_squared));
    }
    else
    {
        *remaining_precision = min(arb_rel_accuracy_bits(last_z_real), arb_rel_accuracy_bits(last_z_imag));
    }

    // Ask for 'more', knowing the trailing digits may be imprecise?
    *last_z_real_str = arb_get_str(last_z_real, digits_precision, ARB_STR_MORE);
    *last_z_imag_str = arb_get_str(last_z_imag, digits_precision, ARB_STR_MORE);

    arb_clear(temp);
    arb_clear(zr_squared);
    arb_clear(zi_squared);
    arb_clear(temp_magnitude);
    arb_clear(temp_sum_a);
    arb_clear(temp_sub_a);
    arb_clear(rad_squared);

    arb_clear(start_real);
    arb_clear(start_imag);
    arb_clear(last_z_real);
    arb_clear(last_z_imag);
}



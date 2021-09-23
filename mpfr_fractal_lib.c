
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "mpfr_fractal_lib.h"

#include "mpfr.h"

void mpfr_mandelbrot_steps(long *result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec)
{
    // Mandelbrot is just a special zero-initial case of julia, so we can share
    // a single implementation.
    const char *zero_string = "0.0";
    mpfr_julia_steps(result, last_z_real_str, last_z_imag_str, start_real_str, start_imag_str, zero_string, zero_string, radius, max_iter, prec);
/* 
    // Step-wise algorithm modified from: 
    // https://randomascii.wordpress.com/2011/08/13/faster-fractals-through-algebra/
    // Calculates (zreal + zimag)^2 to get
    // zr^2 + 2*zr*zi + zi^2
    // Which is helpful, because for the iteration, zr^2 and zi^2 
    // were already calculated.
    // So subtract them, and the remaining term is just 2*zr*zi.

    mpfr_t temp;
    mpfr_t zr_squared;
    mpfr_t zi_squared;
    mpfr_t temp_magnitude;
    mpfr_t temp_sum_a;
    mpfr_t temp_sub_a;
    mpfr_t rad_squared;

    mpfr_t start_real;
    mpfr_t start_imag;
    mpfr_t last_z_real;
    mpfr_t last_z_imag;
 
    long print_buffer_size = 65536;
    char print_buffer[print_buffer_size];

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

    // r^2
    mpfr_set_d(rad_squared, (double)(radius * radius), MPFR_RNDN); 

    // initialize Z as zero
    mpfr_set_zero(last_z_real,1);
    mpfr_set_zero(last_z_imag,1);
    mpfr_set_zero(zr_squared,1);
    mpfr_set_zero(zi_squared,1); 

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

        // Looks like ~4 percent increase in speed by using mpfr_sqr where
        // possible, instead of mpfr_mul (which is interesting, because
        // arb_sqr just redirects to mul).
        //
        // The mpfr docs claim in-place operations work too (where I'm pretty
        // sure I had bugs trying to use arb_add(a,a,b) functions), and
        // should be preferred over temporary values... but I haven't run
        // any comparisons on that yet, so don't know?
        mpfr_add(temp_sum_a, last_z_real, last_z_imag, MPFR_RNDN);
        mpfr_sqr(temp, temp_sum_a, MPFR_RNDN);
        mpfr_sub(temp_sub_a, temp, zr_squared, MPFR_RNDN);
        mpfr_sub(temp, temp_sub_a, zi_squared, MPFR_RNDN);

        mpfr_add(last_z_imag, temp, start_imag, MPFR_RNDN);

        mpfr_sub(temp_sub_a, zr_squared, zi_squared, MPFR_RNDN);
        mpfr_add(last_z_real, temp_sub_a, start_real, MPFR_RNDN);

        mpfr_sqr(zr_squared, last_z_real, MPFR_RNDN);
        mpfr_sqr(zi_squared, last_z_imag, MPFR_RNDN);
    }

    // Print the value to the print buffer, then create a new string
    // of appropriate length from there?
    mpfr_sprintf(print_buffer, "%Re", last_z_real);
    *last_z_real_str = strdup(print_buffer);
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
*/
}

void mpfr_julia_steps(long *result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, const char *julia_real_str, const char *julia_imag_str, float radius, const long max_iter, long prec)
{
    // Step-wise algorithm modified from: 
    // https://randomascii.wordpress.com/2011/08/13/faster-fractals-through-algebra/
    // Calculates (zreal + zimag)^2 to get
    // zr^2 + 2*zr*zi + zi^2
    // Which is helpful, because for the iteration, zr^2 and zi^2 
    // were already calculated.
    // So subtract them, and the remaining term is just 2*zr*zi.

    mpfr_t temp;
    mpfr_t zr_squared;
    mpfr_t zi_squared;
    mpfr_t temp_magnitude;
    mpfr_t temp_sum_a;
    mpfr_t temp_sub_a;
    mpfr_t rad_squared; 

    mpfr_t start_real;
    mpfr_t start_imag;
    mpfr_t last_z_real;
    mpfr_t last_z_imag;
 
    long print_buffer_size = 65536;
    char print_buffer[print_buffer_size];

    mpfr_set_default_prec(prec);

    mpfr_init(temp);
    mpfr_init(zr_squared);
    mpfr_init(zi_squared);
    mpfr_init(temp_magnitude);
    mpfr_init(temp_sum_a);
    mpfr_init(temp_sub_a);
    mpfr_init(rad_squared);

    mpfr_init(start_real);
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

    // r^2
    mpfr_set_d(rad_squared, (double)(radius * radius), MPFR_RNDN); 

    // Julia init is to the julia value, which means squares should be initialized too. 
    mpfr_set_str(last_z_real, julia_real_str, 10, MPFR_RNDN); // 10 = base, not precision
    mpfr_set_str(last_z_imag, julia_imag_str, 10, MPFR_RNDN);
    mpfr_sqr(zr_squared, last_z_real, MPFR_RNDN);
    mpfr_sqr(zi_squared, last_z_imag, MPFR_RNDN);

    // Since the zero iteration is just setting the starting values, it's not 
    // counted in the iterations count (so loop until *equal to* max_iter).
    // This *seems* to still be reasonable for Julia iteration, even though the
    // bit about 'just setting the starting values' doesn't seem as correct.
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

        // Looks like ~4 percent increase in speed by using mpfr_sqr where
        // possible, instead of mpfr_mul (which is interesting, because
        // arb_sqr just redirects to mul).
        //
        // The mpfr docs claim in-place operations work too (where I'm pretty
        // sure I had bugs trying to use arb_add(a,a,b) functions), and
        // should be preferred over temporary values... but I haven't run
        // any comparisons on that yet, so don't know?
        mpfr_add(temp_sum_a, last_z_real, last_z_imag, MPFR_RNDN);
        mpfr_sqr(temp, temp_sum_a, MPFR_RNDN);
        mpfr_sub(temp_sub_a, temp, zr_squared, MPFR_RNDN);
        mpfr_sub(temp, temp_sub_a, zi_squared, MPFR_RNDN);

        mpfr_add(last_z_imag, temp, start_imag, MPFR_RNDN);

        mpfr_sub(temp_sub_a, zr_squared, zi_squared, MPFR_RNDN);
        mpfr_add(last_z_real, temp_sub_a, start_real, MPFR_RNDN);

        mpfr_sqr(zr_squared, last_z_real, MPFR_RNDN);
        mpfr_sqr(zi_squared, last_z_imag, MPFR_RNDN);
    }

    // Print the value to the print buffer, then create a new string
    // of appropriate length from there?
    mpfr_sprintf(print_buffer, "%Re", last_z_real);
    *last_z_real_str = strdup(print_buffer);
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


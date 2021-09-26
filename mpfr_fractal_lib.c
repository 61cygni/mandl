
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

// Notably, the type of 'result' is different for distance estimation, 
// because the extra smoothing calculation is handled in-place where the
// derivative is available.
void mpfr_mandelbrot_distance_estimate(long *result, long double *distance_result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, float radius, const long max_iter, long prec)
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

    mpfr_t dz_real;
    mpfr_t dz_imag;
    mpfr_t dz_real_temp;
 
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

    mpfr_init(dz_real);
    mpfr_init(dz_imag);
    mpfr_init(dz_real_temp);
 
    // for set_str, magic number param 10 = base, not precision
    mpfr_set_str(start_real, start_real_str, 10, MPFR_RNDN);
    mpfr_set_str(start_imag, start_imag_str, 10, MPFR_RNDN);

    // Mandelbrot derivative initializes to (0+0j),
    // FYI: however, Julia derivative inittialize to (1+0j), 
    mpfr_set_d(dz_real, 0.0, MPFR_RNDN);
    mpfr_set_d(dz_imag, 0.0, MPFR_RNDN);

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

    // Mandelbrot inits are to zeroes
    mpfr_set_d(last_z_real, 0.0, MPFR_RNDN); 
    mpfr_set_d(last_z_imag, 0.0, MPFR_RNDN);

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

        // Mandelbrot derivative is Z' -> 2*Z*Z' + 1
        // FYI: Julia derivative is Z' -> 2*Z*Z', without the extra "+1"
        mpfr_mul(temp_sum_a, last_z_real, dz_real, MPFR_RNDN); 
        mpfr_mul(temp_sub_a, last_z_imag, dz_imag, MPFR_RNDN);
        mpfr_sub(temp, temp_sum_a, temp_sub_a, MPFR_RNDN);
        mpfr_add(temp_sum_a, temp, temp, MPFR_RNDN);
        mpfr_add_d(dz_real_temp, temp_sum_a, 1.0, MPFR_RNDN);

        mpfr_mul(temp_sum_a, last_z_real, dz_imag, MPFR_RNDN);
        mpfr_mul(temp, last_z_imag, dz_real, MPFR_RNDN);
        mpfr_add(dz_imag, temp_sum_a, temp, MPFR_RNDN);
        mpfr_swap(dz_real, dz_real_temp);

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

    if(*result == max_iter)
    {
        *distance_result = *result;
        //sprintf(print_buffer, "%lu", *result);
        //*distance_result = strdup(print_buffer);
    }
    else
    {
        // Just reusing variables here, even though they're named
        // for the main loop above.  Sorry.
        mpfr_add(temp, zr_squared, zi_squared, MPFR_RNDN);
        mpfr_sqrt(temp_magnitude, temp, MPFR_RNDN); 

        mpfr_sqr(dz_real_temp, dz_real, MPFR_RNDN);
        mpfr_sqr(temp_sum_a, dz_imag, MPFR_RNDN);
        mpfr_add(temp, dz_real_temp, temp_sum_a, MPFR_RNDN);
        mpfr_sqrt(dz_real_temp, temp, MPFR_RNDN); 

        // Now, temp_magnitude is zMagnitude,
        // and dz_real_temp is dzMagnitude.
        if(mpfr_cmp_d(temp_magnitude, 0.0) <= 0 ||
           mpfr_cmp_d(dz_real_temp, 0.0) <= 0)
        {
            *distance_result = *result;
            //sprintf(print_buffer, "%lu", *result);
            //*distance_result = strdup(print_buffer);
        } 
        else
        {
            mpfr_log(temp_sum_a, temp_magnitude, MPFR_RNDN);
            mpfr_mul(temp, temp_sum_a, temp_magnitude, MPFR_RNDN);
            mpfr_div(temp_sum_a, temp, dz_real_temp, MPFR_RNDN);
            mpfr_add_si(temp, temp_sum_a, *result, MPFR_RNDN);   

            *distance_result = mpfr_get_ld(temp, MPFR_RNDN); 
            //mpfr_sprintf(print_buffer, "%Re", temp); 
            //*distance_result = strdup(print_buffer);
        }
    }

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

    mpfr_clear(dz_real);
    mpfr_clear(dz_imag);
    mpfr_clear(dz_real_temp);
}

void mpfr_julia_distance_estimate(long *result, long double *distance_result, char **last_z_real_str, char **last_z_imag_str, const char *start_real_str, const char *start_imag_str, const char *julia_real_str, const char *julia_imag_str, float radius, const long max_iter, long prec)
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

    mpfr_t dz_real;
    mpfr_t dz_imag;
    mpfr_t dz_real_temp;
 
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

    mpfr_init(dz_real);
    mpfr_init(dz_imag);
    mpfr_init(dz_real_temp);
 
    // for set_str, magic number param 10 = base, not precision
    mpfr_set_str(start_real, start_real_str, 10, MPFR_RNDN); 
    mpfr_set_str(start_imag, start_imag_str, 10, MPFR_RNDN);

    // Julia derivative inittialize to (1+0j), but mandelbrot is (0+0j), FYI
    mpfr_set_d(dz_real, 1.0, MPFR_RNDN);
    mpfr_set_d(dz_imag, 0.0, MPFR_RNDN);

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

        // Julia derivative is Z' -> 2*Z*Z'
        // FYI: There's an extra "+1" for mandelbrot dzReal, but
        // julia is just 2*Z*Z'
        mpfr_mul(temp_sum_a, last_z_real, dz_real, MPFR_RNDN); 
        mpfr_mul(temp_sub_a, last_z_imag, dz_imag, MPFR_RNDN);
        mpfr_sub(temp, temp_sum_a, temp_sub_a, MPFR_RNDN);
        mpfr_add(dz_real_temp, temp, temp, MPFR_RNDN);

        mpfr_mul(temp_sum_a, last_z_real, dz_imag, MPFR_RNDN);
        mpfr_mul(temp, last_z_imag, dz_real, MPFR_RNDN);
        mpfr_add(dz_imag, temp_sum_a, temp, MPFR_RNDN);
        mpfr_swap(dz_real, dz_real_temp);

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

    if(*result == max_iter)
    {
        *distance_result = *result;
        //sprintf(print_buffer, "%lu", *result);
        //*distance_result = strdup(print_buffer);
    }
    else
    {
        // Just reusing variables here, even though they're named
        // for the main loop above.  Sorry.
        mpfr_add(temp, zr_squared, zi_squared, MPFR_RNDN);
        mpfr_sqrt(temp_magnitude, temp, MPFR_RNDN); 

        mpfr_sqr(dz_real_temp, dz_real, MPFR_RNDN);
        mpfr_sqr(temp_sum_a, dz_imag, MPFR_RNDN);
        mpfr_add(temp, dz_real_temp, temp_sum_a, MPFR_RNDN);
        mpfr_sqrt(dz_real_temp, temp, MPFR_RNDN); 

        // Now, temp_magnitude is zMagnitude,
        // and dz_real_temp is dzMagnitude.
        if(mpfr_cmp_d(temp_magnitude, 0.0) <= 0 ||
           mpfr_cmp_d(dz_real_temp, 0.0) <= 0)
        {
            *distance_result = *result;
            //sprintf(print_buffer, "%lu", *result);
            //distance_result = strdup(print_buffer);
        } 
        else
        {
            mpfr_log(temp_sum_a, temp_magnitude, MPFR_RNDN);
            mpfr_mul(temp, temp_sum_a, temp_magnitude, MPFR_RNDN);
            mpfr_div(temp_sum_a, temp, dz_real_temp, MPFR_RNDN);
            mpfr_add_si(temp, temp_sum_a, *result, MPFR_RNDN);   
 
            //mpfr_sprintf(print_buffer, "%Re", temp); 
            //*distance_result = strdup(print_buffer);
            *distance_result = mpfr_get_ld(temp, MPFR_RNDN);
        }
    }

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

    mpfr_clear(dz_real);
    mpfr_clear(dz_imag);
    mpfr_clear(dz_real_temp);
}


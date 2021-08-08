/*-----------------------------------------------------------------------------
 * file: nativemandel.c 
 * date Sat Aug 07 08:48:52 PDT 2021 :
 * Author: Martin Casado 
 *
 * Description:
 *
 *---------------------------------------------------------------------------*/


#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "libbf.h"

static bf_context_t bf_ctx;

static void *my_bf_realloc(void *opaque, void *ptr, size_t size)
{
    return realloc(ptr, size);
}

static void print_bf(const bf_t *ptr){
    char *digits;
    size_t digits_len;
    digits = bf_ftoa(&digits_len, ptr, 10, 128,
                         BF_FTOA_FORMAT_FIXED | BF_RNDZ);
    printf("%s\n",digits);
    free(digits);

}

//--
//Take a null terminated string and convert to a bf_t
//--
static int str_to_bf_t(bf_t* result,char* str, int precision){
    bf_t digit;
    bf_t position;
    
    bf_init(&bf_ctx, &digit);
    bf_init(&bf_ctx, &position);

    bf_set_ui(&digit, 0);
    bf_set_ui(result, 0);

    bf_set_ui(&position, 1); // position of digit after decimal

    int top = atoi(str); // pull off int before the decimal
    int neg = 1;
    if(top < 0){
        neg = -1; 
        top *= neg; // turn positive while we add digits
    }
    bf_set_si(result, top); 
    // print_bf(result);

    int index = 0;

    while (str[index] && str[index++] != '.') { ;
    }
    if(! str[index]){
        bf_mul_si(result, result, neg, precision, BF_RNDU); 
        return 0;
    }
    // Should be just past the decimal point here ...
    int num;
    uint64_t val = 0;

    while (str[index]){

        num = str[index] - '0'; // convert char to int
        bf_set_ui(&digit, num);
        bf_mul_ui(&position, &position, 10, precision, BF_RNDU); 
        bf_div(&digit, &digit, &position, precision, BF_RNDU);  
        bf_add(result, result, &digit, precision, BF_RNDU);
        val++;
        index++;
    }
    bf_mul_si(result, result, neg, precision, BF_RNDU); 
    return val; // number of digits after decimal point
}

void squared_modulus(bf_t* result, bf_t* real, bf_t* imag, int precision){
    bf_t tmp;
    bf_t tmp2;
    bf_init(&bf_ctx, &tmp);
    bf_init(&bf_ctx, &tmp2);

    bf_mul(&tmp,  real, real, precision, BF_RNDU);
    bf_mul(&tmp2, imag, imag, precision, BF_RNDU);
    bf_add(result, &tmp, &tmp2, precision, BF_RNDU);
}

float calc_pixel(bf_t* re_x, bf_t* im_y, int precision) {
    int squared_rad = 256 * 256;
    int max_iter = 10000;

    bf_t z_real;
    bf_t z_imag;

    bf_t tmp;
    bf_t tmp2;
    bf_t two;
    bf_t sm;
    bf_t rad;

    bf_init(&bf_ctx, &z_real);
    bf_init(&bf_ctx, &z_imag);
    bf_init(&bf_ctx, &tmp);
    bf_init(&bf_ctx, &tmp2);
    bf_init(&bf_ctx, &two);
    bf_init(&bf_ctx, &sm);
    bf_init(&bf_ctx, &rad);

    bf_set_ui(&z_real, 0);
    bf_set_ui(&z_imag, 0);
    bf_set_ui(&two,    2);
    bf_set_ui(&rad,    squared_rad);

    for(int i = 0; i < max_iter; ++i) {
        // z_real =  (z_real*z_real - z_imag*z_imag) + re_x
        bf_mul(&tmp,   &z_real, &z_real, precision, BF_RNDU);  
        bf_mul(&tmp2,  &z_imag, &z_imag, precision, BF_RNDU);  
        bf_sub(&tmp,    &tmp,   &tmp2,   precision, BF_RNDU);  
        bf_add(&z_real, &tmp, re_x, precision, BF_RNDU); 

        // z_imag = hpf(2)*z_real*z_imag + imag )
        bf_mul(&tmp, &two, &z_real, precision, BF_RNDU);
        bf_mul(&tmp, &tmp, &z_imag, precision, BF_RNDU);
        bf_add(&z_imag, &tmp, im_y, precision, BF_RNDU);

        //if csquared_modulus(z_real, z_imag) >= squared_er: 
        squared_modulus(&sm, &z_real, &z_imag, precision);
        if(! bf_cmp_lt(&sm, &rad)) {
            return i;
        }
    }
    return max_iter;
}

float calc_pixel_smooth(bf_t* re_x, bf_t* im_y, int precision) {
    int squared_rad = 256 * 256;
    int max_iter = 10000;

    bf_t z_real;
    bf_t z_imag;

    bf_t tmp;
    bf_t tmp2;
    bf_t two;
    bf_t sm;
    bf_t rad;

    bf_init(&bf_ctx, &z_real);
    bf_init(&bf_ctx, &z_imag);
    bf_init(&bf_ctx, &tmp);
    bf_init(&bf_ctx, &tmp2);
    bf_init(&bf_ctx, &two);
    bf_init(&bf_ctx, &sm);
    bf_init(&bf_ctx, &rad);

    bf_set_ui(&z_real, 0);
    bf_set_ui(&z_imag, 0);
    bf_set_ui(&two,    2);
    bf_set_ui(&rad,    squared_rad);

    float l = 0.0;

    for(int i = 0; i < max_iter; ++i) {
        // z_real =  (z_real*z_real - z_imag*z_imag) + re_x
        bf_mul(&tmp,   &z_real, &z_real, precision, BF_RNDU);  
        bf_mul(&tmp2,  &z_imag, &z_imag, precision, BF_RNDU);  
        bf_sub(&tmp,    &tmp,   &tmp2,   precision, BF_RNDU);  
        bf_add(&z_real, &tmp, re_x, precision, BF_RNDU); 

        // z_imag = hpf(2)*z_real*z_imag + imag )
        bf_mul(&tmp, &two, &z_real, precision, BF_RNDU);
        bf_mul(&tmp, &tmp, &z_imag, precision, BF_RNDU);
        bf_add(&z_imag, &tmp, im_y, precision, BF_RNDU);

        //if csquared_modulus(z_real, z_imag) >= squared_er: 
        squared_modulus(&sm, &z_real, &z_imag, precision);
        if(! bf_cmp_lt(&sm, &rad)) {
            break; 
        }
        l += 1.0;
    }
    if(l>= max_iter) {
        return 1.0;
    }

    // sl = (l - math.log2(math.log2(csquared_modulus(z_real,z_imag)))) + 4.0;
    // sm should alrady contain sqaure_mod of z_real and z_imag
    bf_const_log2(&sm, precision, BF_RNDU);
    bf_const_log2(&sm, precision, BF_RNDU);

    bf_set_float64(&tmp, l);
    bf_sub(&tmp, &tmp, &sm, precision, BF_RNDU);
    bf_set_float64(&tmp2, 4.0);
    bf_add(&tmp, &tmp, &tmp2, precision, BF_RNDU);

    double ret;
    bf_get_float64(&tmp, &ret, BF_RNDU);
    return ret;
}


int main(int argc, char **argv)
{
    int precision = 128; 

    char *str_real = "-1.76938317919551501821384728608547378290574726365475143746552821652788819126";
    char *str_imag = "0.00423684791873677221492650717136799707668267091740375727945943565011234400"; 
    //char *str_real = "-1.";
    //char *str_imag = "0.";

    bf_context_init(&bf_ctx, my_bf_realloc, NULL);

    bf_t c_real; 
    bf_t c_imag; 

    bf_init(&bf_ctx, &c_real);
    bf_init(&bf_ctx, &c_imag);

    str_to_bf_t(&c_real, str_real, precision);
    //print_bf(&c_real);
    str_to_bf_t(&c_imag, str_imag, precision);
    // print_bf(&c_imag);

    // Fractal variables start here 
    int img_w = 400, img_h = 300;
    bf_t cmplx_w;  
    bf_t cmplx_h;  

    float COMPLEX_WIDTH = .09;


    bf_init(&bf_ctx, &cmplx_w);
    bf_init(&bf_ctx, &cmplx_h);

    bf_set_float64(&cmplx_w, COMPLEX_WIDTH); 
    bf_set_float64(&cmplx_h, COMPLEX_WIDTH * ((float)img_h / (float)img_w) ); 

    printf("Complex width :");
    print_bf(&cmplx_w);
    printf("Complex height :");
    print_bf(&cmplx_h);

    // Calc Current Frame
    bf_t re_start;
    bf_t re_end;
    bf_t im_start;
    bf_t im_end;
    bf_t two;

    bf_init(&bf_ctx, &re_start);
    bf_init(&bf_ctx, &re_end);
    bf_init(&bf_ctx, &im_start);
    bf_init(&bf_ctx, &im_end);
    bf_init(&bf_ctx, &two);

    // re_start = cmplx_center.real - cmplx_w / 2
    // re_end   = cmplx_center.real + cmplx_w / 2
    bf_t tmp;
    bf_init(&bf_ctx, &tmp);
    bf_t tmp2;
    bf_init(&bf_ctx, &tmp2);
    
    bf_set_ui(&two, 2);
    bf_div(&tmp, &cmplx_w, &two, precision, BF_RNDU);

    // print_bf(&tmp);

    bf_sub(&re_start, &c_real, &tmp, precision, BF_RNDU); 
    bf_add(&re_end, &c_real,   &tmp, precision, BF_RNDU); 

    printf("re_start: ");
    print_bf(&re_start);
    printf("re_end: ");
    print_bf(&re_end);

    // im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
    // im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))
    bf_div(&tmp, &cmplx_h, &two, precision, BF_RNDU);
    bf_sub(&im_start, &c_imag, &tmp, precision, BF_RNDU); 
    bf_add(&im_end,   &c_imag, &tmp, precision, BF_RNDU); 

    printf("im_start: ");
    print_bf(&im_start);
    printf("im_end: ");
    print_bf(&im_end);

    // main loop here!!
    bf_t re_x;
    bf_t im_y;
    bf_init(&bf_ctx, &re_x);
    bf_init(&bf_ctx, &im_y);

    float prog; // fraction of progress
    int res;
    printf("d = {};\n");
    for(int y = 0; y < img_h; ++y){
        for(int x = 0; x < img_w; ++x){
            // map from pixels to complex coordinates 
            // Re_x = (re_start) + (x / img_width)  * (re_end - re_start)
            prog = (float)x/img_w; 
            bf_sub(&tmp, &re_end, &re_start, precision, BF_RNDU);  // re_end - re_start
            bf_set_float64(&tmp2, prog);
            bf_mul(&tmp, &tmp2, &tmp, precision, BF_RNDU);
            bf_add(&re_x, &re_start, &tmp, precision, BF_RNDU);

            // Im_y = (im_start) + (y / img_height) * (im_end - im_start)
            prog = (float)y/img_h;
            bf_sub(&tmp, &im_end, &im_start, precision, BF_RNDU); 
            bf_set_float64(&tmp2, prog);
            bf_mul(&tmp, &tmp2, &tmp, precision, BF_RNDU);
            bf_add(&im_y, &im_start, &tmp, precision, BF_RNDU);

            // printf("re_x: ");
            // print_bf(&re_x);
            // printf("im_y: ");
            // print_bf(&im_y);

            res = calc_pixel(&re_x, &im_y, precision); // main calculation!
            //printf("d[(%d,%d)] = %d; ",x,y,res);
             if(res == 256)
                 printf("*");
             else
                 printf(".");
        } // y
        printf("\n");
        fflush(stdout);
    } // x

    return 0;
}

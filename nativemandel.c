/*-----------------------------------------------------------------------------
 * file: nativemandel.c 
 * date Sat Aug 07 08:48:52 PDT 2021 :
 * Author: Martin Casado 
 *
 * Description:
 *
 * c kernel for high precision mandelbrot calculations. This is not meant
 * as a standalone program, but expected to be driven by a python harness
 * which handles the output.
 *
 * Smoothing implementation based on :
 *
 * https://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
 * https://www.shadertoy.com/view/4df3Rn coloring and smoothing working
 *
 * TODO:
 * - move initializations out of calc_pixel_smooth
 * - test with lower escape rad (4?)
 * - test with infinite precision? BF_PREC_INF 
 * - have debugging output go to stderr
 * - print timing for frames and output
 *
 *
 *---------------------------------------------------------------------------*/


#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "libbf.h"

static int img_w = 80, img_h = 60;
static limb_t precision  = 1600; 
static int max_iter      = 40000;
//char *str_real = "-1.";
//char *str_imag = "0.";
static char *str_real    = "-1.7693831791955150182138472860854737829057472636547514374655282165278881912647564588361634463895296673044858257818203031574874912384217194031282461951137475212550848062085787454772803303225167998662391124184542743017129214423639793169296754394181656831301342622793541423768572435783910849972056869527305207508191441734781061794290699753174911133714351734166117456520272756159178932042908932465102671790878414664628213755990650460738372283470777870306458882";
static char *str_imag    = "0.0042368479187367722149265071713679970766826709174037572794594356501123440008055451573024309950236365063135326833596525718230049480553873630612752481493929235593089283439205079672488790492198666604557662694690066610349401490471432372558697978990852065668320265806402411530037882678978639464162203534105510290045630572371868452721037732584630791751262877467200569332623280695382279675583251718887347912436143098948549550112409632942168282733069353217150536"; 
//static char *str_cmplx_w = ".000000000000000000001";
//static char *str_cmplx_w = ".0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";
static char *str_cmplx_w = ".000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";

static bf_context_t bf_ctx;

static void *my_bf_realloc(void *opaque, void *ptr, size_t size) {
    return realloc(ptr, size);
}

static void print_bf(const bf_t *ptr, limb_t precision) {
    char *digits;
    size_t digits_len;
    digits = bf_ftoa(&digits_len, ptr, 10, precision,
                         BF_FTOA_FORMAT_FIXED | BF_RNDZ);
    printf("%s",digits);
    free(digits);
}

static int bf_log2(bf_t* ret, const bf_t* in, limb_t prec,
        bf_flags_t flags) {

    bf_t logofin;
    bf_t logoftwo;
    bf_t two;

    bf_init(&bf_ctx, &logofin);
    bf_init(&bf_ctx, &logoftwo);
    bf_init(&bf_ctx, &two);

    bf_set_ui(&two, 2);

    // log2(x) = log10(x) / log10(x)
    bf_log(&logofin,  in,   prec, BF_RNDU);
    bf_log(&logoftwo, &two, prec, BF_RNDU);
    return bf_div(ret, &logofin, &logoftwo, prec, BF_RNDU);
}

//--
//Take a null terminated string and convert to a bf_t
//--
static int str_to_bf_t(bf_t* result,char* str, int prec){
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
        bf_mul_si(result, result, neg, prec, BF_RNDU); 
        return 0;
    }
    // Should be just past the decimal point here ...
    int num;
    uint64_t val = 0;

    while (str[index]){

        num = str[index] - '0'; // convert char to int
        bf_set_ui(&digit, num);
        bf_mul_ui(&position, &position, 10, prec, BF_RNDU); 
        bf_div(&digit, &digit, &position, prec, BF_RNDU);  
        bf_add(result, result, &digit, prec, BF_RNDU);
        val++;
        index++;
    }
    bf_mul_si(result, result, neg, prec, BF_RNDU); 
    return val; // number of digits after decimal point
}

void squared_modulus(bf_t* result, bf_t* real, bf_t* imag, int prec){
    bf_t tmp;
    bf_t tmp2;
    bf_init(&bf_ctx, &tmp);
    bf_init(&bf_ctx, &tmp2);

    bf_mul(&tmp,  real, real, prec, BF_RNDU);
    bf_mul(&tmp2, imag, imag, prec, BF_RNDU);
    bf_add(result, &tmp, &tmp2, prec, BF_RNDU);
}

float calc_pixel(bf_t* re_x, bf_t* im_y, int prec) {
    int squared_rad = 256 * 256;

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
        bf_mul(&tmp,   &z_real, &z_real, prec, BF_RNDU);  
        bf_mul(&tmp2,  &z_imag, &z_imag, prec, BF_RNDU);  
        bf_sub(&tmp,    &tmp,   &tmp2,   prec, BF_RNDU);  
        bf_add(&z_real, &tmp, re_x, prec, BF_RNDU); 

        // z_imag = hpf(2)*z_real*z_imag + imag )
        bf_mul(&tmp, &two, &z_real, prec, BF_RNDU);
        bf_mul(&tmp, &tmp, &z_imag, prec, BF_RNDU);
        bf_add(&z_imag, &tmp, im_y, prec, BF_RNDU);

        //if csquared_modulus(z_real, z_imag) >= squared_er: 
        squared_modulus(&sm, &z_real, &z_imag, prec);
        if(! bf_cmp_lt(&sm, &rad)) {
            return i;
        }
    }
    return max_iter;
}

float calc_pixel_smooth(bf_t* re_x, bf_t* im_y, int prec) {
    int squared_rad = 256 * 256;

    bf_t z_real;
    bf_t z_imag;

    bf_t tmp;
    bf_t tmp2;
    bf_t log2;
    bf_t loglog2;
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
    bf_init(&bf_ctx, &log2);
    bf_init(&bf_ctx, &loglog2);

    bf_set_ui(&z_real, 0);
    bf_set_ui(&z_imag, 0);
    bf_set_ui(&two,    2);
    bf_set_ui(&rad,    squared_rad);

    float l = 0.0;

    for(int i = 0; i < max_iter; ++i) {
        // z_real =  (z_real*z_real - z_imag*z_imag) + re_x
        bf_mul(&tmp,  &z_real, &z_real, prec, BF_RNDU);  
        bf_mul(&tmp2, &z_imag, &z_imag, prec, BF_RNDU);  
        bf_sub(&tmp,  &tmp,    &tmp2,   prec, BF_RNDU);  
        bf_add(&tmp2, &tmp,     re_x,   prec, BF_RNDU);  // save z_real result while we calc second portion

        // z_imag = 2.0 *z_real*z_imag + imag 
        bf_mul(&tmp,    &two, &z_real, prec, BF_RNDU);
        bf_mul(&tmp,    &tmp, &z_imag, prec, BF_RNDU);
        bf_add(&z_imag, &tmp,  im_y,   prec, BF_RNDU);

        bf_set(&z_real, &tmp2);

        //if csquared_modulus(z_real, z_imag) >= squared_er: 
        // squared_modulus(&sm, &z_real, &z_imag, prec);
        bf_mul(&tmp,  &z_real, &z_real, prec, BF_RNDU);
        bf_mul(&tmp2, &z_imag, &z_imag, prec, BF_RNDU);
        bf_add(&sm,   &tmp,    &tmp2,   prec, BF_RNDU);

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
    bf_log2(&log2,    &sm,   prec, BF_RNDU);
    bf_log2(&loglog2, &log2, prec, BF_RNDU);

    bf_set_float64(&tmp, l);
    // l - log2(log2(sqaured_mod()))
    bf_sub(&tmp, &tmp, &loglog2, prec, BF_RNDU);
    bf_set_float64(&tmp2, 4.0);
    bf_add(&tmp, &tmp, &tmp2, prec, BF_RNDU);

    double ret;
    bf_get_float64(&tmp, &ret, BF_RNDU);
    return ret;
}

void print_header() {
    printf(" # -- \n");
    printf(" #\n");
    printf(" # Image w: %d", img_w);
    printf(" # Image h: %d", img_h);
    printf(" #\n");
    printf(" # Center: \n");
    printf(" #     Re %s: \n", str_real);
    printf(" #     Im %s: \n", str_imag);
    printf(" #\n");
    printf(" #     Max iter %d: \n",  max_iter);
    printf(" #     Precision %d: \n", (int)precision);
    printf(" #\n");
    printf(" # -- \n");
    printf(" #\n");
}

int main(int argc, char **argv)
{

    bf_context_init(&bf_ctx, my_bf_realloc, NULL);

    bf_t c_real; 
    bf_t c_imag; 

    bf_init(&bf_ctx, &c_real);
    bf_init(&bf_ctx, &c_imag);

    str_to_bf_t(&c_real, str_real, precision);
    //print_bf(&c_real);
    str_to_bf_t(&c_imag, str_imag, precision);
    // print_bf(&c_imag);
    
    print_header();

    // Fractal variables start here 
    bf_t cmplx_w;  
    bf_t cmplx_h;  

    bf_init(&bf_ctx, &cmplx_w);
    bf_init(&bf_ctx, &cmplx_h);

    str_to_bf_t(&cmplx_w, str_cmplx_w, precision);

    bf_set_float64(&cmplx_h, ((float)img_h / (float)img_w) ); 
    bf_mul(&cmplx_h, &cmplx_h, &cmplx_w, precision, BF_RNDU); 

    // printf("Complex width :");
    // print_bf(&cmplx_w);
    // printf("Complex height :");
    // print_bf(&cmplx_h);

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

    bf_sub(&re_start, &c_real, &tmp, precision, BF_RNDU); 
    bf_add(&re_end, &c_real,   &tmp, precision, BF_RNDU); 

    printf("re_start = \"");
    print_bf(&re_start, precision); printf("\"\n");
    printf("re_end =  \"");
    print_bf(&re_end, precision);  printf("\"\n");

    // im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
    // im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))
    bf_div(&tmp, &cmplx_h, &two, precision, BF_RNDU);
    bf_sub(&im_start, &c_imag, &tmp, precision, BF_RNDU); 
    bf_add(&im_end,   &c_imag, &tmp, precision, BF_RNDU); 

    printf("im_start = \"");
    print_bf(&im_start, precision); printf("\"\n");
    printf("im_end = \"");
    print_bf(&im_end, precision); printf("\"\n");

    // main loop here!!
    bf_t re_x;
    bf_t im_y;
    bf_init(&bf_ctx, &re_x);
    bf_init(&bf_ctx, &im_y);

    float prog; // fraction of progress
    float res;
    //int res;
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

            // res = calc_pixel(&re_x, &im_y, precision); // main calculation!
            res = calc_pixel_smooth(&re_x, &im_y, precision); // main calculation!
            printf("d[(%d,%d)] = %f; ",x,y,res);
        } // y
        printf("\n");
        fflush(stdout);
    } // x

    return 0;
}

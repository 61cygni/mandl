/*-----------------------------------------------------------------------------
 * file: nativemandel.c 
 * date Sat Aug 07 08:48:52 PDT 2021 :
 * Author: Martin Casado 
 *
 * Description:
 *
 * c kernel for high precision mandelbrot calculations. This is not meant
 * as a standalone program, but expected to be driven by a python harness
 * which handles the output. It's also specifically not meant for anything
 * but deep dives. So it doesn't use any optimizations such as skipping M1, M2
 * or native floats at low zoom levels. 
 *
 * Smoothing implementation based on :
 *
 * https://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
 * https://www.shadertoy.com/view/4df3Rn coloring and smoothing working
 *
 *
 * usage : nativemandel 
 *              -i generate a PNG image (for debugging
 *              -v dump debugging output to stderr
 *
 * TODO:
 * - move initializations out of calc_pixel_smooth
 * - print timing for frames and output
 *
 *---------------------------------------------------------------------------*/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "libbf.h"
#include "libattopng.h"

#define RGBA(r, g, b) ((r) | ((g) << 8) | ((b) << 16))

static int img_w = 2048, img_h = 1536;
static limb_t precision  = 100; 
static int max_iter      = 5000;

static char *str_real = "-.749696000010025";
static char *str_imag = "0.031456625003";
static char *str_cmplx_w = ".0000000000001";


// Full mandelbrot set
// static char *str_real = "-1.";
// static char *str_imag = "0.";
// static char *str_cmplx_w = "4.";

// The following values are used to max out the system. This is a snapshot at 10^-512 (should be the 8-fold circle)
// static int img_w = 160, img_h = 120;
// static limb_t precision  = 2000; 
// static int max_iter      = 80000;
// static char *str_real    = "-1.76938317919551501821384728608547378290574726365475143746552821652788819126475645883616344638952966730448582578182030315748749123842171940312824619511374752125508480620857874547728033032251679986623911241845427430171292144236397931692967543941816568313013426227935414237685724357839108499720568695273052075081914417347810617942906997531749111337143517341661174565202727561591789320429089324651026717908784146646282137559906504607383722834707778703064588828982026040017443489083888449628870745058537070958320394103234549205405343784";
// static char *str_imag    = "0.00423684791873677221492650717136799707668267091740375727945943565011234400080554515730243099502363650631353268335965257182300494805538736306127524814939292355930892834392050796724887904921986666045576626946900666103494014904714323725586979789908520656683202658064024115300378826789786394641622035341055102900456305723718684527210377325846307917512628774672005693326232806953822796755832517188873479124361430989485495501124096329421682827330693532171505367455526637382706988583456915684673202462211937384523487065290004627037270912"; 
// static char *str_cmplx_w = ".00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";

static bf_context_t bf_ctx;

static void *my_bf_realloc(void *opaque, void *ptr, size_t size) {
    return realloc(ptr, size);
}


static void fprint_bf(FILE* fp, const bf_t *ptr, limb_t precision) {
    char *digits;
    size_t digits_len;
    digits = bf_ftoa(&digits_len, ptr, 10, precision,
                         BF_FTOA_FORMAT_FIXED | BF_RNDZ);
    fprintf(fp, "%s",digits);
    free(digits);
}

static void print_bf(const bf_t *ptr, limb_t precision) {
    fprint_bf(stdout, ptr, precision);
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
    if(str[0] == '-' || top < 0){
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

void map_to_color(float value, int* red, int* blue, int* green);

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

void usage() {
    printf("Usage: nativemandel [-i] [-v]\n");
    printf(" -i dump to image for debugging\n");
    printf(" -v debug output to stderr\n");
    exit(0);
}

int main(int argc, char **argv)
{
    int ch, iflag = 0, vflag = 0;

    while ((ch = getopt(argc, argv, "iv")) != -1) {
        switch (ch) {
            case 'i':
                iflag = 1;
                break;
            case 'v':
                vflag = 1;
                break;
            case '?':
            default:
                usage();
        }
    }

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
    bf_t cmplx_w;  
    bf_t cmplx_h;  

    bf_init(&bf_ctx, &cmplx_w);
    bf_init(&bf_ctx, &cmplx_h);

    str_to_bf_t(&cmplx_w, str_cmplx_w, precision);

    bf_set_float64(&cmplx_h, ((float)img_h / (float)img_w) ); 
    bf_mul(&cmplx_h, &cmplx_h, &cmplx_w, precision, BF_RNDU); 


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

    // im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
    // im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))
    bf_div(&tmp, &cmplx_h, &two, precision, BF_RNDU);
    bf_sub(&im_start, &c_imag, &tmp, precision, BF_RNDU); 
    bf_add(&im_end,   &c_imag, &tmp, precision, BF_RNDU); 

    if(! iflag){
        print_header();
        printf("re_start = \"");
        print_bf(&re_start, precision); printf("\"\n");
        printf("re_end =  \"");
        print_bf(&re_end, precision);  printf("\"\n");
        printf("im_start = \"");
        print_bf(&im_start, precision); printf("\"\n");
        printf("im_end = \"");
        print_bf(&im_end, precision); printf("\"\n");
    }

    if(vflag){
        fprintf(stderr, "re_start ");
        fprint_bf(stderr, &re_start, precision); fprintf(stderr,"\n");
        fprintf(stderr, "re_end =  ");
        fprint_bf(stderr, &re_end, precision);  fprintf(stderr, "\n");
        fprintf(stderr, "im_start ");
        fprint_bf(stderr, &im_start, precision); fprintf(stderr, "\n");
        fprintf(stderr, "im_end ");
        fprint_bf(stderr, &im_end, precision); fprintf(stderr, "\n");
        fprintf(stderr, "Complex width :");
        fprint_bf(stderr, &cmplx_w, precision);
        fprintf(stderr, "\n");
        fprintf(stderr, "Complex height :");
        fprint_bf(stderr, &cmplx_h, precision);
        fprintf(stderr, "\n");
    }

    // main loop here!!
    bf_t re_x;
    bf_t im_y;
    bf_init(&bf_ctx, &re_x);
    bf_init(&bf_ctx, &im_y);


    float prog; // fraction of progress
    float res;
    //int res;

    int red,green,blue;
    libattopng_t* png = 0;

    if(iflag) {
        png = libattopng_new(img_w, img_h, PNG_RGB);
    }else{
        printf("d = {};\n");
    }

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
            if(iflag) {
                map_to_color(res,&red,&green,&blue);
                libattopng_set_pixel(png, x, y, RGBA(red,green,blue)); 
            }else{
                printf("d[(%d,%d)] = %f; ",x,y,res);
                fflush(stdout);
            }
        } // y
        if(vflag){
            fprintf(stderr,".");
        }
        if(! iflag){
            printf("\n");
            fflush(stdout);
        }
    } // x
    fprintf(stderr,"\n");

    if(iflag){
        libattopng_save(png, "nativemandel.png");
        libattopng_destroy(png);
    }

    return 0;
}

void map_to_color(float val, int* red, int* green, int* blue) {
    // change these values to change color
    float sc0 = 0.7;
    float sc1 = 0.0;
    float sc2 = 0.2;

    float c1 = 0.;
    float c2 = 0.;
    float c3 = 0.;

    c1 +=  1 + cos( 3.0 + val*0.15 + sc0);
    c2 +=  1 + cos( 3.0 + val*0.15 + sc1);
    c3 +=  1 + cos( 3.0 + val*0.15 + sc2);

    *red    = (int)(255.*((c1/4.) * 3.) / 1.5);
    *green  = (int)(255.*((c2/4.) * 3.) / 1.5);
    *blue   = (int)(255.*((c3/4.) * 3.) / 1.5);
}

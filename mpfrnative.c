/*-----------------------------------------------------------------------------
 * file: mpfrnative.c 
 *
 * manderlbrot kernel built around libmpfr for very deep dives. This has
 * been tested to e-510. 
 *
 * Because this is built around a high precision library that is very
 * slow relative to native floats, it doesn't make sense to use for
 * frames with zooms less than e-13 or so.
 *
 *---------------------------------------------------------------------------*/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "mpfr.h"
#include "libattopng.h"

#define RGBA(r, g, b) ((r) | ((g) << 8) | ((b) << 16))
#define FILESTR_LEN 64

static int img_w = 0, img_h = 0;


// Full mandelbrot set
// Doesn't really make sense to use
// static char *str_real = "-1.";
// static char *str_imag = "0.";
// static char *str_cmplx_w = "4.";

// These are good defaults for depth. e-20, renders sufficiently fast. 
static int precision  = 80; 
static int max_iter   = 3000;
static char *str_real = "0.36024044343761436323612524444954530848260780795858575048837581474019534605";
static char *str_imag = "-0.64131306106480317486037501517930206657949495228230525955617754306444857417";
static char *str_cmplx_w = ".0000000000000000001";


// The following values are used to max out the system. This is a snapshot at 10^-512 (should be the 8-fold circle)
// static int img_w = 160, img_h = 120;
// static int precision  = 2000; 
// static int max_iter      = 80000;
// static char *str_real    = "-1.76938317919551501821384728608547378290574726365475143746552821652788819126475645883616344638952966730448582578182030315748749123842171940312824619511374752125508480620857874547728033032251679986623911241845427430171292144236397931692967543941816568313013426227935414237685724357839108499720568695273052075081914417347810617942906997531749111337143517341661174565202727561591789320429089324651026717908784146646282137559906504607383722834707778703064588828982026040017443489083888449628870745058537070958320394103234549205405343784";
// static char *str_imag    = "0.00423684791873677221492650717136799707668267091740375727945943565011234400080554515730243099502363650631353268335965257182300494805538736306127524814939292355930892834392050796724887904921986666045576626946900666103494014904714323725586979789908520656683202658064024115300378826789786394641622035341055102900456305723718684527210377325846307917512628774672005693326232806953822796755832517188873479124361430989485495501124096329421682827330693532171505367455526637382706988583456915684673202462211937384523487065290004627037270912"; 
// static char *str_cmplx_w = ".00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";


float calc_pixel_smooth(mpfr_t re_x, mpfr_t im_y) {
    int squared_rad = 256 * 256;

    mpfr_t z_real;
    mpfr_t z_imag;

    mpfr_t tmp;
    mpfr_t tmp2;
    mpfr_t log2;
    mpfr_t loglog2;
    mpfr_t two;
    mpfr_t sm;
    mpfr_t rad;

    mpfr_init(z_real);
    mpfr_init(z_imag);
    mpfr_init(tmp);
    mpfr_init(tmp2);
    mpfr_init(two);
    mpfr_init(sm);
    mpfr_init(rad);
    mpfr_init(log2);
    mpfr_init(loglog2);

    mpfr_set_ui(z_real, 0, MPFR_RNDN);
    mpfr_set_ui(z_imag, 0, MPFR_RNDN);
    mpfr_set_ui(two,    2, MPFR_RNDN);
    mpfr_set_ui(rad,    squared_rad, MPFR_RNDN);

    float l = 0.0;

    for(int i = 0; i < max_iter; ++i) {
        // z_real =  (z_real*z_real - z_imag*z_imag) + re_x
        mpfr_mul(tmp,  z_real, z_real, MPFR_RNDN);
        mpfr_mul(tmp2, z_imag, z_imag, MPFR_RNDN);
        mpfr_sub(tmp,  tmp,    tmp2, MPFR_RNDN);
        mpfr_add(tmp2, tmp,    re_x, MPFR_RNDN);  // save z_real result while we calc second portion

        // z_imag = 2.0 *z_real*z_imag + imag 
        mpfr_mul(tmp,    two, z_real, MPFR_RNDN);
        mpfr_mul(tmp,    tmp, z_imag, MPFR_RNDN);
        mpfr_add(z_imag, tmp,  im_y, MPFR_RNDN);

        mpfr_set(z_real, tmp2, MPFR_RNDN);

        //if squared_modulus(z_real, z_imag) >= squared_er: 
        mpfr_mul(tmp,  z_real, z_real, MPFR_RNDN);
        mpfr_mul(tmp2, z_imag, z_imag, MPFR_RNDN);
        mpfr_add(sm,   tmp,    tmp2, MPFR_RNDN);

        if( mpfr_cmp(sm, rad) >= 0) {
            break; 
        }
        l += 1.0;
    }
    if(l>= max_iter) {
        return 1.0;
    }


    // sl = (l - math.log2(math.log2(csquared_modulus(z_real,z_imag)))) + 4.0;
    // sm should alrady contain sqaure_mod of z_real and z_imag
    mpfr_log2(log2,    sm,   MPFR_RNDN);
    mpfr_log2(loglog2, log2, MPFR_RNDN);

    mpfr_set_d(tmp, l, MPFR_RNDN);
    // l - log2(log2(sqaured_mod()))
    mpfr_sub(tmp, tmp, loglog2, MPFR_RNDN);
    mpfr_set_d(tmp2, 4.0, MPFR_RNDN);
    mpfr_add(tmp, tmp, tmp2, MPFR_RNDN);

    return mpfr_get_d(tmp, MPFR_RNDN);
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
    printf(" -i <filename> dump to image for debugging\n");
    printf(" -v debug output to stderr\n");
    printf(" -w specify image width (REQUIRED) \n");
    printf(" -h specify image height \n");
    printf(" -m <int> max iter \n");
    printf(" -n specify number of chunks \n");
    printf(" -b specify chunk number to compute \n");
    printf(" -p <int> precision (number of bits) \n");
    printf(" -x \"str\" real value of center on complex plane \n");
    printf(" -y \"str\" imag value of center on complex plane \n");
    printf(" -l width on complex plane \n");
    exit(0);
}

int main(int argc, char **argv)
{
    int ch, iflag = 0, vflag = 0;
    int numblocks = 0, blockno = 0;
    char filename[FILESTR_LEN];
    char* cla_c_real  = 0;
    char* cla_c_imag  = 0;
    char* cla_cmplx_w = 0;

    strncpy(filename, "hpcnative.png",FILESTR_LEN - 1);

    while ((ch = getopt(argc, argv, "i:vw:h:n:b:x:y:p:l:m:")) != -1) {
        switch (ch) {
            case 'i':
                iflag = 1;
                if(strlen(optarg) > FILESTR_LEN){
                    fprintf(stderr," * error : filename on command line too large\n");
                    return 0;
                }
                strncpy(filename, optarg, FILESTR_LEN);
                break;
            case 'v':
                vflag = 1;
                break;
            case 'w': 
                img_w = atoi(optarg); 
                break;
            case 'h': 
                img_h = atoi(optarg); 
                break;
            case 'm': 
                max_iter = atoi(optarg); 
                break;
            case 'n': 
                numblocks = atoi(optarg); 
                break;
            case 'b': 
                blockno = atoi(optarg); 
                break;
            case 'x': 
                cla_c_real = strdup(optarg); 
                break;
            case 'y': 
                cla_c_imag = strdup(optarg); 
                break;
            case 'p': 
                precision = atoi(optarg); 
                break;
            case 'l': 
                cla_cmplx_w = strdup(optarg); 
                break;
            case '?':
            default:
                usage();
        }
    }

    if(!img_w){
        fprintf(stderr, " Error: you must specify image width\n");
        return 0;
    }

    if(!img_h){ // fill using ratio of w/h 1024/768
        img_h = (float)img_w*(768. / 1024.);
    }

    if(!numblocks || !blockno) {
        numblocks = blockno = 1;
    }

    mpfr_set_default_prec(precision);

    mpfr_t c_real; 
    mpfr_t c_imag; 

    mpfr_init (c_real);      
    mpfr_init (c_imag);   

    if(cla_c_real){
        mpfr_set_str (c_real, cla_c_real, 10, MPFR_RNDN);
        free(cla_c_real);
    }else{
        mpfr_set_str (c_real, str_real, 10, MPFR_RNDN);
    }

    if(cla_c_imag){
        mpfr_set_str(c_imag, cla_c_imag, precision, MPFR_RNDN);
        free(cla_c_imag);
    }else{
        mpfr_set_str(c_imag, str_imag, 10, MPFR_RNDN);
    }


    // Fractal variables start here 
    mpfr_t cmplx_w;  
    mpfr_t cmplx_h;  

    mpfr_init (cmplx_w);      
    mpfr_init (cmplx_h);   

    if(cla_cmplx_w){
        mpfr_set_str(cmplx_w, cla_cmplx_w, 10, MPFR_RNDN);
        free(cla_cmplx_w);
    }else{
        mpfr_set_str(cmplx_w, str_cmplx_w, 10, MPFR_RNDN);
    }

    mpfr_set_d(cmplx_h, ((float)img_h / (float)img_w), MPFR_RNDN ); 
    mpfr_mul(cmplx_h, cmplx_h, cmplx_w, MPFR_RNDN); 


    // Calc Current Frame
    mpfr_t re_start;
    mpfr_t re_end;
    mpfr_t im_start;
    mpfr_t im_end;
    mpfr_t two;

    mpfr_init(re_start);
    mpfr_init(re_end);
    mpfr_init(im_start);
    mpfr_init(im_end);
    mpfr_init(two);

    // re_start = cmplx_center.real - cmplx_w / 2
    // re_end   = cmplx_center.real + cmplx_w / 2
    mpfr_t tmp;
    mpfr_init(tmp);
    mpfr_t tmp2;
    mpfr_init(tmp2);

    mpfr_set_ui(two, 2, MPFR_RNDN);
    mpfr_div(tmp, cmplx_w, two, MPFR_RNDN);

    mpfr_sub(re_start, c_real, tmp, MPFR_RNDN);
    mpfr_add(re_end,   c_real, tmp, MPFR_RNDN);

    // im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
    // im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))
    mpfr_div(tmp,      cmplx_h, two, MPFR_RNDN);
    mpfr_sub(im_start, c_imag,  tmp, MPFR_RNDN);
    mpfr_add(im_end,   c_imag,  tmp, MPFR_RNDN);

    if(! iflag){
        print_header();
        printf("re_start = \"");
        mpfr_out_str(stdout, 10, 100, re_start, MPFR_RNDN);
       fprintf(stdout,"\n");
        printf("re_end =  \"");
        mpfr_out_str(stdout, 10, 100, re_end, MPFR_RNDN);
       fprintf(stdout,"\n");
        printf("im_start = \"");
        mpfr_out_str(stdout, 10, 100, im_start, MPFR_RNDN);
       fprintf(stdout,"\n");
        printf("im_end = \"");
        mpfr_out_str(stdout, 10, 100, im_end, MPFR_RNDN);
       fprintf(stdout,"\n");
    }

    if(vflag){
        fprintf(stderr, "img width  %d\n", img_w);
        fprintf(stderr, "img height %d\n", img_h);
        fprintf(stderr, "max iter  %d\n",  max_iter);
        fprintf(stderr, "precision  %d\n", (int)precision);
        fprintf(stderr, "c_real ");
        mpfr_out_str(stderr, 10, 100, c_real, MPFR_RNDN);fprintf(stderr,"\n");
        fprintf(stderr, "c_imag ");
        mpfr_out_str(stderr, 10, 100, c_imag, MPFR_RNDN);fprintf(stderr,"\n");
        fprintf(stderr, "re_start ");
        mpfr_out_str(stderr, 10, 100, re_start, MPFR_RNDN);fprintf(stderr,"\n");
        fprintf(stderr, "re_end =  ");
        mpfr_out_str(stderr, 10, 100, re_end, MPFR_RNDN);fprintf(stderr,"\n");
        fprintf(stderr, "im_start ");
        mpfr_out_str(stderr, 10, 100, im_start, MPFR_RNDN);fprintf(stderr,"\n");
        fprintf(stderr, "im_end ");
        mpfr_out_str(stderr, 10, 100, im_end, MPFR_RNDN);fprintf(stderr,"\n");
        fprintf(stderr, "Complex width :");
        mpfr_out_str(stderr, 10, 100, cmplx_w, MPFR_RNDN);
        fprintf(stderr, "\n");
        fprintf(stderr, "Complex height :");
        mpfr_out_str(stderr, 10, 100, cmplx_h, MPFR_RNDN);
        fprintf(stderr, "\n");
    }

    // main loop here!!
    mpfr_t re_x;
    mpfr_t im_y;
    mpfr_init(re_x);
    mpfr_init(im_y);


    float prog; // fraction of progress
    float res;
    //int res;

    int red,green,blue;
    libattopng_t* png = 0;

    // we only want to calculate our block 
    int blocksize = img_h / numblocks;
    int ystart    = (blockno-1) * blocksize;
    int yend      = ystart + blocksize;
    if(blockno == numblocks){
        yend = img_h;
    }

    if(iflag) {
        png = libattopng_new(img_w, yend - ystart, PNG_RGB);
    }else{
        printf("d = {};\n");
    }


    for(int y = 0; y < img_h; ++y){
        if(y < ystart || y > yend)
            continue;
        for(int x = 0; x < img_w; ++x){
            // map from pixels to complex coordinates 
            // Re_x = (re_start) + (x / img_width)  * (re_end - re_start)
            prog = (float)x/img_w; 
            mpfr_sub(tmp, re_end, re_start, MPFR_RNDN);  // re_end - re_start
            mpfr_set_d(tmp2, prog, MPFR_RNDN);
            mpfr_mul(tmp, tmp2, tmp, MPFR_RNDN);
            mpfr_add(re_x, re_start, tmp, MPFR_RNDN);

            // Im_y = (im_start) + (y / img_height) * (im_end - im_start)
            prog = (float)y/img_h;
            mpfr_sub(tmp, im_end, im_start, MPFR_RNDN);
            mpfr_set_d(tmp2, prog, MPFR_RNDN);
            mpfr_mul(tmp, tmp2, tmp, MPFR_RNDN);
            mpfr_add(im_y, im_start, tmp, MPFR_RNDN);

            // res = calc_pixel(&re_x, &im_y, precision); // main calculation!
            res = calc_pixel_smooth(re_x, im_y); // main calculation!
            if(iflag) {
                map_to_color(res,&red,&green,&blue);
                libattopng_set_pixel(png, x, y - ((blockno-1)*blocksize), RGBA(red,green,blue)); 
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
        if(vflag) {
            fprintf(stderr,"Writing to image file %s\n", filename);
        }
        libattopng_save(png, filename);
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

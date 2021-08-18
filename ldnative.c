/*-----------------------------------------------------------------------------
 * file: ldnative.c 
 * date: Wed Aug 11 08:13:56 PDT 2021  
 * Author: Martin Casado 
 *
 * Mostly to push performance limits with some precision
 *
 *---------------------------------------------------------------------------*/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "libattopng.h"

#define RGBA(r, g, b) ((r) | ((g) << 8) | ((b) << 16))
#define FILESTR_LEN 64

static int img_w = 0, img_h = 0;
static int max_iter = 2000;

static long double c_real  = -1;
static long double c_imag  = 0; 
static long double cmplx_w = 4.; 
static long double cmplx_h = .0; // calculated in body from imgw/imgh 

                                     #
// static long double c_real  = -0.05;
// static long double c_imag  = .6805; 
// static long double c_real  = -1.769383202706626;
// static long double c_imag  =  0.0042368369187367722149;
// static long double cmplx_w =  .000000008; 


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

int inside_M1_or_M2(long double real, long double imag){
    long double c2 = ((real*real)+(imag*imag));

    // skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
    if (256.0*c2*c2 - 96.0*c2 + 32.0*real - 3.0 < 0.0)
        return 1;
    // skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
    if (16.0*(c2+2.0*real+1.0) - 1.0 < 0.0)
        return 1;

    return 0;
}


float calc_pixel_smooth(long double re_x, long double im_y) {
    int squared_rad = 256 * 256;

    long double z_real = 0.;
    long double z_imag = 0.;
    long double tmp;

    float l = 0.0;

    if(inside_M1_or_M2(re_x, im_y)){
        return 1.0; 
    }

    for(int i = 0; i < max_iter; ++i) {
        // z_real =  (z_real*z_real - z_imag*z_imag) + re_x
        tmp    = (z_real*z_real - z_imag*z_imag) + re_x;
        z_imag = (2.0 * z_real * z_imag) + im_y; 
        z_real = tmp;

        //if csquared_modulus(z_real, z_imag) >= squared_er: 
        // squared_modulus(&sm, &z_real, &z_imag, prec);
        if(((z_real*z_real)+(z_imag*z_imag)) > squared_rad){
            break; 
        }
        l += 1.0;
    }
    if(l>= max_iter) {
        return 1.0;
    }

    // sl = (l - math.log2(math.log2(csquared_modulus(z_real,z_imag)))) + 4.0;
    // sm should alrady contain sqaure_mod of z_real and z_imag
    return (l - log2(log2((z_real*z_real+z_imag*z_imag)))) + 4.0;
}

void map_to_color(float value, int* red, int* blue, int* green);

void print_header() {
    // printf(" # -- \n");
    // printf(" #\n");
    // printf(" # Image w: %d", img_w);
    // printf(" # Image h: %d", img_h);
    // printf(" #\n");
    // printf(" # Center: \n");
    // printf(" #     Re %s: \n", str_real);
    // printf(" #     Im %s: \n", str_imag);
    // printf(" #\n");
    // printf(" #     Max iter %d: \n",  max_iter);
    // printf(" #     Precision %d: \n", (int)precision);
    // printf(" #\n");
    // printf(" # -- \n");
    // printf(" #\n");
}

void usage() {
    printf("Usage: nativemandel [-i filname] [-v] [-w] [-h] [-n NUM] [-b NUM]\n");
    printf(" -i <filename> dump to image for debugging\n");
    printf(" -v debug output to stderr\n");
    printf(" -w specify image width (REQUIRED) \n");
    printf(" -h specify image height \n");
    printf(" -m <int> max iter \n");
    printf(" -n specify number of chunks \n");
    printf(" -b specify chunk number to compute \n");
    printf(" -x real value of center on complex plane \n");
    printf(" -y imag value of center on complex plane \n");
    printf(" -l width on complex plane \n");
    exit(0);
}

int main(int argc, char **argv)
{
    int ch, iflag = 0, vflag = 0;
    int numblocks = 0, blockno = 0;
    char filename[FILESTR_LEN];

    strncpy(filename, "longdouble.png",FILESTR_LEN - 1);

    while ((ch = getopt(argc, argv, "i:vw:h:n:b:x:y:l:m:")) != -1) {
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
            case 'x': 
                c_real = strtold(optarg, 0); 
                break;
            case 'y': 
                c_imag = strtold(optarg, 0); 
                break;
            case 'l': 
                cmplx_w = strtold(optarg, 0); 
                break;
            case 'b': 
                blockno = atoi(optarg); 
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

    cmplx_h = cmplx_w*((float)img_h/(float)img_w);

    long double re_start = c_real - (cmplx_w / 2.0);
    long double re_end   = c_real + (cmplx_w / 2.0);
    long double im_start = c_imag - (cmplx_h / 2.0);
    long double im_end   = c_imag + (cmplx_h / 2.0);


    //if(! iflag){
    //    print_header();
    //}

    if(vflag){
        fprintf(stderr, "img width %d\n", img_w);
        fprintf(stderr, "img height %d\n", img_h);
        fprintf(stderr, "max iter  %d\n",  max_iter);
        fprintf(stderr, "re_start %.20Lf\n", re_start);
        fprintf(stderr, "re_end   %.20Lf\n", re_end);
        fprintf(stderr, "im_start %.20Lf\n", im_start);
        fprintf(stderr, "im_end   %.20Lf\n", im_end);
        fprintf(stderr, "Complex width : %Lf\n", cmplx_w);
        fprintf(stderr, "Complex height : %Lf\n", cmplx_h);
    }

    // main loop here!!
    long double re_x;
    long double  im_y;

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
            re_x = re_start+((float)x/(float)img_w) * (re_end - re_start);
            // Im_y = (im_start) + (y / img_height) * (im_end - im_start)
            im_y = im_start + (((float)y/(float)img_h) * (im_end - im_start));

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
            fflush(stderr);
        }
        if(! iflag){
            printf("\n");
            fflush(stdout);
        }
    } // x
    fprintf(stderr,"\n");

    if(iflag){
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

    c1 +=  1 + cos( 3.0 + val*0.05 + sc0);
    c2 +=  1 + cos( 3.0 + val*0.05 + sc1);
    c3 +=  1 + cos( 3.0 + val*0.05 + sc2);

    //c1 +=  1 + cos( 3.0 + val*0.15 + sc0);
    //c2 +=  1 + cos( 3.0 + val*0.15 + sc1);
    //c3 +=  1 + cos( 3.0 + val*0.15 + sc2);

    *red    = (int)(255.*((c1/4.) * 3.) / 1.5);
    *green  = (int)(255.*((c2/4.) * 3.) / 1.5);
    *blue   = (int)(255.*((c3/4.) * 3.) / 1.5);
}

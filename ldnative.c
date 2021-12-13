/*------e----------------------------------------------------------------------
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

#define MAX_SAMPLES 128
static int samples = 65; // number of samples per pixel
static float x_spiral_offset[MAX_SAMPLES];
static float y_spiral_offset[MAX_SAMPLES];

static long double c_real  = -1;
static long double c_imag  = 0; 
static long double cmplx_w = 4.; 
static long double cmplx_h = .0; // calculated in body from imgw/imgh 

static float red   = 0.1; 
static float green = 0.2;
static float blue  = 0.6;

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

        if(((z_real*z_real)+(z_imag*z_imag)) > squared_rad){
            break; 
        }
        l += 1.0;
    }
    if(l>= max_iter) {
        return 1.0;
    }

    return (l - log2(log2((z_real*z_real+z_imag*z_imag)))) + 4.0;
}

void map_to_color(float* val, int numres, int* r, int* g, int* b);
void calc_sample_offsets ();

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
    printf("Usage: nativemandel [-i filname] [-v] [-w] [-h] [-n NUM] [-c NUM]\n");
    printf(" -i <filename> dump to image for debugging\n");
    printf(" -v debug output to stderr\n");
    printf(" -w specify image width (REQUIRED) \n");
    printf(" -h specify image height \n");
    printf(" -m <int> max iter \n");
    printf(" -n specify number of chunks \n");
    printf(" -c specify chunk number to compute \n");
    printf(" -x real value of center on complex plane \n");
    printf(" -y imag value of center on complex plane \n");
    printf(" -l width on complex plane \n");
    printf(" -r red\n");
    printf(" -g green\n");
    printf(" -b blue\n");
    exit(0);
}

int main(int argc, char **argv)
{
    int ch, iflag = 0, vflag = 0;
    int numblocks = 0, blockno = 0;
    char filename[FILESTR_LEN];

    strncpy(filename, "longdouble.png",FILESTR_LEN - 1);

    while ((ch = getopt(argc, argv, "i:vw:h:n:c:x:y:l:m:s:r:g:b:")) != -1) {
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
            case 'c': 
                blockno = atoi(optarg); 
                break;
            case 's': 
                samples = atoi(optarg); 
                break;
            case 'r': 
                red = strtof(optarg, 0); 
                break;
            case 'g': 
                green = strtof(optarg, 0); 
                break;
            case 'b': 
                blue = strtof(optarg, 0); 
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
        fprintf(stderr, "samples   %d\n",  samples);
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



    int r,g,b;
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

    // for sampling for higher precision
    float res[MAX_SAMPLES + 1];
    long double fraction_x = (re_end - re_start) / img_w;
    long double fraction_y = (im_end - im_start) / img_h;

    calc_sample_offsets();

    int sample_step = 0;
    if(samples > 1){
        sample_step = MAX_SAMPLES / (samples - 1);
    }

    int sample_count = 0;
    for(int y = 0; y < img_h; ++y){
        if(y < ystart || y > yend)
            continue;
        for(int x = 0; x < img_w; ++x){
            // map from pixels to complex coordinates 
            re_x = re_start+((float)x/(float)img_w) * (re_end - re_start);
            im_y = im_start + (((float)y/(float)img_h) * (im_end - im_start));


            // calculate center pixel 
            sample_count = 0;
            res[sample_count++] = calc_pixel_smooth(re_x, im_y); 

            if(samples > 1){
                    for(int i = 0; i < MAX_SAMPLES; i+=sample_step){
                        res[sample_count++] = calc_pixel_smooth(re_x + (fraction_x * x_spiral_offset[i]),
                                                                im_y + (fraction_y * y_spiral_offset[i]));
                    }
            }

            if(iflag) {
                map_to_color(res, sample_count, &r, &g, &b);
                libattopng_set_pixel(png, x, y - ((blockno-1)*blocksize), RGBA(r,g,b)); 
            }else{
                printf("d[(%d,%d)] = %f; ",x,y,res[0]);
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

// --
//  calculate the offsets for a spiral around the pixel using
//  the Archimedean spireal equation r = a + b*theta
//  We use a = .05 and b = .0035
//  x/y ranges end right below .5 
// --
void calc_sample_offsets () {
    float a = .05;
    float b = .0035;

    for (int i = 0; i < MAX_SAMPLES ; ++i) {
        int  theta = (float)i; // for clarity
        float r = a + (b * theta);
        x_spiral_offset[i] = r * cos(theta);
        y_spiral_offset[i] = r * sin(theta);
        //printf("%f:%f\n",x_spiral_offset[i], y_spiral_offset[i]);
        //fflush(stdout);
    }
}

void map_to_color(float* val, int numres, int* r, int* g, int* b) {

    float c1 = 0.;
    float c2 = 0.;
    float c3 = 0.;

    for( int i = 0; i < numres; ++i){
        c1 +=  1 + cos( 3.0 + val[i]*0.15 + red);
        c2 +=  1 + cos( 3.0 + val[i]*0.15 + green);
        c3 +=  1 + cos( 3.0 + val[i]*0.15 + blue);
    }
    c1 /= numres;
    c2 /= numres;
    c3 /= numres;

    *r = (int)(255.*((c1/4.) * 3.) / 1.5);
    *g = (int)(255.*((c2/4.) * 3.) / 1.5);
    *b = (int)(255.*((c3/4.) * 3.) / 1.5);
}

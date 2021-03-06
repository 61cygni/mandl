# --
# File: cjulia.pyx
# 
# Cython implementation of julia sets
#
# -- 

import math
from decimal import *

hpf = Decimal
getcontext().prec = 500 

import cython
import numpy  as np

from libc.math cimport log2
from libc.math cimport log
from libc.math cimport cos

from libc.stdio cimport printf
from libc.stdio cimport fflush 
from libc.stdio cimport stdout 

import fractalutil as fu

from algo import Algo

default_julia_c = -.8+.156j

#cdef long double julia_c_real = -.8
#cdef long double julia_c_imag = .156 
cdef long double julia_c_real = -.4
cdef long double julia_c_imag = .6 

cdef int c_sample         = 9; # number of samples per pixel
cdef long double c_width  = 0.
cdef long double c_height = 0.
cdef long double c_real   = 0.
cdef long double c_imag   = 0.
cdef float scaling_factor = 0.
cdef long double magnification = 0.

# Declare offsets for sampling
cdef int MAX_SAMPLES = 128
cdef float x_spiral_offset[128]
cdef float y_spiral_offset[128]

@cython.profile(False)
cdef inline float csquared_modulus(long double real, long double imag):
    return ((real*real)+(imag*imag))

@cython.profile(False)
cdef inline float ccalc_pixel(long double real, long double imag, int max_iter, int escape_rad):

    cdef float l = 0.0
    cdef long double z_real = real, z_imag = imag

    for i in range(0, max_iter):
        z_real, z_imag = ( z_real*z_real - z_imag*z_imag + julia_c_real,
                           2*z_real*z_imag + julia_c_imag )
        if csquared_modulus(z_real, z_imag) >= escape_rad * escape_rad:
            break
        l += 1.0    
            
    if (l >= max_iter):
        return 1.0

    sl = l - log2(log2(csquared_modulus(z_real,z_imag))) + 4.0;
    return sl

@cython.boundscheck(False)
cdef cmap_to_color(val, float red, float green, float blue, int[:] colors):

    cdef float c1 = 0.
    cdef float c2 = 0.
    cdef float c3 = 0.

    for m in val:
        c1 +=  1 + math.cos( 3.0 + m*0.15 + red);
        c2 +=  1 + math.cos( 3.0 + m*0.15 + green);
        c3 +=  1 + math.cos( 3.0 + m*0.15 + blue);

    c1 /= len(val)    
    c2 /= len(val)    
    c3 /= len(val)    

    cdef short c1int = int(255.*((c1/4.) * 3.) / 1.5)
    cdef short c2int = int(255.*((c2/4.) * 3.) / 1.5)
    cdef short c3int = int(255.*((c3/4.) * 3.) / 1.5)

    colors[0] = c1int
    colors[1] = c2int
    colors[2] = c3int

    return

@cython.profile(False)
def ccalc_cur_frame(int img_width, int img_height, long double re_start, long double re_end,
                    long double im_start, long double im_end, int max_iter, int escape_rad):
    values = {}

    cdef long double Re_x
    cdef long double Im_y

    cdef long double in_x
    cdef long double in_y

    # calculate langth of space pixel represents 
    fraction_x = (re_end - re_start) / img_width
    fraction_y = (im_end - im_start) / img_height

    sample_step = 1
    if c_sample > 1:
        sample_step = MAX_SAMPLES / (c_sample-1)
        
    printf("[")
    for x in range(0, img_width):
        for y in range(0, img_height):
            in_x = x
            in_y = y
            # ap from pixels to complex coordinates
            Re_x = (re_start) + (in_x / img_width)  *  (re_end - re_start)
            Im_y = (im_start) + (in_y / img_height) * (im_end - im_start)

            m = []
            # Call primary calculation function on center pixel 
            m.append(ccalc_pixel(Re_x, Im_y, max_iter, escape_rad))
            if c_sample <= 1:
                values[(x,y)] = m 
                continue

            # calculate samples on a spiral moving away 
            for i in range(0,MAX_SAMPLES, sample_step):
                m.append(ccalc_pixel(Re_x + (fraction_x * x_spiral_offset[i]) , 
                                     Im_y + (fraction_y * y_spiral_offset[i]), max_iter, escape_rad))

            values[(x,y)] = m 
        printf(".")
        fflush(stdout)

    printf("]\n");
    return values

class CJulia(Algo):
    
    def __init__(self, context):
        super(CJulia, self).__init__(context) 
        self.color = (.1,.2,.3) 


    def parse_options(self, opts, args):    
        global c_sample
        global julia_c_real
        global julia_c_imag

        for opt,arg in opts:
            # take color as an RGB tuple (.1,.2,.3)
            if opt in ['--setcolor']: # take colors 
                self.color = eval(arg) 
            elif opt in ['--julia-c']:
                self.julia_c = complex(arg) 

                
        c_sample    = self.context.samples 
        julia_c_real = self.julia_c.real
        julia_c_imag = self.julia_c.imag

        print('+ color to %s'%(str(self.color)))
        print('+ number of samples %d'%(c_sample))

    def set_default_params(self):

        # set a more interesting point if we're going to be doing a dive    
        if self.context.dive and not self.context.c_real: 
            self.context.c_real = hpf(-0.235125)
            self.context.c_imag = hpf(0.827215)
        if not self.context.escape_rad:        
            self.context.escape_rad   = 256.
        if not self.context.max_iter:        
            self.context.max_iter     = 512

    def calc_cur_frame(self, img_width, img_height, x, xx, xxx, xxxx):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        cdef long double re_start = c_real - (c_width / 2.)
        cdef long double re_end   = c_real + (c_width / 2.)

        cdef long double im_start = c_imag - (c_height / 2.)
        cdef long double im_end   = c_imag + (c_height / 2.)

        return ccalc_cur_frame(img_width, img_height, re_start, re_end, im_start, im_end, self.context.max_iter, self.context.escape_rad)

    def calc_pixel(self, c):
        return ccalc_pixel(c.real, c.imag, self.context.max_iter, self.context.escape_rad)


    def _map_to_color(self, val):
        c = np.zeros((3), dtype=np.int32)
        cmap_to_color(val, self.color[0], self.color[1], self.color[2], c)
        return (c[0], c[1], c[2]) 


    def map_value_to_color(self, val):
        return self._map_to_color(val)
        

    def animate_step(self, t):
        self.zoom_in()

    def setup(self):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        # since this isn't a high precision implementation, cast to native float
        c_width  = float(self.context.cmplx_width)
        c_height = float(self.context.cmplx_height)
        c_real = float(self.context.c_real)
        c_imag = float(self.context.c_imag)

        scaling_factor = self.context.scaling_factor
        magnification = self.context.magnification
        num_epochs = self.context.num_epochs

        # calculate x and y offsets for samples
        print('+ calculating sample offsets ')

        # calculate the offsets for a spiral around the pixel using
        # the Archimedean spireal equation r = a + b*theta
        # We use a = .05 and b = .0035
        # x/y ranges end right below .5 
        a = .05
        b = .0035
        for i in range(0, MAX_SAMPLES):
            theta = float(i)
            r = a + (b * theta) 
            x_spiral_offset[i] = r * math.cos(theta) 
            y_spiral_offset[i] = r * math.sin(theta)

        #for i in range(0, MAX_SAMPLES):
        #    print('x:%f y:%f'%(x_spiral_offset[i], y_spiral_offset[i]))
            
            
    def zoom_in(self, iterations=1):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        while iterations > 0:
            c_width   *= scaling_factor
            c_height  *= scaling_factor
            magnification *= scaling_factor
            iterations -= 1

            self.context.num_epochs += 1

def _instance(context):
    return CJulia(context)

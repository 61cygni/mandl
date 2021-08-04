# --
# File: csmooth.pyx
#
# Implementation of the smoothing algorithm for iteration escape method
# of drawing mandelbrot as describe and implemented by Inigo Quilez
#
# https://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
# https://www.shadertoy.com/view/4df3Rn coloring and smoothing working
#
#
# Seashell cove - -0.745+0.186j
#
# -- 
import math

import cython
import numpy  as np

from libc.math cimport log2
from libc.math cimport log
from libc.math cimport cos

import fractalutil as fu

from algo import Algo

@cython.profile(False)
cdef inline float csquared_modulus(long double real, long double imag):
    return ((real*real)+(imag*imag))


@cython.profile(False)
cdef inline bint cinside_M1_or_M2(long double real, long double imag):
    cdef double c2 = csquared_modulus(real, imag) 

    # skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
    if  256.0*c2*c2 - 96.0*c2 + 32.0*real - 3.0 < 0.0: 
        return 1 
    # skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
    if 16.0*(c2+2.0*real+1.0) - 1.0 < 0.0: 
        return 1 

    return 0 

@cython.profile(False)
cdef inline float ccalc_pixel(long double real, long double imag, int max_iter, int escape_rad):

    if cinside_M1_or_M2(real, imag):
        return 0 

    cdef float l = 0.0
    cdef long double z_real = 0., z_imag = 0.

    for i in range(0, max_iter):
        z_real, z_imag = ( z_real*z_real - z_imag*z_imag + real,
                           2*z_real*z_imag + imag )
        if csquared_modulus(z_real, z_imag) >= escape_rad * escape_rad:
            break
        l += 1.0    
            
    if (l >= max_iter):
        return 1.0

    sl = l - log2(log2(csquared_modulus(z_real,z_imag))) + 4.0;
    return sl

@cython.boundscheck(False)
cdef cmap_to_color(val, long double cmplx_width, int[:] colors):
    cdef long double magnification = 1. / cmplx_width
    if magnification <= 100:
        magnification = 100 
    denom = log(log(magnification))

    cdef float sc0 = 0.1
    cdef float sc1 = 0.2
    cdef float sc2 = 0.3

    cdef float c1 = 0.
    cdef float c2 = 0.
    cdef float c3 = 0.

    # (yellow blue 0,.6,1.0)
    c1 +=  0.5 + 0.5*cos( 3.0 + val*0.15 + sc0);
    c1 +=  0.5 + 0.5*cos( 3.0 + val*0.15 + sc0);
    c2 +=  0.5 + 0.5*cos( 3.0 + val*0.15 + sc1);
    c2 +=  0.5 + 0.5*cos( 3.0 + val*0.15 + sc1);
    c3 +=  0.5 + 0.5*cos( 3.0 + val*0.15 + sc2);
    c3 +=  0.5 + 0.5*cos( 3.0 + val*0.15 + sc2);
    cdef short c1int = int(255.*((c1/4.) * 3.) / denom)
    cdef short c2int = int(255.*((c2/4.) * 3.) / denom)
    cdef short c3int = int(255.*((c3/4.) * 3.) / denom)

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

    for x in range(0, img_width):
        for y in range(0, img_height):
            # ap from pixels to complex coordinates
            Re_x = (re_start) + (x / img_width)  *  (re_end - re_start)
            Im_y = (im_start) + (y / img_height) * (im_end - im_start)

            # Call primary calculation function here
            m = ccalc_pixel(Re_x, Im_y, max_iter, escape_rad)

            values[(x,y)] = m 

    return values

class CSmooth(Algo):
    
    def __init__(self, context):
        super(CSmooth, self).__init__(context) 
        self.color = (.1,.2,.3) 
        #self.color = (.0,.6,1.0) 

    def parse_options(self, opts, args):    
        for opt,arg in opts:
            if opt in ['--nocolor']:
                self.color = None 
            if opt in ['--setcolor']: # XXX TODO
                pass
                #self.color = (.1,.2,.3)   # dark
                #self.color = (.0,.6,1.0) # blue / yellow

    def set_default_params(self):

        # set a more interesting point if we're going to be doing a dive    
        if self.context.dive and not self.context.cmplx_center: 
            self.context.cmplx_center = self.context.ctxc(-0.235125,0.827215)
        if not self.context.escape_rad:        
            self.context.escape_rad   = 256.
        if not self.context.max_iter:        
            self.context.max_iter     = 512

    def calc_cur_frame(self, img_width, img_height, re_start, re_end, im_start, im_end):
        return ccalc_cur_frame(img_width, img_height, re_start, re_end, im_start, im_end, self.context.max_iter, self.context.escape_rad)

    def calc_pixel(self, c):
        return ccalc_pixel(c.real, c.imag, self.context.max_iter, self.context.escape_rad)


    def _map_to_color(self, val):
        c = np.zeros((3), dtype=np.int32)
        cmap_to_color(val, self.context.cmplx_width, c)
        return (c[0], c[1], c[2]) 


    def map_value_to_color(self, val):

        if self.color:
            c1 = self._map_to_color(val)
            return c1 
        else:        
            magnification = 1. / self.context.cmplx_width
            if magnification <= 100:
                magnification = 100 
            denom = math.log(math.log(magnification))
            cint = int((val * 3.) / denom)
            return (cint,cint,cint)

    def animate_step(self, t):
        self.zoom_in()

def _instance(context):
    return CSmooth(context)

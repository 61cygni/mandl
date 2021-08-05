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
cimport numpy  as np

from decimal import *

from libc.math cimport log2
from libc.math cimport log
from libc.math cimport cos

import fractalutil as fu

from algo import Algo

getcontext().prec = 64

c_width  = Decimal(0.)
c_height = Decimal(0.)
c_real   = Decimal(0.)
c_imag   = Decimal(0.)
cdef float scaling_factor = 0.
magnification = Decimal(0.)
cdef int num_epochs     = 0

cdef class cmplx_view:
    
    cdef long double width 
    cdef long double height 
    cdef long double c_real 
    cdef long double c_imag 
    cdef float scaling_factor
    cdef long double magnification
    cdef int num_epochs

    def __init__(self):
        self.width  = 0.0
        self.height = 0.0
        self.c_real  = 0.0
        self.c_imag  = 0.0
        self.scaling_factor = 0.
        self.magnification  = 0.
        self.num_epochs     = 0

    def setup(self, long double w, long double h, long double r, long double i,  float s, long double m, int n):
        self.width  = w
        self.height = h
        self.c_real = r
        self.c_imag = i
        self.scaling_factor = s
        self.magnification = m
        self.num_epochs = n

    def zoom_in(self, iterations = 1):    
        while iterations:
            self.width   *= self.scaling_factor
            self.height  *= self.scaling_factor
            self.magnification *= self.scaling_factor
            self.num_epochs += 1
            iterations -= 1
    
@cython.profile(False)
def csquared_modulus(real, imag):
    return ((real*real)+(imag*imag))

#@cython.profile(False)
#cdef inline float csquared_modulus(long double real, long double imag):
#    return ((real*real)+(imag*imag))


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
def ccalc_pixel(real, imag, int max_iter, int escape_rad):

    #if cinside_M1_or_M2(real, imag):
    #    return 0 

    cdef float l = 0.0
    z_real = Decimal(0.) 
    z_imag = Decimal(0.)

    for i in range(0, max_iter):
        z_real, z_imag = ( z_real*z_real - z_imag*z_imag + real,
                           Decimal(2)*z_real*z_imag + imag )
        if csquared_modulus(z_real, z_imag) >= escape_rad * escape_rad:
            break
        l += 1.0    
            
    if (l >= max_iter):
        return 1.0

    sl = l - math.log2(math.log2(csquared_modulus(z_real,z_imag))) + 4.0;
    return sl

#@cython.profile(False)
#cdef inline float ccalc_pixel(long double real, long double imag, int max_iter, int escape_rad):
#
#    if cinside_M1_or_M2(real, imag):
#        return 0 
#
#    cdef float l = 0.0
#    cdef long double z_real = 0., z_imag = 0.
#
#    for i in range(0, max_iter):
#        z_real, z_imag = ( z_real*z_real - z_imag*z_imag + real,
#                           2*z_real*z_imag + imag )
#        if csquared_modulus(z_real, z_imag) >= escape_rad * escape_rad:
#            break
#        l += 1.0    
#            
#    if (l >= max_iter):
#        return 1.0
#
#    sl = l - log2(log2(csquared_modulus(z_real,z_imag))) + 4.0;
#    return sl

@cython.boundscheck(False)
cdef cmap_to_color(val, long double cmplx_width, int[:] colors):
    global c_width

    #cdef long double magnification = 1. / c_width
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
def ccalc_cur_frame(int img_width, int img_height, re_start, re_end,
                    im_start, im_end, int max_iter, int escape_rad):
    values = {}



    for x in range(0, img_width):
        for y in range(0, img_height):
            in_x = Decimal(x)
            in_y = Decimal(y)
            # ap from pixels to complex coordinates
            Re_x = (re_start) + (in_x / img_width)  *  (re_end - re_start)
            Im_y = (im_start) + (in_y / img_height) * (im_end - im_start)

            # Call primary calculation function here
            m = ccalc_pixel(Re_x, Im_y, max_iter, escape_rad)

            values[(x,y)] = m 

    return values

class CSmooth(Algo):
    
    def __init__(self, context):
        super(CSmooth, self).__init__(context) 
        self.cv   = cmplx_view()
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


    def calc_cur_frame(self, img_width, img_height, x, xx, xxx, xxxx):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        re_start = Decimal(c_real - (c_width / Decimal(2.)))
        re_end   = Decimal(c_real + (c_width / Decimal(2.)))

        im_start = Decimal(c_imag - (c_height / Decimal(2.)))
        im_end   = Decimal(c_imag + (c_height / Decimal(2.)))

        print("XXXX %s %s %s %r"%(str(re_start), str(re_end), str(im_start), str(im_end)))
        
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
        #self.cv.zoom_in()

    def setup(self):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        c_width  = Decimal(self.context.cmplx_width)
        c_height = Decimal(self.context.cmplx_height)
        c_real = Decimal('-1.76938317919551501821384728608547378290574726365475143746552821652788819126')
        c_imag = Decimal('0.00423684791873677221492650717136799707668267091740375727945943565011234400')

        scaling_factor = self.context.scaling_factor
        magnification = self.context.magnification
        num_epochs = self.context.num_epochs

        #self.cv.setup(self.context.cmplx_width,
        #              self.context.cmplx_height,
        #              self.context.cmplx_center.real,
        #              self.context.cmplx_center.imag,
        #              self.context.scaling_factor,
        #              self.context.magnification,
        #              self.context.num_epochs)
                      

    def zoom_in(self, iterations=1):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        while iterations > 0:
            c_width   *= Decimal(scaling_factor)
            c_height  *= Decimal(scaling_factor)
            magnification *= scaling_factor
            num_epochs += 1
            iterations -= 1

def _instance(context):
    return CSmooth(context)

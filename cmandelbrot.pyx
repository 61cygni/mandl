# --
# File: cmandelbrot.pyc
#

import sys
import math

import cython
import fractalutil as fu

from algo import Algo
import fractalpalette as fp

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
cdef inline int ccalc_pixel(long double real, long double imag, int max_iter, int escape_rad):

    if cinside_M1_or_M2(real, imag):
        return max_iter 

    cdef long double z_real = 0., z_imag = 0.
    cdef int i

    for i in range(0, max_iter):
        z_real, z_imag = ( z_real*z_real - z_imag*z_imag + real,
                           2*z_real*z_imag + imag )
        if (z_real*z_real + z_imag*z_imag) >= escape_rad:
            return i
    if i >= max_iter:
        return max_iter
    return i 

class CMandelbrot(Algo):
    
    def __init__(self, context):
        super(CMandelbrot, self).__init__(context) 
        self.palette = fp.FractalPalette(context)

    def parse_options(self, opts, args):    
        for opt,arg in opts:
            if opt in ['--color']:
                if str(arg) == "gauss":
                    self.palette.create_gauss_gradient((255,255,255),(0,0,0))
                elif str(arg) == "exp":    
                    self.palette.create_exp_gradient((255,255,255),(0,0,0))
                elif str(arg) == "exp2":    
                    self.palette.create_exp2_gradient((0,0,0),(128,128,128))
                elif str(arg) == "list":    
                    self.palette.create_gradient_from_list()
                else:
                    print("Error: --palette arg must be one of gauss|exp|list")
                    sys.exit(0)

    def set_default_params(self):
        # This is close t Misiurewicz point M32,2
        # fractal_ctx.cmplx_center = fractal_ctx.ctxc(-.77568377, .13646737)

        if self.context.dive and not self.context.cmplx_center:
            self.context.cmplx_center = self.context.ctxc(-1.769383179195515018213,0.00423684791873677221)

    def calc_pixel(self, c):
        m = ccalc_pixel(c.real, c.imag, self.context.max_iter, self.context.escape_rad)
        self.palette.raw_calc_from_algo(m)
        return m 

    def map_value_to_color(self, loc, values):
        return self.palette.map_value_to_color(values[loc])
        
    def pre_image_hook(self):
        self.palette.calc_hues()

    def cache_loaded(self, values):
        self.palette.per_frame_reset() # just in case
        for coords in values:
            self.palette.raw_calc_from_algo(values[coords])
        

    def per_frame_reset(self):
        self.palette.per_frame_reset()
        
    def animate_step(self, t):
        self.zoom_in()

def _instance(context):
    return CMandelbrot(context)

# --
# File: smooth.py
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

import fractalutil as fu

from algo import Algo

class Smooth(Algo):
    
    def __init__(self, context):
        super(Smooth, self).__init__(context) 
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
            self.context.cmplx_center = complex(-0.235125,0.827215)
        if not self.context.escape_rad:        
            self.context.escape_rad   = 256.
        if not self.context.max_iter:        
            self.context.max_iter     = 512

    def calc_pixel(self, real, imag):
        # we don't do high precision so snap back to native floats
        c = complex(float(real), float(imag))

        if fu.inside_M1_or_M2(c):
            return 0.0

        B = self.context.escape_rad 
        l = 0.0
        z = complex(0)

        for i in range(0, self.context.max_iter):
            z = z*z + c
            if fu.squared_modulus(z) > B*B:
                break
            l += 1.0    

        if (l >= self.context.max_iter):
            return 1.0
        
        # This is the generalized smoothing algorithm
        #sn = n - log(log(length(z))/log(B))/log(2.0); 
        # below is the optimized algorithm just for mandelbrot
        sl = l - math.log2(math.log2(fu.squared_modulus(z))) + 4.0;

        #ret =  fu.linear_interpolate_f(l, sl, 1.0)

        return sl 


    def _map_to_color(self, val):

        c1 = 0.
        c2 = 0.
        c3 = 0.

        # (yellow blue 0,.6,1.0)
        c1 +=  1 + math.cos( 3.0 + val*0.15 + self.color[0]);
        c2 +=  1 + math.cos( 3.0 + val*0.15 + self.color[1]);
        c3 +=  1 + math.cos( 3.0 + val*0.15 + self.color[2]);

        c1int = int(255.*((c1/4.) * 3.) / 1.5)
        c2int = int(255.*((c2/4.) * 3.) / 1.5)
        c3int = int(255.*((c3/4.) * 3.) / 1.5)

        return (c1int,c2int,c3int)

    def map_value_to_color(self, val):
        return self._map_to_color(val)
        

    def animate_step(self, t):
        self.zoom_in()

def _instance(context):
    return Smooth(context)

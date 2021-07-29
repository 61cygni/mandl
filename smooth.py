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
        self.color = None
        self.neighbors = {}

    def parse_options(self, opts, args):    
        for opt,arg in opts:
            if opt in ['--color']:
                self.color = (.1,.2,.3)   # dark
                #self.color = (.0,.6,1.0) # blue / yellow
                #self.color = (.0,.6,1.0)

    def set_default_params(self):
        self.context.cmplx_center = self.context.ctxc(-1.769383179195515018213,0.00423684791873677221)
        self.context.escape_rad   = 256.
        self.context.max_iter     = 512

    def _calc_pixel(self, c):

        if fu.inside_M1_or_M2(c):
            return 0.0

        B = self.context.escape_rad 
        l = 0.0
        z = self.context.ctxc(0)

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

    def calc_pixel(self, c):
        self.neighbors[c] = []

        step_w = self.context.cmplx_width  / float(self.context.img_width)
        step_h = self.context.cmplx_height / float(self.context.img_height)

        step_w = step_w / 4.0
        step_h = step_h / 4.0 # half steps :)

        c

        val = self._calc_pixel(c)

        # sample c and 4 points around it 
        #new_c = complex(c.real - step_w, c.imag)
        #val = val+ self._calc_pixel(new_c)
        #new_c = complex(c.real + step_w, c.imag)
        #val = val + self._calc_pixel(new_c)
        #new_c = complex(c.real, c.imag - step_h)
        #val = val + self._calc_pixel(new_c)
        #new_c = complex(c.real, c.imag + step_h)
        #val = val + self._calc_pixel(new_c)
        #return val / 5.

        return val

    def _map_to_color(self, val):
        magnification = 1. / self.context.cmplx_width
        if magnification <= 100:
            magnification = 100 
        denom = math.log(math.log(magnification))

        c1 = 0.
        c2 = 0.
        c3 = 0.

        # (yellow blue 0,.6,1.0)
        c1 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[0]);
        c1 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[0]);
        c2 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[1]);
        c2 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[1]);
        c3 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[2]);
        c3 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[2]);
        c1int = int(255.*((c1/4.) * 3.) / denom)
        c2int = int(255.*((c2/4.) * 3.) / denom)
        c3int = int(255.*((c3/4.) * 3.) / denom)
        return (c1int,c2int,c3int)

    def map_value_to_color(self, loc, vals):

        val = vals[loc]

        magnification = 1. / self.context.cmplx_width
        if magnification <= 100:
            magnification = 100 
        denom = math.log(math.log(magnification))

        if self.color:
            c1 = self._map_to_color(val)
            return c1 
        else:        
            #magnification = 1. / self.context.cmplx_width
            cint = int((val * 3.) / denom)
            return (cint,cint,cint)
        

    def animate_step(self, t):
        self.zoom_in()

def _instance(context):
    return Smooth(context)

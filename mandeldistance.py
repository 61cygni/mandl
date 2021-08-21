# --
# File: mandeldistance.py
#
# https://www.shadertoy.com/view/lsX3W4
# --

from algo import Algo

import math

import fractalpalette as fp


def squared_modulus(z):
    return ((z.real*z.real)+(z.imag*z.imag))

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)


class MandelDistance(Algo):

    def __init__(self, context):
        super(MandelDistance, self).__init__(context) 

    def set_default_params(self):
        pass
        # fractal_ctx.cmplx_center = fractal_ctx.ctxc(-.77568377, .13646737)
        #self.context.cmplx_center = self.context.ctxc("-0.05+.6805j")

    def calc_pixel(self, c):

        c2 = c.real*c.real + c.imag*c.imag
        # skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
        if  256.0*c2*c2 - 96.0*c2 + 32.0*c.real - 3.0 < 0.0: 
            return 0.0
        # skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
        if 16.0*(c2+2.0*c.real+1.0) - 1.0 < 0.0: 
            return 0.0

        # iterate
        di =  1.0;
        z  = complex(0.0);
        m2 = 0.0;
        dz = complex(0.0);
        for i in range(0,self.context.max_iter): 
            if m2>1024.0 : 
              di=0.0 
              break

            # Z' -> 2·Z·Z' + 1
            dz = 2.0*complex(z.real*dz.real-z.imag*dz.imag, z.real*dz.imag + z.imag*dz.real) + complex(1.0,0.0);
                
            # Z -> Z² + c           
            z = complex( z.real*z.real - z.imag*z.imag, 2.0*z.real*z.imag ) + c;
                
            m2 = squared_modulus(z) 

        # distance  
        # d(c) = |Z|·log|Z|/|Z'|
        d = 0.5*math.sqrt(squared_modulus(z)/squared_modulus(dz))*math.log(squared_modulus(z));
        if  di>0.5:
            d=0.0
        
        return d             

    def map_value_to_color(self, val):
        zoo = .1 
        zoom_level = 1. / (self.context.cmplx_width)
        d = clamp( pow(zoom_level*val/zoo,0.1), 0.0, 1.0 );
        cint = int(d*255)

        return (cint,cint,cint)

    def animate_step(self, t):
        self.zoom_in()

def _instance(context):
    return MandelDistance(context)


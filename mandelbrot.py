# --
# File: mandelbrot.py
#
# Basic escape iteration method for calculating the mandlebrot set
#
# Code cribbed from all over the place ... notably :
#
# https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
# http://linas.org/art-gallery/escape/escape.html
#
# Misiurewicz points also cribbed from all over 
#
# https://mrob.com/pub/muency/misiurewiczpoint.html
# https://www.youtube.com/watch?v=u1pwtSBTnPU&t=274s
#
#
# MPs:
#
# 0.4244 + 0.200759i;
from algo import Algo


class Mandelbrot(Algo):
    
    def __init__(self, context):
        super(Mandelbrot, self).__init__(context) 

    def calc_pixel(self, c):

        z = self.context.ctxc(0)
        n = 0

        squared_escape = self.context.escape_rad * self.context.escape_rad

        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane)
        #
        # However, instead we just square both sides of the inequality to
        # avoid the sqrt
        while ((z.real*z.real)+(z.imag*z.imag)) <= squared_escape  and n < self.context.max_iter:
            z = z*z + c
            n += 1

        if n >= self.context.max_iter:
            return self.context.max_iter
        
        # The following code smooths out the colors so there aren't bands
        # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
        if self.context.smoothing:
            z = z*z + c; n+=1 # a couple extra iterations helps
            z = z*z + c; n+=1 # decrease the size of the error
            mu = n + 1 - math.log(math.log2(abs(z)))
            return mu 
        else:    
            return n 

    def animate_step(self, t):
        self.context.zoom_in()

def _instance(context):
    return Mandelbrot(context)

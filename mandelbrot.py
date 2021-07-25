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
import fractalpalette as fp

class Mandelbrot(Algo):
    
    def __init__(self, context):
        super(Mandelbrot, self).__init__(context) 
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


    def _calc_pixel(self, c):

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

    def calc_pixel(self, c):
        m = self._calc_pixel(c)
        self.palette.raw_calc_from_algo(m)
        return m 

    def map_value_to_color(self, val):
        return self.palette.map_value_to_color(val)
        
    def pre_image_hook(self):
        self.palette.calc_hues()

    def per_frame_reset(self):
        self.palette.per_frame_reset()
        
    def animate_step(self, t):
        self.zoom_in()

def _instance(context):
    return Mandelbrot(context)

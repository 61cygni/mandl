import math
from algo import Algo

import fractalpalette as fp

default_julia_c = -.8+.145j

class Julia(Algo):
    
    def __init__(self, context):
        super(Julia, self).__init__(context) 
        self.palette = fp.FractalPalette(context)

        self.julia_c    = None
        self.julia_list = None
        self.julia_orig = None


    def parse_options(self, opts, args):    

        for opt,arg in opts:
            if opt in ['--julia-c']:
                self.julia_c = complex(arg) 
            elif opt in ['--julia-walk']:
                self.julia_list = eval(arg)  # expects a list of complex numbers
                if len(self.julia_list) <= 1:
                    print("Error: List of complex numbers for Julia walk must be at least two points")
                    sys.exit(0)
                self.julia_c    = self.julia_list[0]
            elif opt in ['--color']:
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
        self.context.cmplx_center = self.context.ctxc(0,0)

        if not self.julia_c:
            print(" * Warning no julia c value specified, defaulting to %s"%(str(default_julia_c)))
            self.julia_c = default_julia_c

        # needed to uniquely identify cache frame
        self.algo_specific_cache = self.julia_c    


    # Some interesting c values
    # c = complex(-0.8, 0.156)
    # c = complex(-0.4, 0.6)
    # c = complex(-0.7269, 0.1889)

    def _calc_pixel(self, z0):
        z = z0
        n = 0
        while abs(z) <= 2 and n < self.context.max_iter:
            z = z*z + self.julia_c 
            n += 1

        if n == self.context.max_iter:
            return self.context.max_iter

        return n + 1 - math.log(math.log2(abs(z)))

    # Use Bresenham's line drawing algo for a simple walk between two
    # complex points
    def animate_step(self, t):

        if type(self.julia_list) == type(None):
            self.zoom_in()
            return
    
        duration   = self.context.duration
        fps        = self.context.fps

        # duration of a leg
        leg_d      = float(duration) / float(len(self.julia_list) - 1)
        # which leg are we walking?
        leg        = math.floor(float(t) / leg_d)
        # how far along are we on that leg?
        timeslice  = float(duration) / (float(duration) * float(fps))
        fraction   = (float(t) - (float(leg) * leg_d)) / (leg_d - timeslice)

        #print("T %f Leg %d leg_d %d Fraction %f"%(t,leg,leg_d,fraction))

        cp1 = self.julia_list[leg]
        cp2 = self.julia_list[leg + 1]


        if self.julia_orig != cp1:
            self.julia_orig = cp1

        x0 = self.julia_orig.real
        x1 = cp2.real 
        y0 = self.julia_orig.imag 
        y1 = cp2.imag 

        new_x = ((x1 - x0)*fraction) + x0
        new_y = ((y1 - y0)*fraction) + y0 
        self.julia_c = complex(new_x, new_y)

        # needed to uniquely identify cache frame
        self.algo_specific_cache = self.julia_c    

    def calc_pixel(self, z0):
        m = self._calc_pixel(z0)
        self.palette.raw_calc_from_algo(m)
        return m 

    def map_value_to_color(self, loc, vals):
        return self.palette.map_value_to_color(vals[loc])

    def pre_image_hook(self):
        self.palette.calc_hues()

    def cache_loaded(self, values):
        self.palette.per_frame_reset() # just in case
        for coords in values:
            self.palette.raw_calc_from_algo(values[coords])

    def per_frame_reset(self):
        self.palette.per_frame_reset()

def _instance(context):
    return Julia(context)

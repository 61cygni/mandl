import math
from algo import Algo

default_julia_c = -.8+.145j


class Julia(Algo):
    
    def __init__(self, context):
        super(Julia, self).__init__(context) 

    def parse_options(self, opts, args):    

        for opt,arg in opts:
            if opt in ['--julia-c']:
                self.context.julia_c = complex(arg) 

        if not self.context.julia_c:
            print("Error: no julia c value specified, defaulting to %s"%(str(default_julia_c)))
            self.context.julia_c = default_julia_c


    # Some interesting c values
    # c = complex(-0.8, 0.156)
    # c = complex(-0.4, 0.6)
    # c = complex(-0.7269, 0.1889)

    def calc_pixel(self, z0):
        z = z0
        n = 0
        while abs(z) <= 2 and n < self.context.max_iter:
            z = z*z + self.context.julia_c 
            n += 1

        if n == self.context.max_iter:
            return self.context.max_iter

        return n + 1 - math.log(math.log2(abs(z)))

def _instance(context):
    return Julia(context)

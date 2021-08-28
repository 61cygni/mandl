import sys

import decimal
hpf = decimal.Decimal

class Algo(object):

    def __init__(self, context):
        self.context = context
        self.algo_specific_cache = None

    def parse_options(self, opts, args):    
        pass

    def set_default_params(self):
        pass

    def setup(self):
        pass

    def animate_step(self, t):
        pass

    def burn_string(self):
        return None
   
    def pre_image_hook(self):
        pass

    def per_frame_reset(self):
        pass

    def cache_loaded(self, values):
        pass 

    def zoom_in(self, iterations=1):
        while iterations:
            self.context.cmplx_width   *= hpf(self.context.scaling_factor)
            self.context.cmplx_height  *= hpf(self.context.scaling_factor)
            self.context.magnification *= self.context.scaling_factor
            self.context.num_epochs += 1
            iterations -= 1

    def calc_cur_frame(self, img_width, img_height, re_start, re_end, im_start, im_end):

        print(" + [algo]calculating frame at center %.20e %.20ei"%(self.context.c_real, self.context.c_imag))
        print(" + Re_start %f Im_start %f"%(re_start, im_start))
        values = {}

        print("[",end="")
        sys.stdout.flush()
        for x in range(0, img_width):
            for y in range(0, img_height):
                # ap from pixels to complex coordinates
                Re_x = (re_start) + hpf(x / img_width)  * (re_end - re_start)
                Im_y = (im_start) + hpf(y / img_height) * (im_end - im_start)

                # Call primary calculation function here
                m = self.calc_pixel(Re_x, Im_y)

                values[(x,y)] = m 
            print(".",end="")
            sys.stdout.flush()
        print("]")
        sys.stdout.flush()

        return values
        

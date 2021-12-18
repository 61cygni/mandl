import sys
import math

import decimal
hpf = decimal.Decimal

class Algo(object):

    def __init__(self, context):
        self.context = context
        self.algo_specific_cache = None

        self.x_spiral_offset = []
        self.y_spiral_offset = []
        self.MAX_SAMPLES     = 128

        # calculate the offsets for a spiral around the pixel using
        # the Archimedean spireal equation r = a + b*theta
        # We use a = .05 and b = .0035
        # x/y ranges end right below .5 
        a = .05
        b = .0035
        for i in range(0, self.MAX_SAMPLES):
            theta = float(i)
            r = a + (b * theta) 
            self.x_spiral_offset.append(r * math.cos(theta))
            self.y_spiral_offset.append(r * math.sin(theta))

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

        # calculate langth of space pixel represents 
        fraction_x = (re_end - re_start) / img_width
        fraction_y = (im_end - im_start) / img_height

        sample_step = 1

        if self.context.samples > 1:
            sample_step = int(self.MAX_SAMPLES / (self.context.samples - 1))
        print("XXX SAMPLES %d:%d:"%(self.context.samples, sample_step))

        print("[",end="")
        sys.stdout.flush()
        for x in range(0, img_width):
            for y in range(0, img_height):
                # ap from pixels to complex coordinates
                Re_x = (re_start) + hpf(x / img_width)  * (re_end - re_start)
                Im_y = (im_start) + hpf(y / img_height) * (im_end - im_start)

                m = []
                # Call primary calculation function here
                m.append(self.calc_pixel(Re_x, Im_y))

                if self.context.samples <= 1:
                    values[(x,y)] = m 
                    sys.stdout.flush()
                    continue

                # calculate samples on a spiral moving away 
                for i in range(0, self.MAX_SAMPLES, sample_step):
                    #print("%f:%f"%(self.x_spiral_offset[i], self.y_spiral_offset[i]))
                    m.append(self.calc_pixel(Re_x + (fraction_x * hpf(self.x_spiral_offset[i])) , 
                                             Im_y + (fraction_y * hpf(self.y_spiral_offset[i]))))

                values[(x,y)] = m 
            print(".",end="")
            sys.stdout.flush()
        print("]")
        sys.stdout.flush()

        return values
        

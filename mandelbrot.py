# --
# File: mandelbrot.py
# 
# Driver file for playing around with the Mandelbrot set 
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
#
# --

import getopt
import sys
import math
import os

from decimal import * # Surely more specific on this will be better, sorry
import multiprocessing # Can't actually make this work yet - gonna need pickling?

from collections import defaultdict

import numpy as  np
import mpmath as mp
import flint

import moviepy.editor as mpy
from scipy.stats import norm

from moviepy.audio.tools.cuts import find_audio_period

from PIL import Image, ImageDraw, ImageFont

MANDL_VER = "0.1"

DECIMAL_HIGH_PRECISION_SIZE = 16  # 16 places is roughly equivalent to 53 bits for float64, right?
#FLINT_HIGH_PRECISION_SIZE = 53 # 53 is how many bits are in float64
FLINT_HIGH_PRECISION_SIZE = 200 
MPMATH_HIGH_PRECISION_SIZE = 53 # 53 bits in a float64
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class MandlPalette:
    """
    Color gradient
    """

    # Color in RGB 
    def __init__(self):
        self.gradient_size = 1024
        self.palette = []


    def linear_interpolate(color1, color2, fraction):
        new_r = int(math.ceil((color2[0] - color1[0])*fraction) + color1[0])
        new_g = int(math.ceil((color2[1] - color1[1])*fraction) + color1[1])
        new_b = int(math.ceil((color2[2] - color1[2])*fraction) + color1[2])
        return (new_r, new_g, new_b)


    def create_gauss_gradient(self, c1, c2, mu=0.0, sigma=1.0): 
        
        print("+ Creating color gradient with gaussian decay")
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)

        self.palette.append(c2)

        x = 0.0
        while len(self.palette) <= self.gradient_size:
            g = gaussian(x,0,.10)
            c = MandlPalette.linear_interpolate(c1, c2, g) 
            self.palette.append(c)
            x = x + (1./(self.gradient_size+1))

    def create_exp_gradient(self, c1, c2, decay_const = 1.01):    

        print("+ Creating color gradient with exponential decay")
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)


        x = 0.0

        while len(self.palette) <= self.gradient_size:
            fraction = math.pow(math.e,-15.*x)
            c = MandlPalette.linear_interpolate(c1, c2, fraction)
            self.palette.append(c)
            x = x + (1. / 1025) 


    def create_exp2_gradient(self, c1, c2, decay_const = 1.01):    

        print("+ Creating color gradient with varying exponential decay")
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)


        x = 0.0
        c = c1
        # Do a very quick decent for the first 1/16 
        while len(self.palette) <= float(self.gradient_size)/32.:
            fraction = math.pow(math.e,-30.*x)
            c = MandlPalette.linear_interpolate((255,255,255), c1, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 

        last_c = c
        x = 0.0
        # Do another quick decent back to first color for the next 1/16 
        while len(self.palette) <= 2.*(float(self.gradient_size) / 16.):
            fraction = math.pow(math.e,-15.*x)
            c = MandlPalette.linear_interpolate((255,255,255), last_c, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 

        last_c = c
        x = 0.0
        # Do another quick decent back to first color for the next 1/16 
        while len(self.palette) <= 2.*(float(self.gradient_size) / 16.):
            fraction = math.pow(math.e,-5.*x)
            c = MandlPalette.linear_interpolate((255,255,255), last_c, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 

        last_c = c
        # For the remaining go back to white 
        x = 0.0
        while len(self.palette) <= self.gradient_size :
            fraction = math.pow(math.e,-2.*x)
            c = MandlPalette.linear_interpolate((255,255,255),last_c, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 


    def create_normal_gradient(self, c1, c2, decay_const = 1.05):    
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)

        fraction = 1.
        while len(self.palette) <= self.gradient_size:
            c = MandlPalette.linear_interpolate(c1, c2, fraction)
            self.palette.append(c)
            fraction = fraction / decay_const
            

    # Create 255 value gradient
    # Use the following trivial linear interpolation algorithm
    # (color2 - color1) * fraction + color1
    def create_gradient_from_list(self, color_list = [(255,255,255),(0,0,0),(255,255,255),(0,0,0),(255,255,255),(0,0,0),(241, 247, 215),(255,204,204),(204,204,255),(255,255,204),(255,255,255)]):

        print("+ Creating color gradient from color list")

        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)
        
        #the first few colors are critical, so just fill by hand.
        self.palette.append((0,0,0))
        self.palette.append((0,0,0))
        self.palette.append(MandlPalette.linear_interpolate((0,0,0),(255,255,255),.2))
        self.palette.append(MandlPalette.linear_interpolate((0,0,0),(255,255,255),.4))
        self.palette.append(MandlPalette.linear_interpolate((0,0,0),(255,255,255),.6))
        self.palette.append(MandlPalette.linear_interpolate((0,0,0),(255,255,255),.8))

        # The magic number 6 here just denotes the previous colors we
        # filled by hand
        section_size = int(float(self.gradient_size-6)/float(len(color_list)-1))

        for c in range(0, len(color_list) - 1): 
            for i in range(0, section_size+1): 
                fraction = float(i)/float(section_size)
                new_color = MandlPalette.linear_interpolate(color_list[c], color_list[c+1], fraction)
                self.palette.append(new_color)
        while len(self.palette) < self.gradient_size:
            c = self.palette[-1]
            self.palette.append(c)
        #assert len(self.palette) == self.gradient_size    

    def make_frame(self, t):    

        IMG_WIDTH=1024
        IMG_HEIGHT=100
        
        im = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        color_iter = 0
        for x in range(0,IMG_WIDTH):
            color = self.palette[color_iter]
            for y in range(0,IMG_HEIGHT):
                draw.point([x, y], color) 
            color_iter += 1 
        

        return np.array(im)

    def __iter__(self):
        return self.palette

    def __getitem__(self, index):
             return self.palette[index]

    def display(self):
        clip = mpy.VideoClip(self.make_frame, duration=64)
        clip.preview(fps=1) #fps 1 is really all that works
        

class MandlContext:
    """
    The context for a single dive
    """

    def __init__(self, ctxf = None, ctxc = None, mp = None):

        if not ctxf:
            self.ctxf = float
        else:
            self.ctxf = ctxf
        if not ctxc:
            self.ctxc = complex
        else:    
            self.ctxc = ctxc 
        if not mp:
            self.mp = math

        self.img_width  = 0 # int : Wide of Image in pixels
        self.img_height = 0 # int

        self.cmplx_width = 0.0 # width of visualization in complex plane
        self.cmplx_height = 0.0
        self.cmplx_width_decimal = Decimal('0.0')
        self.cmplx_height_decimal = Decimal('0.0')
        self.cmplx_width_flint = flint.arb('0.0')
        self.cmplx_height_flint = flint.arb('0.0')
    
        # point we're going to dive into 
        self.cmplx_center = self.ctxc(0.0) # center of image in complex plane
        self.cmplx_center_real_decimal = Decimal('0.0')
        self.cmplx_center_imag_decimal = Decimal('0.0')
        self.cmplx_center_flint = flint.acb('0.0')

        self.max_iter      = 0  # int max iterations before bailing
        self.escape_rad    = 0. # float radius mod Z hits before it "escapes" 
        self.escape_squared = 0.0 # float squared radius, memoized

        self.scaling_factor = 0.0 #  float amount to zoom each epoch
        self.num_epochs     = 0   #  int, nuber of epochs into the dive

        self.set_zoom_level = 0   # Zoom in prior to the dive

        self.smoothing      = False # bool turn on color smoothing
        self.snapshot       = False # Generate a single, high res shotb

        self.precision = 17 # int decimal precision for calculations

        self.duration  = 0  # int  duration of clip in seconds
        self.fps = 0 # int  number of frames per second

        self.palette = None
        self.burn_in = False
        self.use_high_precision = False
        self.high_precision_type = 'flint'

        self.cache_path = "cache"
        self.build_cache = False
        self.invalidate_cache = False

        self.verbose = 0 # how much to print about progress

    def zoom_in(self, iterations=1):
        if self.use_high_precision == True:
            if self.high_precision_type == 'decimal':
                oldPrecision = getcontext().prec
                getcontext().prec = DECIMAL_HIGH_PRECISION_SIZE
                while iterations:
                    self.cmplx_width_decimal *= Decimal(self.scaling_factor)
                    self.cmplx_height_decimal *= Decimal(self.scaling_factor)
                    self.num_epochs += 1
                    iterations -= 1
                getcontext().prec = oldPrecision
            else: # flint
                while iterations:
                    self.cmplx_width_flint *= self.scaling_factor
                    self.cmplx_height_flint *= self.scaling_factor
                    self.num_epochs += 1
                    iterations -= 1
        else: # Default native python, or maybe mpmath
            while iterations:
                self.cmplx_width  *= self.scaling_factor
                self.cmplx_height *= self.scaling_factor
                self.num_epochs += 1
                iterations -= 1

    def mandelbrot(self, c):
        z = self.ctxc(0)
        n = 0

        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane)
        #
        # However, instead we just square both sides of the inequality to
        # avoid the sqrt
        while ((z.real*z.real)+(z.imag*z.imag)) <= self.escape_squared  and n < self.max_iter:
            z = z*z + c
            n += 1

        if n== self.max_iter:
            return self.max_iter
        
        # The following code smooths out the colors so there aren't bands
        # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
        if self.smoothing:
            z = z*z + c; n+=1 # a couple extra iterations helps
            z = z*z + c; n+=1 # decrease the size of the error
            mu = n + 1 - math.log(self.mp.log2(abs(z)))
            return mu 
        else:    
            return n 

    def mandelbrot_flint(self, c):
        z = flint.acb(0.0)
        n = 0

        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane)
        #
        # However, instead we just square both sides of the inequality to
        # avoid the sqrt
        while ((z.real*z.real)+(z.imag*z.imag)) <= self.escape_squared and n < self.max_iter:
            z = z*z + c
            n += 1
        return n

    def mandelbrot_decimal(self, param_complex_array):
        """Takes the 2 complex number components and returns the 
        number of iterations it took to become greater than the escape value.

        TODO: really need tests to make sure precision is being properly preserved"""
        param_real = param_complex_array[0]
        param_imag = param_complex_array[1]

        z_real = Decimal('0.0')
        z_imag = Decimal('0.0')
        n = 0

        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane)
        # 
        # However, instead we just square both sides of the inequality to
        # avoid the sqrt
        while((z_real*z_real) + (z_imag*z_imag)) <= self.escape_squared and n < self.max_iter:
            prev_z_real = z_real
            prev_z_imag = z_imag

            z_real = prev_z_real * prev_z_real - prev_z_imag * prev_z_imag + Decimal(param_real)
            z_imag = prev_z_real * prev_z_imag * Decimal('2.0') + Decimal(param_imag)
            n += 1

        return n

    def linspace_decimal(self, paramFirst, paramLast, quantity):
        first = Decimal(paramFirst)
        last = Decimal(paramLast)
        dataRange = last - first
        answers = np.zeros((quantity, 1), dtype=object)

        for x in range(0, quantity):
            answers[x] = first + (dataRange / (quantity - 1)) * x

        return answers

    def linspace(self, paramFirst, paramLast, quantity):
        first = paramFirst
        last = paramLast
        dataRange = last - first
        answers = np.zeros((quantity, 1), dtype=object)

        for x in range(0, quantity):
            answers[x] = first + (dataRange / (quantity - 1)) * x

        return answers
        
    def load_frame(self, frame_number, cache_path, build_cache = True, invalidate_cache = False):
        """Cache-aware data tuple loading or calculating"""

        cache_file_name = os.path.join(cache_path, u"%d.npy" % frame_number)
        #print("cache file %s" % cache_file_name)

        if invalidate_cache == True and os.path.exists(cache_file_name) == True and os.path.isfile(cache_file_name):
            os.remove(cache_file_name)
               
        frame_data = np.zeros((1,1), dtype=np.uint8) 
        if os.path.exists(cache_file_name) == False:
            #print("Calculating epoch data")
            if self.use_high_precision == True:
                if self.high_precision_type == 'decimal':
                    #print("  (decimal high precision)")
                    frame_data = self.calculate_epoch_data_decimal(frame_number)
                else:
                    #print("  (flint high precision)")
                    frame_data = self.calculate_epoch_data_flint()
            else:
                #print("  (normal precision as-compiled, perhaps mpmath?)")
                frame_data = self.calculate_epoch_data(frame_number)

            if build_cache == True:
                #print("Writing cache file")
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path)

                np.save(cache_file_name, frame_data) 
        else:
            #print("Loading cache file")
            frame_data = np.load(cache_file_name, allow_pickle=True)

        return frame_data

    def calculate_epoch_data(self, t):
        """Generates the data tuple for every pixel position in the frame"""
    
        # Use center point to determines the box in the complex plane
        # we need to calculatee
        re_start = self.ctxf(self.cmplx_center.real - (self.cmplx_width / 2.))
        re_end =   self.ctxf(self.cmplx_center.real + (self.cmplx_width / 2.))

        im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
        im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))

        if self.verbose > 0:
            print("MandlContext starting epoch %d re range %f %f im range %f %f center %f + %f i .... " %\
                  (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center.real, self.cmplx_center.imag),
                  end = " ")

        values = np.zeros((self.img_width, self.img_height), dtype=np.uint8)
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                # map from pixels to complex coordinates
                Re_x = self.ctxf(re_start) + (self.ctxf(x) / self.ctxf(self.img_width - 1))  * \
                       self.ctxf(re_end - re_start)
                Im_y = self.ctxf(im_start) + (self.ctxf(y) / self.ctxf(self.img_height - 1)) * \
                       self.ctxf(im_end - im_start)

                c = self.ctxc(Re_x, Im_y)

                m = self.mandelbrot(c)

                values[x,y] = m
        return values

    def calculate_epoch_data_flint(self):
        # Use center point to determines the box in the complex plane
        # we need to calculatee
        re_start = self.cmplx_center_flint.real - (self.cmplx_width_flint / 2.0)
        re_end = self.cmplx_center_flint.real + (self.cmplx_width_flint / 2.0)

        im_start = self.cmplx_center_flint.imag - (self.cmplx_height_flint / 2.0)
        im_end = self.cmplx_center_flint.imag + (self.cmplx_height_flint / 2.0)

        if self.verbose > 0:
            print("MandlContext starting epoch %d re range %s %s im range %s %s center %s + %s i .... " %\
                  (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center_flint.real, self.cmplx_center_flint.imag),
                  end = " ")

        # Create 1D arrays of the real and complex ranges for each pixel of the current image
        real_1d = self.linspace(re_start, re_end, self.img_width)
        imag_1d = self.linspace(im_start, im_end, self.img_height)

        # Create two parallel 2D arrays, for the row+column combinations of real and imaginary values,  
        pixel_real_2d, pixel_imag_2d = np.meshgrid(imag_1d, real_1d)

        # Combine two parallel 2D arrays into a 3D array
        # e.g.
        # arr1 = [A, B, C, D]
        # arr2 = [1, 2, 3]
        # pixel_inputs_3 = [[[A,1],[A,2],[A,3]], [[B,1],[B,2],[B,3]], [[C,1],[C,2],[C,3]], [[D,1],[D,2],[D,3]]]
        #
        # Worth noting that vstack seems to need backwards params to get the ordering right
        pixel_inputs_3d = np.vstack(([pixel_imag_2d.T], [pixel_real_2d.T])).T
        #print("shape of pixel_inputs_3d: %s" % str(pixel_inputs_3d.shape))

        pixel_values_2d = np.zeros((self.img_width, self.img_height), dtype=np.uint8)

        show_row_progress = False
        if show_row_progress == True:
            for x in range(0, self.img_width):
                for y in range(0, self.img_height):
                    pixel_values_2d[x,y] = self.mandelbrot_flint(flint.acb(pixel_inputs_3d[x,y,0], pixel_inputs_3d[x,y,1]))
                print("%d-" % x, end="")
                sys.stdout.flush()
        else:
            # Probalby not necessary, but lining up the 2-element subarray
            pixel_inputs_1d = pixel_inputs_3d.reshape(pixel_inputs_3d.shape[0] * pixel_inputs_3d.shape[1], 2)
            #print("shape of pixel_inputs_1d: %s" % str(pixel_inputs_1d.shape))
            pixel_values_1d = np.array([self.mandelbrot_flint(flint.acb(complexAsArray[0], complexAsArray[1])) for complexAsArray in pixel_inputs_1d])
    
            pixel_values_2d = pixel_values_1d.reshape(self.img_width, self.img_height, 1)
            pixel_values_2d = np.squeeze(pixel_values_2d, axis=2) # Incantation to remove a sub-array level
            #print("shape of pixel_values_2d: %s" % str(pixel_values_2d.shape))

        #
        # Graveyard of failed attempts at further vectorizing this, maybe there's a clue in here
        # somewhere...
        #

        # In search of efficient ways to apply the map, and getting stuck with various issues
        # like pickling, which are keeping me from using multiprocessing.Pool

        #nope
        #theFunction = np.vectorize(self.mandelbrot_flint)
        #pixel_values_1d = theFunction(pixel_inputs_1d)

        #pixel_inputs_1d = pixel_inputs.reshape(1,self.img_width * self.img_height)
        # hmm, I see the problem# pixel_values_3d = self.mandelbrot_flint(pixel_inputs_3d)
        #pixel_values_1d = self.mandelbrot_decimal(pixel_inputs_1d)

        #pixel_values_1d = map(self.mandelbrot_flint, pixel_inputs_1d)
        #pixel_values_1d = np.array(list(map(self.mandelbrot_flint, pixel_inputs_1d)))
        #print("shape of pixel_values_1d: %s" % str(pixel_values_1d.shape))

        # Can't pikcle... hmm
        #mandelpool = multiprocessing.Pool(processes = 1)
        #pixel_values_1d = mandelpool.map(self.mandelbrot_flint, pixel_inputs_1d)
        #mandelpool.close()
        #mandelpool.join()

        return pixel_values_2d


    def calculate_epoch_data_decimal(self,t):
        oldPrecision = getcontext().prec
        getcontext().prec = DECIMAL_HIGH_PRECISION_SIZE

        # Use center point to determines the box in the complex plane
        # we need to calculatee
        re_start = self.cmplx_center_real_decimal - (self.cmplx_width_decimal / Decimal('2.0'))
        re_end = self.cmplx_center_real_decimal + (self.cmplx_width_decimal / Decimal('2.0'))

        im_start = self.cmplx_center_imag_decimal - (self.cmplx_height_decimal / Decimal('2.0'))
        im_end = self.cmplx_center_imag_decimal + (self.cmplx_height_decimal / Decimal('2.0'))

        if self.verbose > 0:
            print("MandlContext starting epoch %d re range %s %s im range %s %s center %s + %s i .... " %\
                  (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center_real_decimal, self.cmplx_center_imag_decimal),
                  end = " ")
        # Create 1D arrays of the real and complex ranges for each pixel of the current image
        real_1d = self.linspace_decimal(re_start, re_end, self.img_width)
        imag_1d = self.linspace_decimal(im_start, im_end, self.img_height)

        # Create two parallel 2D arrays, for the row+column combinations of real and imaginary values,  
        pixel_real_2d, pixel_imag_2d = np.meshgrid(imag_1d, real_1d)

        # Combine two parallel 2D arrays into a 3D array
        # Worth noting that vstack seems to need backwards params to get the ordering right
        # arr1 = [A, B, C, D]
        # arr2 = [1, 2, 3]
        # output_arr = [[[A,1],[A,2],[A,3]], [[B,1],[B,2],[B,3]], [[C,1],[C,2],[C,3]], [[D,1],[D,2],[D,3]]]
        pixel_inputs_3d = np.vstack(([pixel_imag_2d.T], [pixel_real_2d.T])).T
        #print("shape of pixel_inputs_3d: %s" % str(pixel_inputs_3d.shape))

        pixel_values_2d = np.zeros((self.img_width, self.img_height), dtype=np.uint8)

        show_row_progress = False
        if show_row_progress == True:
            for x in range(0, self.img_width):
                for y in range(0, self.img_height):
                    pixel_values_2d[x,y] = self.mandelbrot_decimal(pixel_inputs_3d[x,y])
                print("%d." % x, end="")
                sys.stdout.flush()
        else:
            # Probalby not necessary, but lining up the 2-element subarray
            pixel_inputs_1d = pixel_inputs_3d.reshape(pixel_inputs_3d.shape[0] * pixel_inputs_3d.shape[1], 2)
            #print("shape of pixel_inputs_1d: %s" % str(pixel_inputs_1d.shape))
            pixel_values_1d = np.array([self.mandelbrot_decimal(np.array(complexAsArray)) for complexAsArray in pixel_inputs_1d])
    
            pixel_values_2d = pixel_values_1d.reshape(self.img_width, self.img_height, 1) 
            pixel_values_2d = np.squeeze(pixel_values_2d, axis=2) # Incantation to remove a sub-array level
            #print("shape of pixel_values_2d: %s" % str(pixel_values_2d.shape))
        
        getcontext().prec = oldPrecision

        return pixel_values_2d

    def next_epoch(self, t, snapshot_filename = None):
        """Called for each frame of the animation. Will calculate
        current view, and then zoom in"""
    
        ## Use center point to determines the box in the complex plane
        ## we need to calculatee
        #re_start = self.ctxf(self.cmplx_center.real - (self.cmplx_width / 2.))
        #re_end =   self.ctxf(self.cmplx_center.real + (self.cmplx_width / 2.))
        #
        #im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
        #im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))
        # 
        #if self.verbose > 0:
        #    print("MandlContext starting epoch %d re range %f %f im range %f %f center %f + %f i .... " %\
        #        (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center.real, self.cmplx_center.imag),
        #        end = " ")

        # Used to create a histogram of the frequency of iteration
        # deppths retured by the mandelbrot calculation. Helpful for 
        # color selection since many iterations never come up so you
        # loose fidelity by not focusing on those heavily used
        hist = defaultdict(int) 
        values = np.zeros((self.img_width, self.img_height), dtype=np.uint8)

        # Really need to build a reference function that's the frame number oracle...
        # Until then, here's a quick and dirty way
        quick_frame_number = math.floor(view_ctx.fps * t) 
        pixel_values_2d = self.load_frame(quick_frame_number, build_cache=self.build_cache, cache_path=self.cache_path, invalidate_cache=self.invalidate_cache) 

        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                if pixel_values_2d[x,y] < self.max_iter:
                    hist[math.floor(pixel_values_2d[x,y])] += 1

        #oldImage = Image.fromarray(values.astype('uint8'))
        #oldImage.save("calc_plain_%d.gif" % self.num_epochs)
        #newImage = Image.fromarray(pixel_values_2d.astype('uint8')) 
        #newImage.save("calc_precise_%d.gif" % self.num_epochs)
        
        total = sum(hist.values())
        hues = []
        h = 0

        # calculate percent of total for each iteration
        for i in range(self.max_iter):
            if total :
                h += hist[i] / total
            hues.append(h)
        hues.append(h)

        im = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                m = pixel_values_2d[x,y] 

                if not self.palette:
                    c = 255 - int(255 * hues[math.floor(m)]) 
                    color=(c, c, c)
                elif self.smoothing:
                    c1 = self.palette[1024 - int(1024 * hues[math.floor(m)])]
                    c2 = self.palette[1024 - int(1024 * hues[math.ceil(m)])]
                    color = MandlPalette.linear_interpolate(c1,c2,.5) 
                else:
                    color = self.palette[1024 - int(1024 * hues[math.floor(m)])]

                # Plot the point
                draw.point([x, y], color) 


        #print("Finished iteration RErange %f:%f (re width: %f)"%(RE_START, RE_END, RE_END - RE_START))
        #print("Finished iteration IMrange %f:%f (im height: %f)"%(IM_START, IM_END, IM_END - IM_START))

        if self.burn_in == True:
            center_string = str(self.cmplx_center)
            range_string = "(%f,%f)" % (self.cmplx_width, self.cmplx_height)

            if self.use_high_precision == True:
                if self.high_precision_type == 'decimal':
                    center_string = "(%s, %si)" % (str(self.cmplx_center_real_decimal), str(self.cmplx_center_imag_decimal))
                    range_string = "(%s,%s)" % (str(self.cmplx_width_decimal), str(self.cmplx_height_decimal))
                else: # flint
                    center_string = str(self.cmplx_center_flint)
                    range_string = "(%s,%s)" % (str(self.cmplx_width_flint), str(self.cmplx_height_flint))

            burn_in_text = u"%d center: %s\n    size: %s" %\
                (quick_frame_number, center_string, range_string)

            burn_in_location = (10,10)
            burn_in_margin = 5 
            burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
            burn_in_size = burn_in_font.getsize_multiline(burn_in_text)
            draw.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
            draw.text(burn_in_location, burn_in_text, 'white', burn_in_font)

        # Zoom in by scaling factor
        self.zoom_in()

        if self.verbose > 0:
            print("Done]")
        
        if snapshot_filename:
            return im.save(snapshot_filename,"gif")
        else:    
            return np.array(im)
        

    def __repr__(self):
        return """\
[MandlContext Img W:{w:d} Img H:{h:d} Cmplx W:{cw:.20f}
Cmplx H:{ch:.20f} Complx Center:{cc:s} Scaling:{s:f} Epochs:{e:d} Max iter:{mx:d}]\
""".format(
        w=self.img_width,h=self.img_height,cw=self.cmplx_width,ch=self.cmplx_height,
        cc=str(self.cmplx_center),s=self.scaling_factor,e=self.num_epochs,mx=self.max_iter); 

class MediaView: 
    """
    Handle displaying to gif / mp4 / screen etc.  
    """

    def make_frame(self, t):
        return self.ctx.next_epoch(t)

    def __init__(self, duration, fps, ctx):
        self.duration  = duration
        self.fps       = fps
        self.ctx       = ctx
        self.banner    = False 
        self.vfilename = None 


    def intro_banner(self):
        # Generate a text clip
        w,h = self.ctx.img_width, self.ctx.img_height
        banner_text = u"%dx%d center %s duration=%d fps=%d" %\
                       (w, h, str(self.ctx.cmplx_center), self.duration, self.fps)


        txt = mpy.TextClip(banner_text, font='Amiri-regular',
                           color='white',fontsize=12)

        txt_col = txt.on_color(size=(w + txt.w,txt.h+6),
                          color=(0,0,0), pos=(1,'center'), col_opacity=0.6)

        txt_mov = txt_col.set_position((0,h-txt_col.h)).set_duration(4)

        return mpy.CompositeVideoClip([self.clip,txt_mov]).subclip(0,self.duration)

    def create_snapshot(self):    
    
        if not self.vfilename:
            self.vfilename = "snapshot.gif"
        
        self.ctx.next_epoch(-1,self.vfilename)


    def run(self):

        # Check whether we need to zoom in prior to calculation

        if self.ctx.set_zoom_level > 0:
            print("Zooming in by %d epochs" % (self.ctx.set_zoom_level))
            while self.ctx.set_zoom_level > 0:
                self.ctx.zoom_in()
                self.ctx.set_zoom_level -= 1
            

        if self.ctx.snapshot == True:
            self.create_snapshot()
            return

        self.clip = mpy.VideoClip(self.make_frame, duration=self.duration)

        if self.banner:
            self.clip = self.intro_banner()

        if not self.vfilename:
            self.clip.preview(fps=1) #fps 1 is really all that works
        elif self.vfilename.endswith(".gif"):
            self.clip.write_gif(self.vfilename, fps=self.fps)
        elif self.vfilename.endswith(".mp4"):
            self.clip.write_videofile(self.vfilename,
                                  fps=self.fps, 
                                  audio=False, 
                                  codec="mpeg4")
        else:
            print("Error: file extension not supported, must be gif or mp4")
            sys.exit(0)
            

    def __repr__(self):
        return """\
[MediaView duration {du:f} FPS:{f:d} Output:{vf:s}]\
""".format(du=self.duration,f=self.fps,vf=str(self.vfilename))

# For now, use global context for a single dive per run

flint.prec = FLINT_HIGH_PRECISION_SIZE  # Sets flint's precision (in bits)
mp.mp.prec = MPMATH_HIGH_PRECISION_SIZE # Sets mpmath's precision (in bits)

# The extra parameter to instantiation is the way to run with mpmath
#mandl_ctx = MandlContext(ctxc=mp.mpc)

# mpmath isn't as complete an alternate implementation as the others, because params
# can't change instantiation types while running set_default_params with the
# current setup, and second because I didn't check all instantiations to
# be correctly typed.
mandl_ctx = MandlContext()
view_ctx  = MediaView(16, 16, mandl_ctx)


# --
# Default settings for the dive. All of these can be overridden from the
# command line
# --
def set_default_params():
    global mandl_ctx

    mandl_ctx.img_width  = 1024
    mandl_ctx.img_height = 768 

    cmplx_width_str = '5.0'
    cmplx_height_str = '3.5'
    mandl_ctx.cmplx_width  = mandl_ctx.ctxf(float(cmplx_width_str))
    mandl_ctx.cmplx_height = mandl_ctx.ctxf(float(cmplx_height_str))
    mandl_ctx.cmplx_width_decimal = Decimal(cmplx_width_str)
    mandl_ctx.cmplx_height_decimal = Decimal(cmplx_height_str)
    mandl_ctx.cmplx_width_flint = flint.arb(cmplx_width_str)
    mandl_ctx.cmplx_height_flint = flint.arb(cmplx_height_str)

    # This is close t Misiurewicz point M32,2
    # mandl_ctx.cmplx_center = mandl_ctx.ctxc(-.77568377, .13646737)
    center_real_str = '-1.769383179195515018213'
    center_imag_str = '0.00423684791873677221'
    mandl_ctx.cmplx_center = mandl_ctx.ctxc(float(center_real_str),float(center_imag_str))
    mandl_ctx.cmplx_center_real_decimal = Decimal(center_real_str)
    mandl_ctx.cmplx_center_imag_decimal = Decimal(center_imag_str)
    mandl_ctx.cmplx_center_flint = flint.acb(center_real_str, center_imag_str)

    mandl_ctx.scaling_factor = .97
    mandl_ctx.num_epochs     = 0

    mandl_ctx.max_iter       = 255
    mandl_ctx.escape_rad     = 4.
    mandl_ctx.escape_squared = mandl_ctx.escape_rad * mandl_ctx.escape_rad

    mandl_ctx.precision      = 100

    view_ctx.duration       = 16
    view_ctx.fps            = 16


def set_preview_mode():
    global mandl_ctx

    print("+ Running in preview mode ")

    mandl_ctx.img_width  = 300
    mandl_ctx.img_height = 200

    mandl_ctx.cmplx_width  = 3.
    mandl_ctx.cmplx_height = 2.5 

    mandl_ctx.scaling_factor = .75
    mandl_ctx.escape_rad     = 4.

    view_ctx.duration       = 4
    view_ctx.fps            = 4

def set_snapshot_mode():
    global mandl_ctx

    print("+ Running in snapshot mode ")

    mandl_ctx.snapshot = True

    mandl_ctx.img_width  = 3000
    mandl_ctx.img_height = 2000 

    mandl_ctx.max_iter   = 2000

    mandl_ctx.cmplx_width  = 3.
    mandl_ctx.cmplx_height = 2.5 

    mandl_ctx.scaling_factor = .99 # set so we can zoom in more accurately
    mandl_ctx.escape_rad     = 4.

    view_ctx.duration       = 0
    view_ctx.fps            = 0


def parse_options():
    global mandl_ctx

    argv = sys.argv[1:]

    
    opts, args = getopt.getopt(argv, "pd:m:s:f:z:w:h:c:",
                               ["preview",
                                "duration=",
                                "max-iter=",
                                "img-w=",
                                "img-h=",
                                "center=",
                                "scaling-factor=",
                                "snapshot=",
                                "zoom=",
                                "fps=",
                                "gif=",
                                "mpeg=",
                                "verbose=",
                                "palette-test=",
                                "color=",
                                "burn",
                                "flint",
                                "decimal",
                                "cache-path=",
                                "build-cache",
                                "invalidate-cache",
                                "banner",
                                "smooth"])

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode()
        if opt in ['-s', '--snapshot']:
            set_snapshot_mode()

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration = float(arg) 
        elif opt in ['-m', '--max-iter']:
            mandl_ctx.max_iter = int(arg)
        elif opt in ['-w', '--img-w']:
            mandl_ctx.img_width = int(arg)
        elif opt in ['-h', '--img-h']:
            mandl_ctx.img_height = int(arg)
        elif opt in ['-c', '--center']:
            mandl_ctx.cmplx_center= complex(arg)
        elif opt in ['-h', '--img-h']:
            mandl_ctx.img_height = int(arg)
        elif opt in ['--scaling-factor']:
            mandl_ctx.scaling_factor = float(arg)
        elif opt in ['-z', '--zoom']:
            mandl_ctx.set_zoom_level = int(arg)
        elif opt in ['-f', '--fps']:
            view_ctx.fps = int(arg)
        elif opt in ['--smooth']:
            mandl_ctx.smoothing = True 
        elif opt in ['--palette-test']:
            m = MandlPalette()
            if str(arg) == "gauss":
                m.create_gauss_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp":    
                m.create_exp_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp2":    
                m.create_exp2_gradient((0,0,0),(128,128,128))
            elif str(arg) == "list":    
                m.create_gradient_from_list()
            else:
                print("Error: --palette-test arg must be one of gauss|exp|list")
                sys.exit(0)
            m.display()
            sys.exit(0)
        elif opt in ['--color']:
            m = MandlPalette()
            if str(arg) == "gauss":
                m.create_gauss_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp":    
                m.create_exp_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp2":    
                m.create_exp2_gradient((0,0,0),(128,128,128))
            elif str(arg) == "list":    
                m.create_gradient_from_list()
            else:
                print("Error: --palette-test arg must be one of gauss|exp|list")
                sys.exit(0)
            mandl_ctx.palette = m
        elif opt in ['--burn']:
            mandl_ctx.burn_in = True
        elif opt in ['--decimal']:
            mandl_ctx.use_high_precision = True
            mandl_ctx.high_precision_type = "decimal"
        elif opt in ['--flint']:
            mandl_ctx.use_high_precision = True
            mandl_ctx.high_precision_type = "flint"
        elif opt in ['--cache-path']:
            mandl_ctx.cache_path = arg
        elif opt in ['--build-cache']:
            mandl_ctx.build_cache = True
        elif opt in ['--invalidate-cache']:
            mandl_ctx.invalidate_cache = True
        elif opt in ['--banner']:
            view_ctx.banner = True
        elif opt in ['--verbose']:
            verbosity = int(arg)
            if verbosity not in [0,1,2,3]:
                print("Invalid verbosity level (%d) use range 0-3"%(verbosity))
                sys.exit(0)
            mandl_ctx.verbose = verbosity
        elif opt in ['--gif']:
            if view_ctx.vfilename != None:
                print("Error : Already specific media type %s"%(view_ctx.vfilename))
                sys.exit(0)
            view_ctx.vfilename = arg
        elif opt in ['--mpeg']:
            if view_ctx.vfilename != None:
                print("Error : Already specific media type %s"%(view_ctx.vfilename))
                sys.exit(0)
            view_ctx.vfilename = arg

    print(mandl_ctx)
    print(view_ctx)

if __name__ == "__main__":

    print("++ mandlebort.py version %s" % (MANDL_VER))
    
    set_default_params()
    parse_options()

    view_ctx.run()

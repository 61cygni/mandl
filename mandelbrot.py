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

import pickle

import multiprocessing # Can't actually make this work yet - gonna need pickling?

from collections import defaultdict

import numpy as  np

import moviepy.editor as mpy
from scipy.stats import norm

from moviepy.audio.tools.cuts import find_audio_period

from PIL import Image, ImageDraw, ImageFont

MANDL_VER = "0.1"

#FLINT_HIGH_PRECISION_SIZE = 16 # 53 is how many bits are in float64
FLINT_HIGH_PRECISION_SIZE = 53 # 53 is how many bits are in float64
#FLINT_HIGH_PRECISION_SIZE = 200 

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
        

class DiveMathSupport:
    """
    Toolbox for math functions that need to be type-aware, so we
    can support different back-end math libraries.

    This exists because globally swapping out types doesn't do enough
    to keep numeric types all in line. Some calculations also need additional 
    steps to make the math behave right, such as if we were using 
    the Decimal library, and had to keep separate real and imaginary
    components of a complex number ourselves.

    Trying to preserve types where possible, without forcing casting, because sometimes
    all the math operations will already work for custom numeric types.
    """

    def createComplex(self, realComponent, imagComponent):
        """Compatible complex types will return values for .real() and .imag()"""
        return complex(float(realComponent), float(imagComponent)) 

    def createFloat(self, floatValue):
        return float(floatValue)

    def floor(self, value):
        return math.floor(value)

    def scaleValueByFactorForIterations(self, startValue, overallZoomFactor, iterations):
        """
        Shortcut calculate the starting point for the last frame's properties, 
        which we'll use for instantiation with specific widths.  This is
        because if any keyframes are added, the best we can do is measure an
        effective zoom factor between any 2 frames.

        Somehow, this seems to be ok for flint too?!
        """
        return startValue * (overallZoomFactor ** iterations)

    def createLinspace(self, paramFirst, paramLast, quantity):
        """Attempt at a mostly type-agnostic linspace(), seems to work with flint types too"""
        dataRange = paramLast - paramFirst
        answers = np.zeros((quantity), dtype=object)

        for x in range(0, quantity):
            answers[x] = paramFirst + dataRange * (x / (quantity - 1))

        return answers

    def createLinspaceAroundValuesCenter(self, valuesCenter, spreadWidth, quantity):
        """
        Attempt at a mostly type-agnostic linspace-from-center-and-width

        Turns out, we didn't *really* need this for using flint, though it
        was an important for a Python/Decimal version to be able to have a
        type-specific implementation in what was a DiveMeshDecimal function
        """
        return self.createLinspace(valuesCenter - spreadWidth * 0.5, valuesCenter + spreadWidth * 0.5, quantity)

    def interpolateLogTo(self, startX, startY, endX, endY, targetX):
        """
        Probably want additional log-defining params, but for now, let's just bake in one equation
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            # y1 = a * ln(b * x1)
            # y2 = a * ln(b * x2)
            # a = (y1-y2) / ln(x1/x2)
            # b = exp(((y2 * ln(x1)) - (y1 * ln(x2))) / (y1-y2))
            #
            # y = a * ln(b * x)
            # answer = a * ln(b * targetX)
            # answer = ((y1-y2)/ln(x1/x2)) * ln(exp(((y2 * ln(x1)) - (y1 * ln(x2))) / (y1-y2)) * targetX)
            # This kinda worked, but keeps the ln function so you just see part of 
            # some curve through the query points.  Really, want to make sure the asymptote 
            # is at or near the start (or end).
            # aVal = ((startY-endY) / math.log(startX / endX))
            # bVal = math.exp(startY / aVal) / startX
            # return aVal * math.log(bVal * targetX)

            # This version seems like it worked fine, but ended up with a pretty wrong
            # transition
            #
            # But, if we set the crossover point to the first point (add 1 to xVal), then all we
            # need to do is solve for a, right?  Also, using the 0-crossover as the
            # first point means the asymptote can't be hit, because it's 1 less than 
            # the start.
            #
            # endY = a * ln(endX + 1 - startX) + startY
            # a = (endY - startY) / ln(endX - startX + 1)

            aVal = (endY - startY) / math.log(endX - startX + 1)
            return aVal * math.log(targetX - startX + 1) + startY 

    def interpolateRootTo(self, startX, startY, endX, endY, targetX):
        """ 
        Iterative multiplications of window sizes for zooming means we want to be able to
        interpolate between two points using the root if the frame count between them as
        the scale factor
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            root  = endX - startX
            scaleFactor = (endY / startY) ** (1 / root)
            return startY * (scaleFactor ** targetX)

    def interpolateQuadraticEaseIn(self, startX, startY, endX, endY, targetX):
        """
        QuadraticEaseIn puts the majority of changes in the start of the X range.
        
        I *think* this is just the backwards solution to QuadraticEaseOut?  
        So, just swap in and out points?

        Probably want additional quadratic-defining params, but for now, let's just bake in one equation
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            return self.interpolateQuadraticEaseOut(endX, endY, startX, startY, targetX)

    def interpolateQuadraticEaseOut(self, startX, startY, endX, endY, targetX):
        """
        QuadraticEaseOut leaves the majority of changes to the end of the X range.

        Probably want additional quadratic params, but for now, let's just bake in one equation
        which uses the first point as the vertex, and passes through the second point.
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            # Find a, given that the start point is the vertex, and the parabola passes 
            # through the other point
            # y = a * (x - h)**2 + k
            # y = a * (x - startX)**2 + startY
            # endY = a * (endX - startX)**2 + startY
            # a = ((endY-startY)/((endX-startX)**2)
            #
            # answer = a * (targetX - startX)**2 + startY
            # answer = ((endY-startY)/((endX-startX)**2) * ((targetX-startX)**2) + startY
            return (endY-startY)/((endX-startX)**2) * ((targetX-startX)**2) + startY

    def interpolateQuadraticEaseInOut(self, startX, startY, endX, endY, targetX):
        """
        QuadraticEaseInOut finds a (linear) midpoint of the range, then leaves the 
        majority of the change to the middle of the range, easing both out of 
        the start, and into the end along separate parabolas.

        CAUTION: This forces the 'middle' X value to floor().  The idea is that X 
        will be frame numbers, so it should gracefully handle the inflection 
        around an integer-valued frame, instead of doing a (maybe more 
        complicated?) rounding inference.  
        The problem is that this means it isn't a universal solver.
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            midpointFrame = self.floor(((endX-startX) * .5)) + startX
            midpointY = self.interpolateLinear(startX, startY, endX, endY, midpointFrame) 
            if targetX <= midpointFrame:
                return self.interpolateQuadraticEaseOut(startX, startY, midpointFrame, midpointY, targetX)
            else:
                return self.interpolateQuadraticEaseIn(midpointFrame, midpointY, endX, endY, targetX)

    def interpolateLinear(self, startX, startY, endX, endY, targetX):
        # x1=2, y1=1, x2=12, y2=2, targetX = 7 
        # valuesRange = endX - startX = 10
        # targetPercent = (targetX - startX) / valuesRange = .5
        # valuesDomain = endY - startY = 1
        # answer = targetPercent * valuesDomain + startY = 1.5
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            return (((targetX - startX)/ (endX - startX)) * (endY - startY)) + startY

    def mandelbrot(self, c, escapeSquared, maxIter, shouldSmooth=False):
        z = self.createComplex(0, 0)
        n = 0

        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane)
        #
        # However, instead we just square both sides of the inequality to
        # avoid the sqrt
        while ((z.real*z.real)+(z.imag*z.imag)) <= escapeSquared  and n < maxIter:
            z = z*z + c
            n += 1
#
        if n == maxIter:
            return maxIter
        
        # The following code smooths out the colors so there aren't bands
        # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
        if shouldSmooth == True:
            z = z*z + c; n+=1 # a couple extra iterations helps
            z = z*z + c; n+=1 # decrease the size of the error
            mu = n + 1 - math.log(math.log(abs(z)))
            return mu 
        else:    
            return n 

class DiveMathSupportFlint(DiveMathSupport):
    """
    Overrides to instantiate flint-specific complex types

    Looks like flint types are safe to use in base's createLinspace()
    """
    def __init__(self):
        super().__init__()

        self.flint = __import__('flint') # Only imports if you instantiate this DiveMathSupport subclass.
        self.flint.prec = FLINT_HIGH_PRECISION_SIZE  # Sets flint's precision (in bits)

    def createComplex(self, realComponent, imagComponent):
        return self.flint.acb(realComponent, imagComponent)

    def createFloat(self, floatValue):
        return self.flint.arb(floatValue)

    def floor(self, value):
        return self.flint.arb(value).floor()

    def interpolateLogTo(self, startX, startY, endX, endY, targetX):
        """
        Probably want additional log-defining params, but for now, let's just bake in one equation
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            aVal = (endY - startY) / (self.flint.arb(endX - startX + 1).log())
            return aVal * (self.flint.arb(targetX - startX + 1).log()) + startY 

    def mandelbrot(self, c, escapeSquared, maxIter, shouldSmooth=False):
        """ 
        NOTE: Smoothing maybe should be only a post-processing step?  Maybe not?

        This flint-specific implementation only really does flint-y logs in the smoothing.
        The Decimal-specific implementation needed some extra steps, but I've ditched that one.
        """
        z = self.createComplex(0, 0)
        n = 0

        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane)
        #
        # However, instead we just square both sides of the inequality to
        # avoid the sqrt
        #
        # Important to cast the escape answer back to float, or the arb gets sheared bizarrely
        while (float((z.real*z.real)+(z.imag*z.imag))) <= escapeSquared  and n < maxIter:
            z = z*z + c
            n += 1

        if n == maxIter:
            return maxIter
        
        # The following code smooths out the colors so there aren't bands
        # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
        if shouldSmooth == True:
            z = z*z + c; n+=1 # a couple extra iterations helps
            z = z*z + c; n+=1 # decrease the size of the error

            #mu = n + 1 - math.log(self.mp.log2(abs(z))) # (and, there IS NO log2 in mpmath... hrm)
            # Maybe this is the right way to do the same thing?!
            mu = n + 1 - z.abs_lower().log().log()
            return mu 
        else:    
            return n 


class MandlContext:
    """
    The context for a single dive
    """

    def __init__(self, math_support=DiveMathSupport()):
        self.math_support = math_support

        self.img_width  = 0 # int : Wide of Image in pixels
        self.img_height = 0 # int

        self.cmplx_width = self.math_support.createFloat(0.0)
        self.cmplx_height = self.math_support.createFloat(0.0)

        # point we're going to dive into 
        self.cmplx_center = self.math_support.createComplex(0.0, 0.0) # center of image in complex plane

        self.max_iter      = 0  # int max iterations before bailing
        self.escape_rad    = 0. # float radius mod Z hits before it "escapes" 
        self.escape_squared = 0.0 # float squared radius, (redundant, but helpful)

        self.scaling_factor = 0.0 #  float amount to zoom each epoch
        self.num_epochs     = 0   #  int, nuber of epochs into the dive

        self.set_zoom_level = 0   # Zoom in prior to the dive

        self.smoothing      = False # bool turn on color smoothing
        self.snapshot       = False # Generate a single, high res shotb

        self.duration  = 0  # int  duration of clip in seconds
        self.fps = 0 # int  number of frames per second

        self.palette = None
        self.burn_in = False

        # Shifting from the MandlContext being the oracle of frame information, to the Timeline being the oracle.
        # Rather than keeping 'current frame' info in the context, we just keep the timeline, and
        # query it for frame-specific parameters to render with.
        self.timeline = None

        self.project_name = "default_project"
        self.cache_path = u"%s_cache" % self.project_name
        self.build_cache = False
        self.invalidate_cache = False

        self.verbose = 0 # how much to print about progress

    def zoom_in(self, iterations=1):
        while iterations:
            self.cmplx_width  *= self.scaling_factor
            self.cmplx_height *= self.scaling_factor
            self.num_epochs += 1
            iterations -= 1

    def mandelbrot(self, c):
        z = self.math_support.createComplex(0, 0)
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
            mu = n + 1 - math.log(math.log(abs(z)))
            return mu 
        else:    
            return n 

    def load_frame(self, frame_number, cache_path, build_cache = True, invalidate_cache = False):
        """Cache-aware data tuple loading or calculating"""

        cache_file_name = os.path.join(cache_path, u"%d.npy" % frame_number)
        #print("cache file %s" % cache_file_name)

        if invalidate_cache == True and os.path.exists(cache_file_name) == True and os.path.isfile(cache_file_name):
            os.remove(cache_file_name)
               
        frame_data = np.zeros((1,1), dtype=np.uint8) 
        if os.path.exists(cache_file_name) == False:
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
        re_start = self.math_support.createFloat(self.cmplx_center.real - (self.cmplx_width / 2.))
        re_end =   self.math_support.createFloat(self.cmplx_center.real + (self.cmplx_width / 2.))

        im_start = self.math_support.createFloat(self.cmplx_center.imag - (self.cmplx_height / 2.))
        im_end   = self.math_support.createFloat(self.cmplx_center.imag + (self.cmplx_height / 2.))

        if self.verbose > 0:
            print("MandlContext starting epoch %d re range %f %f im range %f %f center %f + %f i .... " %\
                  (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center.real, self.cmplx_center.imag),
                  end = " ")

        values = np.zeros((self.img_width, self.img_height), dtype=np.uint8)
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                # map from pixels to complex coordinates
                Re_x = re_start + (x / (self.img_width - 1)) * (re_end - re_start)
                Im_y = im_start + (y / (self.img_height - 1)) * (im_end - im_start)

                c = self.math_support.createComplex(Re_x, Im_y)
                m = self.mandelbrot(c)

                values[x,y] = m
        return values

    def render_frame_number(self, frame_number, snapshot_filename=None):
        """
        Load or calculate the frame data.

        Once we have frame data, can perform histogram and coloring
        """
        (pixel_values_2d, frame_metadata) = self.timeline.loadResultsForFrameNumber(frame_number, build_cache=self.build_cache, cache_path=self.cache_path, invalidate_cache=self.invalidate_cache)

        # Capturing the transpose of our array, because it looks like I mixed
        # up rows and cols somewhere along the way.
        pixel_values_2d = pixel_values_2d.T

        # Used to create a histogram of the frequency of iteration
        # deppths retured by the mandelbrot calculation. Helpful for 
        # color selection since many iterations never come up so you
        # loose fidelity by not focusing on those heavily used
        hist = defaultdict(int) 
        values = np.zeros((self.img_width, self.img_height), dtype=np.uint8)

        #print("shape of things to come: %s" % str(pixel_values_2d.shape))

        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                if pixel_values_2d[x,y] < self.max_iter:
                    hist[math.floor(pixel_values_2d[x,y])] += 1

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
            burn_in_text = u"%d center: %s\n    realw: %s imagw: %s" % (frame_number, frame_metadata['mesh_center'], frame_metadata['real_width'], frame_metadata['imag_width'])

            burn_in_location = (10,10)
            burn_in_margin = 5 
            burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
            burn_in_size = burn_in_font.getsize_multiline(burn_in_text)
            draw.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
            draw.text(burn_in_location, burn_in_text, 'white', burn_in_font)

        #if self.verbose > 0:
        #    print("Done]")
        
        if snapshot_filename:
            return im.save(snapshot_filename,"gif")
        else:    
            return np.array(im)
        
    def next_epoch(self, t, snapshot_filename = None):
        """Called for each frame of the animation. Will calculate
        current view, and then zoom in"""
    
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

#            if self.use_high_precision == True:
#                if self.high_precision_type == 'flint':
#                    center_string = str(self.cmplx_center_flint)
#                    range_string = "(%s,%s)" % (str(self.cmplx_width_flint), str(self.cmplx_height_flint))

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
[MandlContext Img W:{w:d} Img H:{h:d} Cmplx W:{cw:s}
Cmplx H:{ch:s} Complx Center:{cc:s} Scaling:{s:f} Epochs:{e:d} Max iter:{mx:d}]\
""".format(
        w=self.img_width,h=self.img_height,cw=str(self.cmplx_width),ch=str(self.cmplx_height),
        cc=str(self.cmplx_center),s=self.scaling_factor,e=self.num_epochs,mx=self.max_iter); 

class MediaView: 
    """
    Handle displaying to gif / mp4 / screen etc.  
    """

    def make_frame(self, t):
        return self.ctx.render_frame_number(self.frame_number_from_time(t))

    def __init__(self, duration, fps, ctx):
        self.duration  = duration
        self.fps       = fps
        self.ctx       = ctx
        self.banner    = False 
        self.vfilename = None 

    def frame_number_from_time(self, t):
        return math.floor(self.fps * t) 

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

        if self.ctx.snapshot == True:
            self.create_snapshot()
            return

        self.ctx.timeline =  self.construct_simple_timeline()

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

    def construct_simple_timeline(self):
        """
        Transitional. 

        Basically, let's construct a timeline from one set of start/end points,
        as defined by the current context.
        """
        frame_count = self.duration * self.fps  
        if math.floor(frame_count) != frame_count:
            frame_count = math.floor(frame_count) + 1

        overall_zoom_factor = self.ctx.scaling_factor

        start_width_real = self.ctx.cmplx_width
        start_width_imag = self.ctx.cmplx_height
        
        # Check whether we need to zoom in prior to calculation
        if self.ctx.set_zoom_level > 0:
            print("Zooming in by %d epochs" % (self.ctx.set_zoom_level))
            start_width_real = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_width, overall_zoom_factor, self.ctx.set_zoom_level) 
            start_width_imag = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_height, overall_zoom_factor, self.ctx.set_zoom_level)
       
        # Calculate the endpoint widths, at the last desired frame
        end_width_real = self.ctx.math_support.scaleValueByFactorForIterations(start_width_real, overall_zoom_factor, frame_count - 1)
        end_width_imag = self.ctx.math_support.scaleValueByFactorForIterations(start_width_imag, overall_zoom_factor, frame_count - 1)

        timeline = DiveTimeline(projectFolderName=self.ctx.project_name, framerate=self.fps, frameWidth=self.ctx.img_width, frameHeight=self.ctx.img_height, mathSupport=self.ctx.math_support)
        # Here, I have ctx, which should know escapeSquared, maxIter, and shouldSmooth... 
        #print("Trying to make span of %d frames" % frame_count)
        span = timeline.addNewSpanAtEnd(frame_count, self.ctx.cmplx_center, start_width_real, start_width_imag, end_width_real, end_width_imag, self.ctx.escape_squared, self.ctx.max_iter, self.ctx.smoothing)

        #perspectiveFrame = math.floor(frame_count * .5)
        #span.addNewTiltKeyframe(perspectiveFrame, 4.0, 1.0) 

        return timeline

    def __repr__(self):
        return """\
[MediaView duration {du:f} FPS:{f:s} Output:{vf:s}]\
""".format(du=self.duration,f=str(self.fps),vf=str(self.vfilename))


class DiveTimeline: 
    """
    Representation of an edit timeline. This maps parameters to specific frame numbers.
    A timeline also is the keeper of framerate for a frame sequence. (I think?)

    For now, the timeline also maintains the calculation cache for every frame, though
    this may eventually be the responsibility of a 'Project'

    Overview of the sequencing classes
    ----------------------------------
    DiveTimeline
    DiveTimelineSpan (Basis for setting keyframes)
    DiveSpanKeyframe
    - DiveSpanCenterKeyframe
    - DiveSpanWindowKeyframe
    - DiveSpanUniformKeyframe
    - DiveSpanTiltKeyframe
    DiveMesh
    MeshGenerator
    - MeshGeneratorUniform
    - MeshGeneratorTilt
   
    There are currently only 3 'tracks' of keyframes (for complex 
    center, window base sizes, and perspective).  Keyframes 
    currently all live on integer frame numbers.
    """

    def __init__(self, projectFolderName, framerate, frameWidth, frameHeight, mathSupport):
        self.projectFolderName = projectFolderName
        self.cachePath = os.path.join("%s_cache" % projectFolderName)

        self.framerate = float(framerate)
        self.frameWidth = int(frameWidth)
        self.frameHeight = int(frameHeight)

        self.mathSupport = mathSupport

        # No definition made yet for edit gaps, so let's just enforce adjacency of ranges for now.
        self.timelineSpans = []

    def addNewSpanAtEnd(self, frameCount, center, startWidthReal, startWidthImag, endWidthReal, endWidthImag, escapeSquared, maxEscapeIterations, shouldSmooth):
        """
        Constructs a new span, and adds it to the end of the existing span list

        Also adds center keyframes (that is, keyframes at the end of the span, which
        set the values for the complex center of the image), and window width keyframes 
        to the start and end of the new span.

        Apparently also adding perspective keyframes too.
        """
        span = DiveTimelineSpan(self, frameCount, escapeSquared, maxEscapeIterations, shouldSmooth)
        span.addNewCenterKeyframe(0, center, 'quadratic-to', 'quadratic-to')
        span.addNewCenterKeyframe(frameCount - 1, center, 'quadratic-to', 'quadratic-to')
        span.addNewWindowKeyframe(0, startWidthReal, startWidthImag)
        span.addNewWindowKeyframe(frameCount - 1, endWidthReal, endWidthImag)

        # Setting uniform perspective this way kinda taped on as a solution, but not yet 
        # sure how to more gracefully set up perspective keyframes.
        span.addNewUniformKeyframe(0)
        span.addNewUniformKeyframe(frameCount-1)

        self.timelineSpans.append(span)

        return span

    def getSpanForFrameNumber(self, frameNumber):
        """
        Seems like I should cache or memoize this, to keep from searching for every frame,
        or at least binary search it, but I'm allergic to optimizing before profiling, 
        so 'slow' search it is for now.
        
        10 frames -> {0,9}
        3 frames -> {10,12}
        10 frames -> {13,22}
        """
        nextSpanFirstFrame = 0
        for currSpan in self.timelineSpans:
            nextSpanFirstFrame += currSpan.frameCount
            if frameNumber < nextSpanFirstFrame:
                currSpan.lastObservedStartFrame = nextSpanFirstFrame - currSpan.frameCount
                return currSpan

        return None # Went past the end without finding a valid span, so it's too high a frame number

    def loadResultsForFrameNumber(self, frame_number, build_cache=True, cache_path="cache", invalidate_cache=False):
        """Cache-aware data tuple loading or calculating"""

        cache_file_name = os.path.join(cache_path, u"%d.npy" % frame_number)
        cache_metadata_file_name = os.path.join(cache_path, u"%d.npy.meta" % frame_number)
        #print("cache file %s" % cache_file_name)

        if invalidate_cache == True and os.path.exists(cache_file_name) == True and os.path.isfile(cache_file_name):
            os.remove(cache_file_name)
            os.remove(cache_metadata_file_name)
               
        frame_data = np.zeros((1,1), dtype=np.uint8) 
        frame_metadata = {}
        if os.path.exists(cache_file_name) == False:
            #print("Calculating epoch results")
            (frame_data, frame_metadata) = self.calculateResultsForFrameNumber(frame_number)

            if build_cache == True:
                #print("Writing cache file")
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path)

                # Write 2 separate files out, the numpy array, and a metadata sidecar
                np.save(cache_file_name, frame_data) 
                with open(cache_metadata_file_name, 'wb') as metadataHandle:
                    pickle.dump(frame_metadata, metadataHandle)
        else:
            #print("Loading cache file")
            # Load both the numpy array, and the metadata sidecar
            frame_data = np.load(cache_file_name, allow_pickle=True)
            with open(cache_metadata_file_name, 'rb') as metadataReadHandle:
                frame_metadata = pickle.load(metadataReadHandle)
        return (frame_data, frame_metadata)

    def calculateResultsForFrameNumber(self, frameNumber):
        diveMesh = self.getMeshForFrame(frameNumber)
        mesh = diveMesh.generateMesh()

        show_row_progress = False

        pixel_values_2d = np.zeros((mesh.shape[0], mesh.shape[1]), dtype=np.uint8)
        for x in range(0, mesh.shape[0]):
            for y in range(0, mesh.shape[1]):
                pixel_values_2d[x,y] = self.mathSupport.mandelbrot(mesh[x,y], diveMesh.escapeSquared, diveMesh.maxEscapeIterations, diveMesh.shouldSmooth)
            if show_row_progress == True:
                print("%d-" % x, end="")
                sys.stdout.flush()

        frame_metadata = {'frame_number' : frameNumber, 
            'mesh_center': str(diveMesh.center), 
            'real_width' : str(diveMesh.realMeshGenerator.baseWidth),
            'imag_width' : str(diveMesh.imagMeshGenerator.baseWidth),
            'escape_squared' : str(diveMesh.escapeSquared),
            'max_escape_iterations' : str(diveMesh.maxEscapeIterations)}

        return (pixel_values_2d, frame_metadata)


        ####
        # Graveyard of failed attempts at further vectorizing this, maybe there's a clue in here
        # somewhere...
        ####

        # In search of efficient ways to apply the map, and getting stuck with various issues
        # like pickling, which are keeping me from using multiprocessing.Pool

# Seemed to go exponential run time for some bizarre reason
#            # Probably not necessary, but lining up the 2-element subarray
#            pixel_inputs_1d = pixel_values_2d.reshape((mesh.shape[0] * mesh.shape[1]))
#
#            # Pretty sure this is mistakenly doing an n! pass, or something just as ridiculous.
#            #print("shape of pixel_inputs_1d: %s" % str(pixel_inputs_1d.shape))
#            pixel_values_1d = np.array([self.mathSupport.mandelbrot(complex_value, diveMesh.escapeSquared, diveMesh.maxEscapeIterations, diveMesh.shouldSmooth) for complex_value in pixel_inputs_1d])
#    
#            pixel_values_2d = pixel_values_1d.reshape((mesh.shape[0], mesh.shape[1]))
#            #pixel_values_2d = np.squeeze(pixel_values_2d, axis=2) # Incantation to remove a sub-array level
#            #print("shape of pixel_values_2d: %s" % str(pixel_values_2d.shape))

        #nope
        #theFunction = np.vectorize(self.mandelbrot_flint)
        #pixel_values_1d = theFunction(pixel_inputs_1d)

        #pixel_inputs_1d = pixel_inputs.reshape(1,self.img_width * self.img_height)

        #pixel_values_1d = map(self.mandelbrot_flint, pixel_inputs_1d)
        #pixel_values_1d = np.array(list(map(self.mandelbrot_flint, pixel_inputs_1d)))
        #print("shape of pixel_values_1d: %s" % str(pixel_values_1d.shape))

        # Can't pikcle... hmm
        #mandelpool = multiprocessing.Pool(processes = 1)
        #pixel_values_1d = mandelpool.map(self.mandelbrot_flint, pixel_inputs_1d)
        #mandelpool.close()
        #mandelpool.join()

    def getMeshForFrame(self, frameNumber):
        """
        Calculate a discretized 2d plane of complex points based on spans and keyframes in this timeline.

        # Haven't revisited this logic since building the MeshGenerator objects...
        Order of operations:
         1.) base_complex_center
         2.) distortions on base_complex_center (probably never really want this, but hey, it makes sense here)
         3.) MeshGenerator distortions?
         4.) overall distortions on the calculated 2D mesh

         Procedurally calculate the discretized 2D plane of complex points, based on all
         the known keyframes and modifiers for a given frame number...

         # complexCenter, complexWidth, imaginaryWidth 
         
        """
        # First step is to figure out which span the target frame belongs to
        targetSpan = self.getSpanForFrameNumber(frameNumber)
        if not targetSpan:
            raise IndexError("Frame number '%d' is out of range for this timeline" % frameNumber)

        # Within this span, find the closest upstream and closest downstream keyframes
        # Pretty sure we require least 2 keyframes defined for every span (start and end), so 
        # this should work out.
        localFrameNumber = frameNumber - targetSpan.lastObservedStartFrame
        (previousCenterKeyframe, nextCenterKeyframe) = targetSpan.getKeyframesClosestToFrameNumber('center', localFrameNumber) 
        #print("centers %s -> %s" % (str(previousCenterKeyframe.center), str(nextCenterKeyframe.center)))

        meshCenterValue = targetSpan.interpolateCenterValueBetweenKeyframes(localFrameNumber, previousCenterKeyframe, nextCenterKeyframe)
        #print("  interpolatedCenter: %s" % meshCenterValue)

        (previousWindowKeyframe, nextWindowKeyframe) = targetSpan.getKeyframesClosestToFrameNumber('window', localFrameNumber) 
        #print("windows %s,%s -> %s,%s" % (str(previousWindowKeyframe.realWidth), str(previousWindowKeyframe.imagWidth), str(nextWindowKeyframe.realWidth), str(nextWindowKeyframe.imagWidth)))

        (baseWidthReal, baseWidthImag) = targetSpan.interpolateWindowValuesBetweenKeyframes(localFrameNumber, previousWindowKeyframe, nextWindowKeyframe)

        #print("  interpolatedWidths: %s, %s" % (str(baseWidthReal), str(baseWidthImag)))

        (previousPerspectiveKeyframe, nextPerspectiveKeyframe) = targetSpan.getKeyframesClosestToFrameNumber('perspective', localFrameNumber)

        # More complicated with perspective keyframes, right?
        # Which mesh generator do we use - Uniform or Tilt?

        # Might have to interpolate the (widthFactor,heightFactor) of a tilt keyframe...
        # If both keyframes are uniform, then we don't actually interpolate
        # Hacky isisntance, but whatcha gonna do?
        previousIsUniform = isinstance(previousPerspectiveKeyframe, DiveSpanUniformKeyframe)
        nextIsUniform = isinstance(nextPerspectiveKeyframe, DiveSpanUniformKeyframe)
        if previousIsUniform and nextIsUniform:
            # Might feel like "baseImagWidth" is a typo (because it's distributed vertically), but
            # it's the 'imaginary width', even though we use it as the vertical element in the final mesh
            realMeshGenerator = MeshGeneratorUniform(mathSupport=self.mathSupport, varyingAxis='width', valuesCenter=meshCenterValue.real, baseWidth=baseWidthReal)
            imagMeshGenerator = MeshGeneratorUniform(mathSupport=self.mathSupport, varyingAxis='height', valuesCenter=meshCenterValue.imag, baseWidth=baseWidthImag)
        else:
            (widthTiltFactor, heightTiltFactor) = self.interpolateTiltFactorsBetweenPerspectiveKeyframes(localFrameNumber, previousPerspectiveKeyframe, nextPerspectiveKeyframe)

            # Tilt factor is the multiplier applied to the range.
            realMeshGenerator = MeshGeneratorTilt(mathSupport=self.mathSupport, varyingAxis='width', valuesCenter=meshCenterValue.real, baseWidth=baseWidthReal, tiltFactor=widthTiltFactor)
            imagMeshGenerator = MeshGeneratorTilt(mathSupport=self.mathSupport, varyingAxis='height', valuesCenter=meshCenterValue.imag, baseWidth=baseWidthImag, tiltFactor=heightTiltFactor)
   
        # Passing lots into the dive mesh.  Notably, some info about the DiveTimelineSpan that was
        # responsible for creating this mesh.  Might want to store an actual reference to the object, but
        # doesn't seem needed yet?
        diveMesh = DiveMesh(self.frameWidth, self.frameHeight, meshCenterValue, realMeshGenerator, imagMeshGenerator, self.mathSupport, targetSpan.escapeSquared, targetSpan.maxEscapeIterations, targetSpan.shouldSmooth)
        #print (diveMesh)
        return diveMesh

    def interpolateTiltFactorsBetweenPerspectiveKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        First part of this repeats some hacky isinstance stuff.

        But this time, we know they BOTH won't be uniform, else we never would try to
        interpolate between them.
        """
        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        # Hacky isisntance, but whatcha gonna do?
        leftIsUniform = isinstance(leftKeyframe, DiveSpanUniformKeyframe)
        rightIsUniform = isinstance(rightKeyframe, DiveSpanUniformKeyframe)

        leftFrameNumber = leftKeyframe.lastObservedFrameNumber
        rightFrameNumber = rightKeyframe.lastObservedFrameNumber

        # Use 1.0 as default widthFactor and heightFactor for uniform keyframes.
        if leftIsUniform:
            transitionType = rightKeyframe.transitionIn
            leftWidthValue = 1.0
            leftHeightValue = 1.0
            rightWidthValue = rightKeyframe.widthFactor
            rightHeightValue = rightKeyframe.heightFactor 
        elif rightIsUniform:
            transitionType = leftKeyframe.transitionOut
            leftWidthValue = leftKeyframe.widthFactor
            leftHeightValue = leftKeyframe.heightFactor
            rightWidthValue = 1.0
            rightHeightValue = 1.0
        else: # both are not uniform
            # TODO: probably should enforce same transition type or crash here, but
            # I don't feel like it at the moment.
            transitionType = leftKeyframe.transitionOut
            leftWidthValue = leftKeyframe.widthFactor
            leftHeightValue = leftKeyframe.heightFactor
            rightWidthValue = rightKeyframe.widthFactor
            rightHeightValue = rightKeyframe.heightFactor

        if transitionType == 'log-to':
            widthTiltFactor = self.mathSupport.interpolateLogTo(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateLogTo(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'root-to':
            widthTiltFactor = self.mathSupport.interpolateRootTo(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateRootTo(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'linear':
            widthTiltFactor = self.mathSupport.interpolateLinearTo(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateLinearTo(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'quadratic-to':
            widthTiltFactor = self.mathSupport.interpolateQuadraticEaseOut(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateQuadraticEaseOut(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'quadratic-from':
            widthTiltFactor = self.mathSupport.interpolateQuadraticEaseIn(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateQuadraticEaseIn(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        else: # transitionType == 'quadratic-to-from'
            widthTiltFactor = self.mathSupport.interpolateQuadraticEaseInOut(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateQuadraticEaseInOut(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)

        return(widthTiltFactor, heightTiltFactor)

# Maybe these belong in timeline span?
#    def getFrameNumberForTimecode(self, timecode):
#        return math.floor(
#    def getTimecodeForFrameNumber(self, frame_number):

    def __repr__(self):
        return """\
[DiveTimeline Project:{proj} framerate:{f}]\
""".format(proj=self.title,f=self.framerate)

class DiveTimelineSpan:
    """
    # ?(Can be used to observe/calculate n-1 zoom factors.)?
    """
    def __init__(self, timeline, frameCount, escapeSquared, maxEscapeIterations, shouldSmooth):
        self.timeline = timeline
        self.frameCount = int(frameCount)

        self.escapeSquared = escapeSquared
        self.maxEscapeIterations = maxEscapeIterations
        self.shouldSmooth = shouldSmooth

        # Only a single 'track' for each keyframe type to begin with, represented
        # just as a keyframe lookup for each track.  Being able to stack multiples of
        # similar-typed keyframes will probably be helpful in the long run.
        self.centerKeyframes = {}
        self.windowKeyframes = {}
        self.perspectiveKeyframes = {}
        # Currently, not allowing keyframes to exist outside of the span, even though
        # that is often helpful for defining pleasing transitions.
        # Currently, also not allowing keyframes to exist at non-frame targets, which
        # might lead to some alignment frustrations, because sub-frame calculations
        # are probably kinda important.

        self.lastObservedStartFrame = 0 # To stash frame offset when extracted from Timeline

    ####
    # TODO: All of these helper functions need to perform a modification of upstream and downsatream
    # keyframes when forcing addition, to make sure the transition types are all in order.
    # When creating a new keyframe, default is to inherit the lead-in and lead-out interpolators of the existing span.
    #
    # So, if it's:
    #  |           (lin)             |  (default all linear)
    #
    #  |  +(unspec)K(unspec)         |
    #  |    (lin)  K (lin)           | 
    #
    #  |  +(log-to)K(unspec)         |
    #  | (log-to)  K    (lin)        |
    #
    #  |  +(log-to)K(log-from)       |
    #  | (log-to)  K    (log-from)   |
    #
    # When trying to fill the range, both adjacent keyframes will agree (because it's a span property).
    # I think this means when setting a keyframe, the upstream and downstream keyframes are ALWAYS SET
    # to be consistent with the newly inserted keyframe. (either that, or create single-point-of-reference span structure?)
    #
    # BaseRangeSpan(startKey, endKey, type=lin)
    # +Keyframe(log-from) =>
    # BaseRangeSpan(startKey, keyframe, type=lin) + BaseRangeSpan(keyframe, endkey, type=log-from)
    #
    # Now, some asshole wants to define a 'speed' for a section...
    # This will drag some keyframes along, up to a point, where it can.
    # It might even just warn if reverse is observed?
    #  
    # Looks like we need special handling for when setting a keyframe on top of an existig keyframe, right?
    # If existing keyframe at the new keyframe frane number, and new keyframe is unspecified, then
    # keep the interpolators as-is.

    def addNewCenterKeyframe(self, frameNumber, centerValue, transitionIn='quadratic-to', transitionOut='quadratic-to'):
        # TODO: should probably gracefully handle stomping an existing keyframe, right?
        newKeyframe = DiveSpanCenterKeyframe(self, centerValue, transitionIn, transitionOut)
        self.centerKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def addNewWindowKeyframe(self, frameNumber, realWidth, imagWidth, transitionIn='root-to', transitionOut='root-to'):
        newKeyframe = DiveSpanWindowKeyframe(self, realWidth, imagWidth, transitionIn, transitionOut)
        self.windowKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def addNewUniformKeyframe(self, frameNumber, transitionIn='quadratic-to', transitionOut='quadratic-to'):
        newKeyframe = DiveSpanUniformKeyframe(self, transitionIn, transitionOut)
        self.perspectiveKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def addNewTiltKeyframe(self, frameNumber, widthFactor, heightFactor, transitionIn='quadratic-to-from', transitionOut='quadratic-to-from'):
        newKeyframe = DiveSpanTiltKeyframe(self, widthFactor, heightFactor, transitionIn, transitionOut)
        self.perspectiveKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def getKeyframesClosestToFrameNumber(self, keyframeType, frameNumber):
        """
        Returns a tuple of keyframes, which are the nearest left and right keyframes for the frameNumber.
        When the frameNumber directly has a keyframe, the same keyframe is returned for both values.

        Important to remember to bless the 'lastObservedFrameNumber' into the keyframe.
        """
        typeOptions = ['center', 'window', 'perspective']
        if keyframeType not in typeOptions:
            raise ValueError("keyframeType must be one of (%s)" % ", ".join(typeOptions))

        if frameNumber >= self.frameCount:
            raise IndexError("Requested %s keyframe frame number '%d' is out of range for a span that's '%d' frames long" % (keyframeType, frameNumber, self.frameCount))

        keyframeHash = None
        if keyframeType == 'perspective':
            keyframeHash = self.perspectiveKeyframes
        elif keyframeType == 'window':
            keyframeHash = self.windowKeyframes
        else: # keyframeType == 'center':
            keyframeHash = self.centerKeyframes

        # Direct hit 
        # (really should have non-integer locations for keyframes, shouldn't we?)
        if frameNumber in keyframeHash:
            targetKeyframe = keyframeHash[frameNumber]
            targetKeyframe.lastObservedFrameNumber = frameNumber
            return (targetKeyframe, targetKeyframe)

        previousKeyframe = None
        nextKeyframe = None
        for currFrameNumber in sorted(keyframeHash.keys()):
            if currFrameNumber <= frameNumber:
                previousKeyframe = keyframeHash[currFrameNumber]
                previousKeyframe.lastObservedFrameNumber = currFrameNumber
            if currFrameNumber > frameNumber:
                nextKeyframe = keyframeHash[currFrameNumber]
                nextKeyframe.lastObservedFrameNumber = currFrameNumber
                break # Past the sorted range, so done looking

        return (previousKeyframe, nextKeyframe)

    def interpolateCenterValueBetweenKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        Relies heavily on the stashed/cached 'lastObservedFrameNumber' value of a keyframe
        """
        #print("interpolating %s -> %s at frame %s" % (str(leftKeyframe), str(rightKeyframe), str(frameNumber)))

        # Recognize when left and right are the same, and dont' calculate anything.
        if leftKeyframe == rightKeyframe:
            return leftKeyframe.center

        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        # May want to consider 'close' rather than equal for the center equivalence check.
        if leftKeyframe.center == rightKeyframe.center:
            return leftKeyframe.center

        # Enforce that left keyframe's transitionOut should match right keyframe's transitionIn
        if leftKeyframe.transitionOut != rightKeyframe.transitionIn:
            raise ValueError("Keyframe transition types mismatched for frame number '%d'" % frameNumber)
        transitionType = leftKeyframe.transitionOut

        # Python scopes seep like this, right? Just use the value later?
        #
        # And I kept all these separate, because I'm prety sure there will be more
        # interpolation-specific parameters needed when all's said and done.
        if transitionType == 'log-to':
            interpolatedReal = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'root-to':
            interpolatedReal = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'linear':
            interpolatedReal = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'quadratic-to':
            interpolatedReal = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'quadratic-from':
            interpolatedReal = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        else: # transitionType == 'quadratic-to-from':
            interpolatedReal = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
            
        interpolatedCenter = self.timeline.mathSupport.createComplex(interpolatedReal, interpolatedImag)
        return interpolatedCenter

    def interpolateWindowValuesBetweenKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        Relies heavily on the stashed/cached 'lastObservedFrameNumber' value of a keyframe
        """
        # Recognize when left and right are the same, and dont' calculate anything.
        if leftKeyframe == rightKeyframe:
            return (leftKeyframe.realWidth, leftKeyframe.imagWidth)

        # Enforce that left keyframe's transitionOut should match right keyframe's transitionIn
        if leftKeyframe.transitionOut != rightKeyframe.transitionIn:
            raise ValueError("Keyframe transition types mismatched for frame number '%d'" % frameNumber)

        transitionType = leftKeyframe.transitionOut
        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        interpolatedRealWidth = leftKeyframe.realWidth
        interpolatedImagWidth = leftKeyframe.imagWidth

        # And I kept all these separate, because I'm prety sure there will be more
        # interpolation-specific parameters needed when all's said and done.
        if transitionType == 'log-to':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'root-to':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'linear':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'quadratic-to':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'quadratic-from':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        else: # transitionType == 'quadratic-to-from':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)

        return (interpolatedRealWidth, interpolatedImagWidth)

    def __repr__(self):
        return """\
[DiveTimelineSpan {framecount} frames]\
""".format(framecount=self.frameCount)

class DiveSpanKeyframe:
    def __init__(self, span, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        self.span = span 

        transitionOptions = ['quadratic-to', 'quadratic-from', 'quadratic-to-from', 'log-to', 'root-to', 'linear']
        if transitionIn not in transitionOptions:
            raise ValueError("transitionIn must be one of (%s)" % ", ".join(transitionOptions))
        if transitionOut not in transitionOptions:
            raise ValueError("transitionOut must be one of (%s)" % ", ".join(transitionOptions))
        
        self.transitionIn = transitionIn
        self.transitionOut = transitionOut

        self.lastObservedFrameNumber = 0 # For stashing frame numbers in

    def __repr__(self):
        return """\
[DiveSpanKeyframe, {inType} -> frame {frame} -> {outType}]\
""".format(inType=self.transitionIn, frame=self.lastObservedFrameNumber, outType=self.transitionOut)

class DiveSpanCenterKeyframe(DiveSpanKeyframe):
    def __init__(self, span, center, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        super().__init__(span, transitionIn, transitionOut)
        self.center = center

class DiveSpanWindowKeyframe(DiveSpanKeyframe):
    def __init__(self, span, realWidth, imagWidth, transitionIn='root-to', transitionOut='root-to'):
        super().__init__(span, transitionIn, transitionOut)
        self.realWidth = realWidth
        self.imagWidth = imagWidth

class DiveSpanUniformKeyframe(DiveSpanKeyframe):
    def __init__(self, span, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        super().__init__(span, transitionIn, transitionOut)

class DiveSpanTiltKeyframe(DiveSpanKeyframe):
    def __init__(self, span, widthFactor, heightFactor, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        super().__init__(span, transitionIn, transitionOut)
        self.widthFactor = widthFactor
        self.heightFactor = heightFactor

####
# Big bunch of 'still thinking about' comments here.  Probably should have kept them on my own branch.
###
#
# Conceptual priority of items in a timeline goes something like...
#
# Frame Numbers  
#  Frames are the base grid, so nothing can 'shift' a frame number to a higher or lower value.
#
# DiveSpan
#   Spans define the start and end base values (pre-modifications) of a dive animation
#
# DiveSpanCenterKeyframe and DiveSpanWindowKeyframe
#   Within spans, target values to hit for a frame
#
# I think that's the end of the highest-order items.  Modifications up to this point have to
# all be complete and calculated before the next items can be calculated/analyzed.
#
# For window rages, whether set as keyframes, or interpolated between, you can observe
# the effective zoom factor between any two frames, based on the transition from frame's 
# values to the next

#
# Idea behind ZoomFactorKeyframe
#
#  Attaches to the timeline based on width ranges.  Still haven't solved several
#  aspects of this half-baked idea yet.
#  Main goal is to be able to set a speed factor for a range of frames.
#
#  If a keyframe sets an axis 
#  range for a specific frame number, then the two surrounding frame-spans up to the nearest adjacent
#  range keyframes need to be stretched or squished so that the product of all the zoom factors
#  between keyframes achieves the ranges set in the adjacent range keyframes.  Or more simply,
#  if you add a range keyframe, then the surrounding zoom factors are adjusted so existing 
#  range keyframe targets aren't changed.
#  The default ramp type for zoom factor interpolation is linear.  The logic behind this is that
#  if the factor is linearly monotonically increasing, then the feel is acceleration.  So linear
#  interpolation of zoom factors results in a ramp of speed between keyframes.  I imagine that more
#  explicit curve control for zoom factor interpolation will be desirable.
#
#  [[Linear interpolation does seem to make it harder to set a constant speed though, doesn't it?]]
#  [[Does it work to add a keyframe as "set a constant zoom factor between all these frames"?]]
#

##
# A - attempt at constructing a 'move this image to this frame' kind of behavior 
##
# Take the calculated range values at frame + 4, and shift them to the target frame.
# Other stuff has to follow along... and attempting to write this is showing me what's left dangling...
#
# Quick observation:
# We'll come at this edit by looking at an existing render, and wanting a modification.
# The existing render has specific base information at a frame we're observing.
# This construction should be resilient to upstream base information modifications
# (i.e. if something 'before' is changed, this edit should still work about the same way)

# And this really has to be range-dependent, despite my trying to dodge that over and over and over.
# I keep trying to say "the base is rigidly calculated"
# And keep realizing that speed manipulations for a visible range MUST be dependent on some
# flexible attachment of the effect to a range of base values.
 
# Maybe the deal is, that range manipulations are all calculated in advance, and in the specific order of
# their declaration?  Like, there's no complicated layering of range manipulations, only
# a procedural description of all the range manipulations, which is realized into the baked-in
# base information for every frame, before axes are generated or meshes are manipulated.

# Adding set values non-monotonically, will basically play segments in reverse, right?
# start = 1.0, end = .0001, 100 frames long
# +keyframe in place at frame 10 (so frames 1-10 are equivalent to the default base ranges)
# +keyframe at frame 15, back to 1.0 (so plays start->key->start->end)
# (also, frames 15-100 are equivalent to what used to happen in frames 1-100)
#
# So now, want to set a 'speed' for 15->20
# Or better, want to set the reverse speed to faster...
# But I shouldn't be allowed to say "play a frame range faster", should I?
# Because it's dependent on the interpolated values that it has to hit.
# But I need to be able to say for this UNINTERRUPTED frame span (could verify on application?!), 
# Set my zoom factors to BLAH.
# A speed manipulation is a localized effect, that will change the (?upstream and?) downstream interpolations.
# The probem is, we maybe need to figure out what those ranges are iteratively, before allowing 
# it to 'lock' into place?

#targetFrameNumber = 2
#sampleFromFrameNumber = targetFrameNumber + 4
#
# Looks like that when we're sampling the 'current' shape for a future frame number, 
# we need to force that number into a keyframe, if it's part of an existing manipulation?
#if mainSpan.frameNumberHasParametricEffects(sampleFromFrameNumber):
#if mainSpan.frameNumberHasBaseModifiers(sampleFromFrameNumber):
#    # Creates a 'current values' keyframe, unrolls parametric changes to get to concrete values
#    mainSpan.addKeyframeInPlace(sampleFromFrameNumber) 
#
#addBaseRangeKeyframeInPlace
#
#focusRangeWidth, focusRangeHeight = mainSpan.getRangeShapeForFrameNumber(sampleFromFrameNumber)

# When we set a window keyframe, we've changed the 'effective zoom factor' of all upstream and downstream frames.
# The range of the 'effective speed change' is bounded by the next closest keyframes.

## 
# B - attempt at constructing a "speed up these 4 frames" kinda behavior
##
#
#targetFrameNumber = 3
#targetFrameCount = 4
#startingRangeWidth, startingRangeHeight = mainSpan.getRangeShapeForFrameNumber(targetFrameNumber)
#endingRangeWidth, endingRangeHeight = mainSpan.getRangeShapeForFrameNumber(targetFrameNumber + targetFrameCount - 1)

# No speed assignments that span across existing base keyframes allowed.

# Seems like this is getting close.
# The thing is, for a given range (whether forward or backward), I can't manipulate the speed in a way that
# breaks the base keyframe endpoints.
#
# Maybe that's it...
# No 'speed' assignments possible, only equivalent 'base keyframe' assignments.
# So I can't say 'play from this base range to that base range faster,
#
# Really want speed to ramp as a default, and make it so sudden jumps in speed
# are more difficult to achieve by default.

# THAT'S RIGHT  - I forgot, that speed modifiers don't hang on frame numbers, but they're applied 
# on regions within the visible ranges...
# BUT THAT WON'T WORK as a timeline thing, because if we reverse, then we don't want the 
# speed modifier to have an effect for all the matching ranges.

# Maybe it's that at a given keyframe, BOTH range targets, and a zoom factor can be applied?

## Maybe "speed change" is a function that does a bunch of math for you?
#mainSpan.addRangeSpeedKeyframe(startFrameNumber, leadingTransitionFrames=2, trailingTransitionFrames=2, startingZoomFactor * 2, 
#
#mainSpan.addRangeSpeedModifier(startFrameNumber, leadingTransitionFrames=2, trailingTransitionFrames=2, startingZoomFactor * 2)
#
#internally, it says:
#firstFrameZoomFactor = mainSpan.getZoomFactorBetweenFrameNumbers(startFrameNumber, startFrameNumber + 1)
#?rangeKeyframeAxisGenerator()?
#...generate keyframes for target ranges, based on  this... somehow...

class DiveMesh:
    """
    A DiveMesh is a 2D array of complex numbers that serves as a basis for
    calculation, with the x-axis (across the width) based on real component 
    distribution, and y-axis (across the height) based on imaginary 
    component distribution.  Conceptually, this is the mapping of a portion 
    of complex space into a 2D "imaging"(?) plane.  
    
    Two separate 2D (real-valued) meshes are generated first, one for the real component, 
    and one for the imaginary component.  This means range and distribution types of the 
    component meshes are the highest priority parameters.

    # Not Implemented:
    # Next,  distortions are applied separately to the real mesh, and to the imaginary mesh.
    # Then, the separate 2D meshes are combined to become the overall mesh values.
    # Finally, overall distortions are applied to the overall mesh values.
    """
    def __init__(self, width, height, center, realMeshGenerator, imagMeshGenerator, mathSupport, escapeSquared, maxEscapeIterations, shouldSmooth):
        # Trying not to apply castings to these types, to keep them the same as
        # the original parameters, which could make swapping out different 
        # precision libraries simpler?
        self.meshWidth = width
        self.meshHeight = height
        self.center = center

        self.realMeshGenerator = realMeshGenerator
        self.imagMeshGenerator = imagMeshGenerator

        #self.realMeshDistortions = []
        #self.imagMeshDistortions = []

        #self.meshDistortions = []

        # Technically, it seems redundant to have mathSupport specified separately for MeshGenerators
        # and for DiveMesh, but it's enough of a chicken-and-egg problem that I'll just
        # pass an extra parameter here and there.
        self.mathSupport = mathSupport

        # These properties are attached to the mesh, provided by the DiveTimelineSpan that created them
        self.escapeSquared = escapeSquared
        self.maxEscapeIterations = maxEscapeIterations
        self.shouldSmooth = shouldSmooth

    def generateMesh(self):
        realMesh = self.realMeshGenerator.generateForDiveMesh(self)
        imagMesh = self.imagMeshGenerator.generateForDiveMesh(self)

        if realMesh.shape != imagMesh.shape:
            raise ValueError("Real sub-mesh (%s) and Imaginary sub-mesh (%s) shapes don't match." % (realMesh.shape, imagMesh.shape))

        meshShape = realMesh.shape
        combinedMesh = np.zeros(meshShape, dtype=object) 
        # The native python 'complex' type assigns into "object" type arrays without problems,
        # but not vice-versa, so use object type for everything.

        for x in range(0, meshShape[0]):
            for y in range(0, meshShape[1]):
                combinedMesh[x,y] = self.mathSupport.createComplex(realMesh[x,y], imagMesh[x,y])

        return combinedMesh

    def __repr__(self):
        return """\
[DiveMesh {{{mwidth},{mheight}}} realGenerator:{rgen} imagGenerator:{igen} ]\
""".format(mwidth=self.meshWidth, mheight=self.meshHeight, rgen=str(self.realMeshGenerator), igen=str(self.imagMeshGenerator))

class MeshGenerator:
    def __init__(self, mathSupport, varyingAxis):
        self.mathSupport = mathSupport

        axisOptions = ['width', 'height']
        if varyingAxis not in axisOptions:
            raise ValueError("varyingAxis must be one of (%s)" % ", ".join(axisOptions))
        self.varyingAxis = varyingAxis

    def generateForDiveMesh(self):
        raise NotImplementedError("generateForDiveMesh() must be overridden in a MeshGenerator subclass")

class MeshGeneratorUniform(MeshGenerator):
    """
    Generates a 2D mesh of values distributed across with width, along the axis specified.

    DiveMesh is used for calculations of ranges, so the DiveMesh is responsible for 
    instantiation of correct types.  In other words, we try to avoid doing instantiations here.

    Technically, should probably make the DiveMesh responsible for valuesCenter, so there's a single
    point of reference.  It just seemed less clear that way at the moment.
    """
    def __init__(self, mathSupport, varyingAxis, valuesCenter, baseWidth):
        super().__init__(mathSupport, varyingAxis)
        self.valuesCenter = valuesCenter
        self.baseWidth = baseWidth

    def generateForDiveMesh(self, diveMesh):
        """
        e.g.
        diveMesh.meshWidth=3, diveMesh.meshHeight=2, self.varyingAxis='width', self.valuesCenter=1.0, baseWidth=.2 returns:
        [[0.9, 1.0, 1.1],
         [0.9, 1.0, 1.1]]
    
        diveMesh.meshWidth=3, diveMesh.meshHeight=2, self.varyingAxis='height', self.valuesCenter=1.0, baseWidth=.2 returns:
        [[0.9, 0.9, 0.9],
         [1.1, 1.1, 1.1]]
        """
        mesh = np.zeros((diveMesh.meshHeight, diveMesh.meshWidth), dtype=object)

        if self.varyingAxis == 'width':
            #calculate start/end...  Probably need to be subtype aware for this...
            discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, self.baseWidth, diveMesh.meshWidth)
            mesh[0:] = discretizedValues # Assign the one-row discretization to every row of the mesh
        else: # self.varyingAxis == 'height'
            discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, self.baseWidth, diveMesh.meshHeight)
            # Assign the one-row discretization (as a column) to every column of the mesh
            mesh[0:] = discretizedValues[:,np.newaxis] 

        return mesh

    def __repr__(self):
        return """\
[MeshGeneratorUniform center:{vCenter} baseWidth:{vWidth} along axis:{vAxis}]\
""".format(vCenter=self.valuesCenter, vWidth=self.baseWidth, vAxis=self.varyingAxis)

# Not implemented yet: 
# MeshGeneratorLogTilt, which uses a log scaling anchored at the middle
# Maybe also:
# DiveMeshGeneratorSqueeze (base-width->different-middle-width->base-width)
# Mostly, looking for effects that give me control over something that feels
# like camera behavior, such as lens barrel distortion.

class MeshGeneratorTilt(MeshGenerator):
    def __init__(self, mathSupport, varyingAxis, valuesCenter, baseWidth, tiltFactor):
        """
        Tilt is symmetric about the baseWidth.
        Negative values aren't treated as negative factors, but instead as reversal
        of the scaling direction (e.g. -2 means {.5,2.0}, and 2 => {2.0,.5})
        This means tilt values should not be between {-1.0,1.0}.

        Originally thought this would need 2 axis parameters, but for now, requiring
        the tilt axis to be the same as varying axis seems to make sense.

        Because the tilt factor is spread across the mesh rows, there might be
        some strong aliasing (across dive frames) if there's an even number 
        of rows or columns? I could imagine a back-and-forth wiggle developing if the
        scales and factors line up the right way.

        Axis discretization happens for every frame, on 2 axes (across complex range, and across real range).
        Mesh generation is the combination of these 2 axes (perhaps plus further post-processing modifications).

        A stretched axis (wider range), compared to the current frame's baseline, is akin to 
        calculating previous steps.  For example, if you happen to stretch the axis as much as the previous 
        frame transition's zoom factor, then you're sorta recalculating the previous frame's axis again.
        Similarly, a squished axis (narrower range), is akin to calculating future steps.
        """
        super().__init__(mathSupport, varyingAxis)
        self.valuesCenter = valuesCenter
        self.baseWidth = baseWidth
        self.tiltFactor = tiltFactor

    def generateForDiveMesh(self, diveMesh):
        mesh = np.zeros((diveMesh.meshHeight, diveMesh.meshWidth), dtype=object)

        # Calculating only one side from the tiltFactor, then using the delta from
        # the original as the other side's size.  This keeps our center in the linear
        # center of the ranges, instead of shifting it some amount dependent on the factor.

        # TODO: pretty sure this star/tend calculation needs to have its math done by 
        # the MathSupport too, to keep type requirements localized there?
        startWidth = 1.0 / self.tiltFactor * self.baseWidth
        startDelta = self.baseWidth - startWidth
        endWidth = self.baseWidth + startDelta
        if self.tiltFactor < 0.0:
            endWidth = 1.0 / self.tiltFactor * -1 * self.baseWidth
            endDelta = self.baseWidth - endWidth
            startWidth = self.baseWidth + endDelta

        if self.varyingAxis == 'width':
            # Values vary along the width axis, and the tiltFactor is applied to the
            # range of each row, which effectively treats the width axis as the 
            # rotation point
            meshRowBaseWidths = self.mathSupport.createLinspace(startWidth, endWidth, diveMesh.meshHeight)
            for y in range(0, diveMesh.meshHeight):
                #calculate start/end...  Probably need to be subtype aware for this...
                discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, meshRowBaseWidths[y], diveMesh.meshWidth)
                mesh[y] = discretizedValues # Assign the dscretization to this row
        else: # self.varyingAxis == 'height'
            meshColBaseWidths = self.mathSupport.createLinspace(startWidth, endWidth, diveMesh.meshWidth)
            for x in range(0, diveMesh.meshWidth):
                discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, meshColBaseWidths[x], diveMesh.meshHeight)
                mesh[:,x] = discretizedValues # Assign the discretization (as a column)

        return mesh

    def __repr__(self):
        return """\
[MeshGeneratorTilt center:{vCenter} baseWidth:{vWidth} tiltFactor:{tilt} along axis:'{vAxis}']\
""".format(vCenter=self.valuesCenter, vWidth=self.baseWidth, tilt=self.tiltFactor, vAxis=self.varyingAxis)


# --
# Default settings for the dive. All of these can be overridden from the
# command line
# --
def set_demo1_params(mandl_ctx, view_ctx):
    print("+ Running in demo mode - loading default params")
    mandl_ctx.img_width  = 1024
    mandl_ctx.img_height = 768 

    cmplx_width_str = '5.0'
    cmplx_height_str = '3.5'
    mandl_ctx.cmplx_width  = mandl_ctx.math_support.createFloat(cmplx_width_str)
    mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(cmplx_height_str)

    # This is close t Misiurewicz point M32,2
    # mandl_ctx.cmplx_center = mandl_ctx.ctxc(-.77568377, .13646737)
    center_real_str = '-1.769383179195515018213'
    center_imag_str = '0.00423684791873677221'
    mandl_ctx.cmplx_center = mandl_ctx.math_support.createComplex(center_real_str, center_imag_str)

    mandl_ctx.project_name = 'demo1'
    mandl_ctx.cache_path = u"%s_cache" % mandl_ctx.project_name

    mandl_ctx.scaling_factor = .90

    mandl_ctx.max_iter       = 255
    mandl_ctx.escape_rad     = 4.
    mandl_ctx.escape_squared = mandl_ctx.escape_rad * mandl_ctx.escape_rad

    mandl_ctx.verbose = 3
    mandl_ctx.burn_in = True
    mandl_ctx.build_cache=True

    view_ctx.duration       = 2.0 
    #view_ctx.duration       = 10.0 

    # FPS still isn't set quite right, but we'll get it there eventually.
    view_ctx.fps            = 23.976 / 2.0 
    #view_ctx.fps            = 29.97 / 2.0 


def set_preview_mode(mandl_ctx, view_ctx):
    print("+ Running in preview mode ")

    mandl_ctx.img_width  = 300
    mandl_ctx.img_height = 200

    mandl_ctx.cmplx_width  = mandl_ctx.math_support.createFloat(3.0)
    mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(2.5)

    mandl_ctx.scaling_factor = .75
    mandl_ctx.escape_rad     = 4.
    mandl_ctx.escape_squared = mandl_ctx.escape_rad * mandl_ctx.escape_rad

    view_ctx.duration       = 4
    view_ctx.fps            = 4


def set_snapshot_mode(mandl_ctx, view_ctx, snapshot_filename='snapshot.gif'):
    print("+ Running in snapshot mode ")

    mandl_ctx.snapshot = True
    view_ctx.vfilename = snapshot_filename

    mandl_ctx.img_width  = 3000
    mandl_ctx.img_height = 2000 

    mandl_ctx.max_iter   = 2000

    mandl_ctx.cmplx_width  = mandl_ctx.math_support.createFloat(3.0)
    mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(2.5)

    mandl_ctx.scaling_factor = .99 # set so we can zoom in more accurately
    mandl_ctx.escape_rad     = 4.
    mandl_ctx.escape_squared = mandl_ctx.escape_rad * mandl_ctx.escape_rad

    view_ctx.duration       = 0
    view_ctx.fps            = 0


def parse_options(mandl_ctx, view_ctx):
    argv = sys.argv[1:]
    
    opts, args = getopt.getopt(argv, "pd:m:s:f:z:w:h:c:",
                               ["preview",
                                "demo",
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
                                "project-name=",
                                "build-cache",
                                "invalidate-cache",
                                "banner",
                                "smooth"])

    # Math support as to be handled first, so other parameter 
    # instantiations are properly typed
    for opt, arg in opts:
        if opt in ['--flint']:
            mandl_ctx.math_support = DiveMathSupportFlint()

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode(mandl_ctx, view_ctx)
        elif opt in ['-s', '--snapshot']:
            set_snapshot_mode(mandl_ctx, view_ctx, arg)
        elif opt in ['--demo']:
            set_demo1_params(mandl_ctx, view_ctx)

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
            mandl_ctx.cmplx_center= mandl_ctx.math_support.createComplex(arg)
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
        elif opt in ['--project-name']:
            mandl_ctx.project_name = arg
            mandl_ctx.cache_path = u"%s_cache" % mandl_ctx.project_name
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

    mandl_ctx = MandlContext()
    view_ctx  = MediaView(16, 16, mandl_ctx)

    parse_options(mandl_ctx, view_ctx)
   
    view_ctx.run()


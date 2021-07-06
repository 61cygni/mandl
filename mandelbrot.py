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

from collections import defaultdict

import numpy as  np
import mpmath as mp
import moviepy.editor as mpy

from moviepy.audio.tools.cuts import find_audio_period

from PIL import Image, ImageDraw

MANDL_VER "0.1"

class MandlContext:
    """
    The context for a single dive
    """

    def __init__(self, ctxf = None, ctxc = None):

        if not ctxf:
            self.ctxf = float
        else:
            self.ctxf = ctxf
        if not ctxc:
            self.ctxc = complex
        else:    
            self.ctxc = ctxc 

        self.img_width  = 0 # int : Wide of Image in pixels
        self.img_height = 0 # int

        self.cmplx_width = 0.0 # width of visualization in complex plane
        self.cmplx_height = 0.0

        # point we're going to dive into 
        self.cmplx_center = complex(0.0) # center of image in complex plane

        self.max_iter      = 0  # int max iterations before bailing
        self.escape_rad    = 0. # float radius mod Z hits before it "escapes" 

        self.scaling_factor = 0.0 #  float amount to zoom each epoch
        self.num_epochs     = 0   #  int, nuber of epochs into the dive

        self.precision = 17 # int decimal precision for calculations

        self.duration  = 0  # int  duration of clip in seconds
        self.fps = 0 # int  number of frames per second

        self.verbose = 0 # how much to print about progress

    def zoom_in(self, iterations=1):
        while iterations:
            self.cmplx_width  *= self.scaling_factor
            self.cmplx_height *= self.scaling_factor
            self.num_epochs += 1
            iterations -= 1


    def mandelbrot(self, c):
        z = self.ctxc(0)
        n = 0

        squared_escape = self.escape_rad * self.escape_rad

        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane)
        #
        # However, instead we just square both sides of the inequality to
        # avoid the sqrt
        while ((z.real*z.real)+(z.imag*z.imag)) <= squared_escape  and n < self.max_iter:
            z = z*z + c
            n += 1
        
        # The following code smooths out the colors so there aren't bands
        # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
        # z = z*z + c; n+=1 # a couple extra iterations helps
        # z = z*z + c; n+=1 # decrease the size of the error
        # mu = n + 1 - mp.log(mp.log(fabs(z), b=2))
        #return n + 1 - math.log(math.log2(abs(z)))

        return n 

    def next_epoch(self, t):
        """Called for each frame of the animation. Will calculate
        current view, and then zoom in"""
    
        # Use center point to determines the box in the complex plane
        # we need to calculatee
        re_start = self.ctxf(self.cmplx_center.real - (self.cmplx_width / 2.))
        re_end =   self.ctxf(self.cmplx_center.real + (self.cmplx_width / 2.))

        im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
        im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))

        # Used to create a histogram of the frequency of iteration
        # deppths retured by the mandelbrot calculation. Helpful for 
        # color selection since many iterations never come up so you
        # loose fidelity by not focusing on those heavily used
        hist = defaultdict(int) 
        values = {}

        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                # map from pixels to complex coordinates
                Re_x = self.ctxf(re_start) + (self.ctxf(x) / self.ctxf(self.img_width))  * \
                       self.ctxf(re_end - re_start)
                Im_y = self.ctxf(im_start) + (self.ctxf(y) / self.ctxf(self.img_height)) * \
                       self.ctxf(im_end - im_start)

                c = self.ctxc(Re_x, Im_y)

                m = self.mandelbrot(c)

                values[(x,y)] = m 
                if m < self.max_iter:
                    hist[math.floor(m)] += 1

        total = sum(hist.values())
        hues = []
        h = 0

        # calculate percent of total for each iteration
        for i in range(self.max_iter):
            if total :
                h += hist[i] / total
            hues.append(h)
        hues.append(h)

        im = Image.new('RGB', (self.img_height, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                m = values[(x,y)] 

                # The color depends on the number of iterations    
                #hue = 255 - int(255 * linear_interpolation(hues[floor(m)], hues[ceil(m)], m % 1))
                color = 255 - int(255 * hues[m]) 
                # Plot the point
                draw.point([x, y], (color, color, color))


        #print("Finished iteration RErange %f:%f (re width: %f)"%(RE_START, RE_END, RE_END - RE_START))
        #print("Finished iteration IMrange %f:%f (im height: %f)"%(IM_START, IM_END, IM_END - IM_START))

        # Zoom in by scaling factor
        self.zoom_in()
        
        return np.array(im)
        

    def __repr__(self):
        return """\
[MandlContext Img W:{w:d} Img H:{h:d} Cmplx W:{cw:.20f}
Cmplx H:{ch:.20f} Complx Center:{cc:s} Scaling:{s:f} Epochs:{e:d}]\
""".format(
        w=self.img_width,h=self.img_height,cw=self.cmplx_width,ch=self.cmplx_height,
        cc=str(self.cmplx_center),s=self.scaling_factor,e=self.num_epochs); 

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
        self.vfilename = None 

    def run(self):

        self.clip = mpy.VideoClip(self.make_frame, duration=self.duration)

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
            

# For now, use global context for a single dive per run

mandl_ctx = MandlContext()
view_ctx  = MediaView(16, 16, mandl_ctx)


# --
# Default settings for the dive. All of these can be overridden from the
# command line
# --
def set_default_params():
    global mandl_ctx

    mandl_ctx.img_width  = 600
    mandl_ctx.img_height = 400

    mandl_ctx.cmplx_width  = mandl_ctx.ctxf(3.)
    mandl_ctx.cmplx_height = mandl_ctx.ctxf(2.5)

    # This is close t Misiurewicz point M32,2
    mandl_ctx.cmplx_center = mandl_ctx.ctxc(-.77568377, .13646737)

    mandl_ctx.scaling_factor = .97
    mandl_ctx.num_epochs     = 0

    mandl_ctx.max_iter       = 255
    mandl_ctx.escape_rad     = 20

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

    view_ctx.duration       = 4
    view_ctx.fps            = 4


def parse_options():
    global mandl_ctx

    argv = sys.argv[1:]

    
    opts, args = getopt.getopt(argv, "pd:m:s:f:",
                               ["preview",
                                "duration=",
                                "max_iter=",
                                "scale_factor=",
                                "fps=",
                                "gif=",
                                "mpeg=",
                                "verbose="])

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode()

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration = int(arg) 
        elif opt in ['-m', '--max_iter']:
            mandl_ctx.max_iter = int(arg)
        elif opt in ['-s', '--scale_factor']:
            mandl_ctx.scale_factor = float(arg)
        elif opt in ['-f', '--fps']:
            view_ctx.sfps = float(arg)
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

if __name__ == "__main__":

    print("++ mandlebort.py version %s" % (MANDL_VER))
    
    set_default_params()
    parse_options()

    view_ctx.run()

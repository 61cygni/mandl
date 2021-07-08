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

from PIL import Image, ImageDraw, ImageFont

MANDL_VER = "0.1"

class MandlPalette:
    """
    Color gradient
    """

    # Color in RGB 
    def __init__(self, color_list = [(0,0,0),(255,255,255),(0,0,0),(255,255,0),(255,204,204),(204,204,255),(255,255,204),(255,255,255)]):
        self.gradient_size = 1024
        self.color_list = color_list
        self.palette = []

    def linear_interpolate(color1, color2, fraction):
        new_r = int(math.ceil((color2[0] - color1[0])*fraction) + color1[0])
        new_g = int(math.ceil((color2[1] - color1[1])*fraction) + color1[1])
        new_b = int(math.ceil((color2[2] - color1[2])*fraction) + color1[2])
        return (new_r, new_g, new_b)


    # Create 255 value gradient
    # Use the following trivial linear interpolation algorithm
    # (color2 - color1) * fraction + color1
    def create_gradient(self):
        if len(self.palette) != 0:
            print("Error creating gradient, palette already exists")
            sys.exit(0)
        
        section_size = int(float(self.gradient_size)/float(len(self.color_list)-1))
        for c in range(0, len(self.color_list) - 1): 
            for i in range(0, section_size+1): 
                fraction = float(i)/float(section_size)
                new_color = MandlPalette.linear_interpolate(self.color_list[c], self.color_list[c+1], fraction)
                self.palette.append(new_color)
        while len(self.palette) < self.gradient_size:
            c = self.palette[-1]
            self.palette.append(c)
        #assert len(self.palette) == self.gradient_size    

    def make_frame(self, t):    

        IMG_WIDTH=255
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

        # point we're going to dive into 
        self.cmplx_center = complex(0.0) # center of image in complex plane

        self.max_iter      = 0  # int max iterations before bailing
        self.escape_rad    = 0. # float radius mod Z hits before it "escapes" 

        self.scaling_factor = 0.0 #  float amount to zoom each epoch
        self.num_epochs     = 0   #  int, nuber of epochs into the dive

        self.smoothing      = False # bool turn on color smoothing

        self.precision = 17 # int decimal precision for calculations

        self.duration  = 0  # int  duration of clip in seconds
        self.fps = 0 # int  number of frames per second

        self.palette = None
        self.burn_in = False

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

    def next_epoch(self, t):
        """Called for each frame of the animation. Will calculate
        current view, and then zoom in"""
    
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

                if not self.palette:
                    c = 255 - int(255 * hues[math.floor(m)]) 
                    color=(c, c, c)
                else:
                    color = self.palette[1024 - int(1024 * hues[math.floor(m)])]

                # Plot the point
                draw.point([x, y], color) 


        #print("Finished iteration RErange %f:%f (re width: %f)"%(RE_START, RE_END, RE_END - RE_START))
        #print("Finished iteration IMrange %f:%f (im height: %f)"%(IM_START, IM_END, IM_END - IM_START))

        if self.burn_in == True:
            burn_in_text = u"%d re range %f %f im range %f %f center %f + %f i" %\
                (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center.real, self.cmplx_center.imag)

            burn_in_location = (10,10)
            burn_in_margin = 5 
            burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
            burn_in_size = burn_in_font.getsize(burn_in_text)
            draw.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
            draw.text(burn_in_location, burn_in_text, 'white', burn_in_font)

        # Zoom in by scaling factor
        self.zoom_in()

        if self.verbose > 0:
            print("Done]")
        
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
            

    def __repr__(self):
        return """\
[MediaView duration {du:f} FPS:{f:d} Output:{vf:s}]\
""".format(du=self.duration,f=self.fps,vf=self.vfilename)

# For now, use global context for a single dive per run

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

    mandl_ctx.cmplx_width  = mandl_ctx.ctxf(3.)
    mandl_ctx.cmplx_height = mandl_ctx.ctxf(2.5)

    # This is close t Misiurewicz point M32,2
    # mandl_ctx.cmplx_center = mandl_ctx.ctxc(-.77568377, .13646737)
    mandl_ctx.cmplx_center = mandl_ctx.ctxc(-1.769383179195515018213,0.00423684791873677221)

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
                                "verbose=",
                                "palette-test",
                                "color",
                                "burn",
                                "smooth="])

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode()

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration = float(arg) 
        elif opt in ['-m', '--max_iter']:
            mandl_ctx.max_iter = int(arg)
        elif opt in ['-s', '--scale_factor']:
            mandl_ctx.scale_factor = float(arg)
        elif opt in ['-f', '--fps']:
            view_ctx.sfps = float(arg)
        elif opt in ['--smooth']:
            mandl_ctx.smoothing = bool(arg)
        elif opt in ['--palette-test']:
            m = MandlPalette()
            m.create_gradient()
            m.display()
            sys.exit(0)
        elif opt in ['--color']:
            m = MandlPalette()
            m.create_gradient()
            mandl_ctx.palette = m
        elif opt in ['--burn']:
            mandl_ctx.burn_in = True
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

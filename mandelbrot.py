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

from scipy.stats import norm

from moviepy.audio.tools.cuts import find_audio_period

from PIL import Image, ImageDraw, ImageFont

# -- our files
import fractalcache as fc

MANDL_VER = "0.1"

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

        self.ver = MANDL_VER # used to version cash

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

        self.set_zoom_level = 0   # Zoom in prior to the dive

        self.smoothing      = False # bool turn on color smoothing
        self.snapshot       = False # Generate a single, high res shotb

        self.precision = 17 # int decimal precision for calculations

        self.duration  = 0  # int  duration of clip in seconds
        self.fps = 0 # int  number of frames per second

        self.julia_c      = None
        self.julia_orig   = None
        self.julia_walk_c = None

        self.julia_list   = None 

        self.palette = None
        self.burn_in = False

        self.cache_dir = None

        self.verbose = 0 # how much to print about progress

    def zoom_in(self, iterations=1):
        while iterations:
            self.cmplx_width  *= self.scaling_factor
            self.cmplx_height *= self.scaling_factor
            self.num_epochs += 1
            iterations -= 1

    # Use Bresenham's line drawing algo for a simple walk between two
    # complex points
    def julia_walk(self, t):

        # duration of a leg
        leg_d      = float(self.duration) / float(len(self.julia_list) - 1)
        # which leg are we walking?
        leg        = math.floor(float(t) / leg_d)
        # how far along are we on that leg?
        timeslice  = float(self.duration) / (float(self.duration) * float(self.fps))
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


    # Some interesting c values
    # c = complex(-0.8, 0.156)
    # c = complex(-0.4, 0.6)
    # c = complex(-0.7269, 0.1889)

    def julia(self, c, z0):
        z = z0
        n = 0
        while abs(z) <= 2 and n < self.max_iter:
            z = z*z + c
            n += 1

        if n == self.max_iter:
            return self.max_iter

        return n + 1 - math.log(math.log2(abs(z)))

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

        if n >= self.max_iter:
            return self.max_iter
        
        # The following code smooths out the colors so there aren't bands
        # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
        if self.smoothing:
            z = z*z + c; n+=1 # a couple extra iterations helps
            z = z*z + c; n+=1 # decrease the size of the error
            mu = n + 1 - math.log(math.log2(abs(z)))
            return mu 
        else:    
            return n 

    def calc_cur_frame(self, snapshot_filename = None):        

        # --
        # Calculate box in complex plane from center point
        # --

        re_start = self.ctxf(self.cmplx_center.real - (self.cmplx_width / 2.))
        re_end =   self.ctxf(self.cmplx_center.real + (self.cmplx_width / 2.))

        im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
        im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))

        if self.verbose > 0:
            print("MandlContext starting epoch %d re range %f %f im range %f %f center %f + %f i .... " %\
                  (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center.real, self.cmplx_center.imag),
                  end = " ")


        # Used to create a histogram of the frequency of iteration
        # depths retured by the mandelbrot calculation. Helpful for 
        # color selection since many iterations never come up so you
        # loose fidelity by not focusing on those heavily used

        hist = defaultdict(int) 
        values = {}

        if snapshot_filename:
            print("Generating image [", end="")

        # --
        # Iterate over every point in the complex plane (1:1 mapping per
        # pixel) and run the fractacl calculation. We save the output in
        # a 2x2 array, and also create the histogram of values
        # --

        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                # map from pixels to complex coordinates
                Re_x = self.ctxf(re_start) + (self.ctxf(x) / self.ctxf(self.img_width))  * \
                       self.ctxf(re_end - re_start)
                Im_y = self.ctxf(im_start) + (self.ctxf(y) / self.ctxf(self.img_height)) * \
                       self.ctxf(im_end - im_start)

                c = self.ctxc(Re_x, Im_y)


                if not self.julia_c:
                    m = self.mandelbrot(c)
                else:        
                    z0 = c
                    m = self.julia(self.julia_c, z0) 

                values[(x,y)] = m 
                if m < self.max_iter:
                    hist[math.floor(m)] += 1

            if snapshot_filename:
                print(".",end="")
                sys.stdout.flush()

        if snapshot_filename:
            print("]")

        return values, hist    

    def draw_image_PIL(self, values, hues):    

        im = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                m = values[(x,y)] 

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
            burn_in_text = u"%d re range %f %f im range %f %f center %f + %f i" %\
                (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center.real, self.cmplx_center.imag)

            burn_in_location = (10,10)
            burn_in_margin = 5 
            burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
            burn_in_size = burn_in_font.getsize(burn_in_text)
            draw.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
            draw.text(burn_in_location, burn_in_text, 'white', burn_in_font)

        return im    

    def next_epoch(self, t, snapshot_filename = None):
        """Called for each frame of the animation. Will calculate
        current view, and then zoom in"""
    

        values, hist = None, None
        if self.cache: 
            values, hist = self.cache.read_cache()

        if not values or not hist:    
            # call primary calculation function
            values, hist = self.calc_cur_frame(snapshot_filename)

            if self.cache:
                self.cache.write_cache(values, hist)


        #- 
        # From histogram normalize to percent-of-total. This is
        # effectively a probability distribution of escape values 
        #
        # Note that this is not effecitly a probability distribution for
        # a given escape value. We can use this to calculate the Shannon 

        total = sum(hist.values())
        hues = []
        h = 0

        for i in range(self.max_iter):
            if total :
                h += hist[i] / total
            hues.append(h)
        hues.append(h)


        # -- 
        # Create image for this frame
        # --
        
        im = self.draw_image_PIL(values, hues)

        # -- 
        # Do next step in animation
        # -- 

        if self.julia_list:
            self.julia_walk(t)
        else:    
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
Cmplx H:{ch:.20f} Complx Center:{cc:s} Scaling:{s:f} Smoothing:{sm:b} Epochs:{e:d} Max iter:{mx:d}]\
""".format(
        w=self.img_width,h=self.img_height,cw=self.cmplx_width,ch=self.cmplx_height,
        cc=str(self.cmplx_center),s=self.scaling_factor,e=self.num_epochs,mx=self.max_iter,sm=self.smoothing); 

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

        self.ctx.duration = duration
        self.ctx.fps      = fps


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


    # --
    # Do any setup needed prior to running the calculatiion loop 
    # --

    def setup(self):

        print(self)
        print(self.ctx)

        if self.ctx.cache:
            if self.ctx.julia_c or self.ctx.julia_list:
                print("** Caching doesn't currently support Julia sets")
                self.ctx.cache = None
            else:    
                self.ctx.cache.setup()



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

    mandl_ctx.cmplx_width  = mandl_ctx.ctxf(5.0)
    mandl_ctx.cmplx_height = mandl_ctx.ctxf(3.5)

    # This is close t Misiurewicz point M32,2
    # mandl_ctx.cmplx_center = mandl_ctx.ctxc(-.77568377, .13646737)
    mandl_ctx.cmplx_center = mandl_ctx.ctxc(-1.769383179195515018213,0.00423684791873677221)

    mandl_ctx.scaling_factor = .97
    mandl_ctx.num_epochs     = 0

    mandl_ctx.max_iter       = 255
    mandl_ctx.escape_rad     = 32768. 

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
    mandl_ctx.escape_rad     = 32768. 

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
    mandl_ctx.escape_rad     = 32768. 

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
                                "cmplx-w=",
                                "cmplx-h=",
                                "center=",
                                "scaling-factor=",
                                "snapshot=",
                                "zoom=",
                                "fps=",
                                "gif=",
                                "mpeg=",
                                "verbose=",
                                "julia=",
                                "julia-walk=",
                                "center=",
                                "palette-test=",
                                "color=",
                                "burn",
                                "banner",
                                "cache",
                                "smooth"])

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode()
        if opt in ['-s', '--snapshot']:
            set_snapshot_mode()

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration  = float(arg) 
            mandl_ctx.duration = float(arg) 
        elif opt in ['-m', '--max-iter']:
            mandl_ctx.max_iter = int(arg)
        elif opt in ['-w', '--img-w']:
            mandl_ctx.img_width = int(arg)
        elif opt in ['-h', '--img-h']:
            mandl_ctx.img_height = int(arg)
        elif opt in ['--cmplx-w']:
            mandl_ctx.cmplx_width = float(arg)
        elif opt in ['--cmplx-h']:
            mandl_ctx.cmplx_height = float(arg)
        elif opt in ['-c', '--center']:
            mandl_ctx.cmplx_center= complex(arg)
        elif opt in ['-h', '--img-h']:
            mandl_ctx.img_height = int(arg)
        elif opt in ['--scaling-factor']:
            mandl_ctx.scaling_factor = float(arg)
        elif opt in ['-z', '--zoom']:
            mandl_ctx.set_zoom_level = int(arg)
        elif opt in ['-f', '--fps']:
            view_ctx.fps  = int(arg)
            mandl_ctx.fps = int(arg)
        elif opt in ['--smooth']:
            mandl_ctx.smoothing = True 
        elif opt in ['--julia']:
            mandl_ctx.julia_c = complex(arg) 
        elif opt in ['--cache']:
            mandl_ctx.cache = fc.FractalCache(mandl_ctx) 
        elif opt in ['--julia-walk']:
            mandl_ctx.julia_list = eval(arg)  # expects a list of complex numbers
            if len(mandl_ctx.julia_list) <= 1:
                print("Error: List of complex numbers for Julia walk must be at least two points")
                sys.exit(0)
            mandl_ctx.julia_c    = mandl_ctx.julia_list[0]
        elif opt in ['--center']:
            mandl_ctx.cmplx_center = complex(arg) 
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


if __name__ == "__main__":

    print("++ mandlebort.py version %s" % (MANDL_VER))
    
    set_default_params()
    parse_options()

    view_ctx.setup()
    view_ctx.run()

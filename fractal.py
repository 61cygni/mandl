# --
# File: fractal.py
# 
# Driver file for playing around with fractals 
#
#
# --

import getopt
import sys
import math
import importlib

import numpy as  np
import mpmath as mp

import moviepy.editor as mpy
from PIL import Image, ImageDraw, ImageFont

# -- our files

import fractalcache   as fc
import fractalpalette as fp

FRACTAL_VER = "0.1"


class FractalContext:
    """
    Parameters used across frames for a given animation 
    """

    def __init__(self, ctxf = None, ctxc = None, mp = None):

        self.ver = FRACTAL_VER # used to version cash

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

        self.img_width  = 1024 # int : Wide of Image in pixels
        self.img_height = 768  # int

        self.cmplx_width  = self.ctxf(5.0) # width of visualization in complex plane
        self.cmplx_height = self.ctxf(3.5) 

        self.magnification = 1.0 # track how far we've zoomed in 

        # point we're going to dive into 
        self.cmplx_center = None # require algo to set this 

        # should be in the algos
        self.max_iter      = 255    # int max iterations before bailing
        self.escape_rad    = 32768. # float radius mod Z hits before it "escapes" 

        self.scaling_factor = .95 #  float amount to zoom each epoch
        self.num_epochs     = 0   #  int, nuber of epochs into the dive

        self.advance = 0   # Call advance for this many frames prior to rendering

        self.smoothing      = False # bool turn on color smoothing
        self.snapshot       = False # Generate a single, high res shotb

        self.precision = 17 # int decimal precision for calculations

        self.duration  = 8  # int  duration of clip in seconds
        self.fps       = 8 # int  number of frames per second

        self.algo = None

        self.palette = None
        self.burn_in = False

        self.cache   = None

        self.verbose = 0 # how much to print about progress


    def calc_cur_frame(self, snapshot_filename = None):        

        # --
        # Calculate box in complex plane from center point
        # --

        re_start = self.ctxf(self.cmplx_center.real - (self.cmplx_width / 2.))
        re_end =   self.ctxf(self.cmplx_center.real + (self.cmplx_width / 2.))

        im_start = self.ctxf(self.cmplx_center.imag - (self.cmplx_height / 2.))
        im_end   = self.ctxf(self.cmplx_center.imag + (self.cmplx_height / 2.))

        if self.verbose > 0:
            print("FractalContext starting epoch %d re range %f %f im range %f %f center %f + %f i .... " %\
                  (self.num_epochs, re_start, re_end, im_start, im_end, self.cmplx_center.real, self.cmplx_center.imag),
                  end = " ")


        # Used to create a histogram of the frequency of iteration
        # depths retured by the mandelbrot calculation. Helpful for 
        # color selection since many iterations never come up so you
        # loose fidelity by not focusing on those heavily used

        values = {}

        if snapshot_filename:
            print("Generating image [", end="")

        # --
        # Iterate over every point in the complex plane (1:1 mapping per
        # pixel) and run the fractal calculation. We save the output in
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


                # Call primary calculation function here
                m = self.algo.calc_pixel(c)

                values[(x,y)] = m 

            if snapshot_filename:
                print(".",end="")
                sys.stdout.flush()

        if snapshot_filename:
            print("]")

        return values

    def draw_image_PIL(self, t, values):    

        im = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                m = values[(x,y)] 

                color = self.algo.map_value_to_color(t,m)

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
        current view, and then animate next step"""
    
        values = None
        
        if self.cache: 
            values = self.cache.read_cache()
            if values:
                self.algo.cache_loaded(values)

        if not values:    
            # call primary calculation function
            values = self.calc_cur_frame(snapshot_filename)

            if self.cache:
                self.cache.write_cache(values)


        # -- 
        # Create image for this frame
        # --
        
        self.algo.pre_image_hook()
        im = self.draw_image_PIL(t, values)

        # -- 
        # Do next step in animation
        # -- 

        self.algo.animate_step(t)

        if self.verbose > 0:
            print("Done]")

        self.algo.per_frame_reset()
        
        if snapshot_filename:
            return im.save(snapshot_filename,"gif")
        else:    
            return np.array(im)
        

    def __repr__(self):
        return """\
[FractalContext Img W:{w:d} Img H:{h:d} Cmplx W:{cw:.20f}
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
    
        self.ctx.next_epoch(-1,self.vfilename)

    # --
    # Do any setup needed prior to running the calculatiion loop 
    # --

    def setup(self):

        if not self.ctx.palette:
            print("No palette specified, using default")
            self.ctx.palette = fp.FractalPalette(self.ctx)

        print(self)
        print(self.ctx)

        self.ctx.algo.setup()

        if self.ctx.cache:
            self.ctx.cache.setup()


    def run(self):

        # Check whether we need to advance the animation prior to
        # starting 

        if self.ctx.advance > 0:
            print("Advancing by %d epochs" % (self.ctx.advance))
            cur_t = 0.0
            while self.ctx.advance > 0:
                duration = self.ctx.duration
                fps      = self.ctx.fps

                self.ctx.algo.animate_step(cur_t)
                cur_t += 1. / (float(duration) + float(fps))
                self.ctx.advance -= 1
            

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

fractal_ctx = FractalContext()
view_ctx    = MediaView(16, 16, fractal_ctx)


def set_preview_mode():
    global fractal_ctx

    print("+ Running in preview mode ")

    fractal_ctx.img_width  = 300
    fractal_ctx.img_height = 200

    fractal_ctx.cmplx_width  = 3.
    fractal_ctx.cmplx_height = 2.5 

    fractal_ctx.scaling_factor = .75

    view_ctx.duration       = 4
    view_ctx.fps            = 4

def set_snapshot_mode():
    global fractal_ctx

    print("+ Running in snapshot mode ")

    fractal_ctx.snapshot  = True

    fractal_ctx.img_width  = 3840 
    fractal_ctx.img_height = 2160 

    fractal_ctx.max_iter   = 2000

    fractal_ctx.cmplx_width  = 3.
    fractal_ctx.cmplx_height = 2.5 

    fractal_ctx.scaling_factor = 1. 
    fractal_ctx.escape_rad     = 32768. 

    view_ctx.duration       = 0
    view_ctx.fps            = 0


def parse_options():
    global fractal_ctx

    argv = sys.argv[1:]

    
    opts, args = getopt.getopt(argv, "pd:m:s:f:w:h:c:a:",
                               ["preview",
                                "algo=",
                                "duration=",
                                "max-iter=",
                                "img-w=",
                                "img-h=",
                                "cmplx-w=",
                                "cmplx-h=",
                                "center=",
                                "scaling-factor=",
                                "snapshot=",
                                "advance=",
                                "fps=",
                                "gif=",
                                "mpeg=",
                                "verbose=",
                                "julia-c=",
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
            view_ctx.vfilename = arg
            set_snapshot_mode()
        if opt in ['-a', '--algo']:
            module = importlib.import_module(arg) 
            fractal_ctx.algo = module._instance(fractal_ctx)

    # default to mandelbrot if nothing else is specified
    if not fractal_ctx.algo:
        module = importlib.import_module("mandelbrot") 
        fractal_ctx.algo = module._instance(fractal_ctx)

    print("+ Using algo %s"%(str(fractal_ctx.algo)))
    
    fractal_ctx.algo.set_default_params()    

    if type(fractal_ctx.cmplx_center) == type(None):
        print("Error: algo must set center value")
        sys.exit(0)
        

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration  = float(arg) 
            fractal_ctx.duration = float(arg) 
        elif opt in ['-m', '--max-iter']:
            fractal_ctx.max_iter = int(arg)
        elif opt in ['-w', '--img-w']:
            fractal_ctx.img_width = int(arg)
        elif opt in ['-h', '--img-h']:
            fractal_ctx.img_height = int(arg)
        elif opt in ['--cmplx-w']:
            fractal_ctx.cmplx_width = float(arg)
        elif opt in ['--cmplx-h']:
            fractal_ctx.cmplx_height = float(arg)
        elif opt in ['-c', '--center']:
            fractal_ctx.cmplx_center= complex(arg)
        elif opt in ['-h', '--img-h']:
            fractal_ctx.img_height = int(arg)
        elif opt in ['--scaling-factor']:
            fractal_ctx.scaling_factor = float(arg)
        elif opt in ['--advance']:
            fractal_ctx.advance = int(arg)
        elif opt in ['-f', '--fps']:
            view_ctx.fps  = int(arg)
            fractal_ctx.fps = int(arg)
        elif opt in ['--smooth']:
            fractal_ctx.smoothing = True 
        elif opt in ['--cache']:
            fractal_ctx.cache = fc.FractalCache(fractal_ctx) 
        elif opt in ['--center']:
            fractal_ctx.cmplx_center = complex(arg) 
        elif opt in ['--palette-test']:
            m = fp.FractalPalette(fractal_ctx)
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
            m = fp.FractalPalette(fractal_ctx)
            if str(arg) == "gauss":
                m.create_gauss_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp":    
                m.create_exp_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp2":    
                m.create_exp2_gradient((0,0,0),(128,128,128))
            elif str(arg) == "list":    
                m.create_gradient_from_list()
            else:
                print("Error: --palette arg must be one of gauss|exp|list")
                sys.exit(0)
            fractal_ctx.palette = m
        elif opt in ['--burn']:
            fractal_ctx.burn_in = True
        elif opt in ['--banner']:
            view_ctx.banner = True
        elif opt in ['--verbose']:
            verbosity = int(arg)
            if verbosity not in [0,1,2,3]:
                print("Invalid verbosity level (%d) use range 0-3"%(verbosity))
                sys.exit(0)
            fractal_ctx.verbose = verbosity
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

        
    fractal_ctx.algo.parse_options(opts, args)

if __name__ == "__main__":

    print("++ fractal.py version %s" % (FRACTAL_VER))
    
    parse_options()

    view_ctx.setup()
    view_ctx.run()

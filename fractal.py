# --
# File: fractal.py
# 
# Driver file for playing around with fractals 
#
# Examples:
#  python3 fractal.py        # take a snapshot of the mandelbrot
#  python3 fractal.py --dive # dive into the mandelbrot 
#  python3 fractal.py --dive --keyframe=7 # 4k dive using keyframes 
#  python3 fractal.py --algo=julia # take a snapshot of a julia set
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

        self.img_width  = 0  # int : Width of Image in pixels
        self.img_height = 0  # int

        self.cmplx_width  = self.ctxf(0) # width of visualization in complex plane
        self.cmplx_height = self.ctxf(0) 

        self.magnification = 1.0 # track how far we've zoomed in 

        # point we're going to dive into 
        self.cmplx_center = None 

        # should be in the algos
        self.max_iter      = 0      # int max iterations before bailing
        self.escape_rad    = 0.     # float radius mod Z hits before it "escapes" 

        self.scaling_factor = .90 #  float amount to zoom each epoch
        self.num_epochs     = 0   #  int, nuber of epochs into the dive

        self.advance = 0   # Call advance for this many frames prior to rendering

        self.smoothing      = False # bool turn on color smoothing
        self.snapshot       = False # Generate a single, high res shotb

        self.precision = 17 # int decimal precision for calculations

        self.duration  = 8  # int  duration of clip in seconds
        self.fps       = 8 # int  number of frames per second

        self.algo      = None
        self.algo_name = None

        self.dive    = False # Dive into the fractal
        self.animate = False # run algorithm-specific animation

        self.keyframe = 0
        self.cur_keyframe = None 

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

        if self.keyframe or snapshot_filename:
            print("Generating frame [", end="")

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

            if self.keyframe or snapshot_filename:
                print(".",end="")
                sys.stdout.flush()

        if self.keyframe or snapshot_filename:
            print("]")

        return values

    def draw_image_PIL(self, t, values):    

        im = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):

                color = self.algo.map_value_to_color((x,y),values)

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

    def zoom_and_crop_keyframe(self):    
        crop_w = self.img_width  * self.scaling_factor
        crop_h = self.img_height * self.scaling_factor
        diff_w = self.img_width  - crop_w
        diff_h = self.img_height - crop_h
        crop_x = int(diff_w / 2.) 
        crop_y = int(diff_h / 2.)
        crop_img   = self.cur_keyframe.crop((crop_x,crop_y,crop_x + crop_w, crop_y+crop_h))
        resize_img = crop_img.resize((self.img_width, self.img_height), Image.LANCZOS)

        self.cmplx_width   *= self.scaling_factor
        self.cmplx_height  *= self.scaling_factor
        self.magnification *= self.scaling_factor
        self.num_epochs += 1

        if self.num_epochs % self.keyframe == 0:
            self.cur_keyframe = None
        else:
            self.cur_keyframe = resize_img

        return np.array(resize_img)

    def next_epoch(self, t, snapshot_filename = None):
        """Called for each frame of the animation. Will calculate
        current view, and then animate next step"""
    
        values = None

        if self.keyframe and self.cur_keyframe: 
            return self.zoom_and_crop_keyframe()
        
        if self.cache: 
            values = self.cache.read_cache()
            if values:
                self.algo.cache_loaded(values)

        if not values:    
            # primary calculation function
            values = self.calc_cur_frame(snapshot_filename)

            if self.cache:
                self.cache.write_cache(values)


        self.algo.pre_image_hook()
        im = self.draw_image_PIL(t, values)

        if self.keyframe and not self.cur_keyframe:
            self.cur_keyframe = im

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
        self.vfilename = "pyfractal.gif"

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

        if not self.ctx.dive and not self.ctx.animate:
            set_snapshot_mode()
        elif self.ctx.dive:    
            set_dive_mode()

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

    if not fractal_ctx.img_width:
        fractal_ctx.img_width  = 3840 
    if not fractal_ctx.img_height:     
        fractal_ctx.img_height = 2160 
    if not fractal_ctx.cmplx_width:
        fractal_ctx.cmplx_width  = 4.
    if not fractal_ctx.cmplx_height:
        fractal_ctx.cmplx_height = 3.5 

    if not fractal_ctx.max_iter:
        fractal_ctx.max_iter   = 2048
    if not fractal_ctx.escape_rad:    
        fractal_ctx.escape_rad = 32768. 

    if not fractal_ctx.cmplx_center :
        if fractal_ctx.algo_name == "julia":
            print(" * Warning no center specified, setting to 0+0j")
            fractal_ctx.cmplx_center = complex(0+0j)
        else:    
            print(" * Warning no center specified, setting to -1+0j")
            fractal_ctx.cmplx_center = complex(-1+0j)

    if view_ctx.duration or view_ctx.fps:
        print(" * Warning : duration and FPS not used in snapshot mode")

def set_dive_mode():
    global fractal_ctx

    print("+ Running in dive mode ")
    if fractal_ctx.keyframe:
        print("+ Generating keyframe every %d frames "%(fractal_ctx.keyframe))

    assert fractal_ctx.dive

    if not fractal_ctx.img_width:
        if not fractal_ctx.keyframe:
            fractal_ctx.img_width  = 1024 
        else:    
            fractal_ctx.img_width  = 3840 
    if not fractal_ctx.img_height:     
        if not fractal_ctx.keyframe:
            fractal_ctx.img_height = 768 
        else:    
            fractal_ctx.img_height = 2160 
    if not fractal_ctx.cmplx_width:
        fractal_ctx.cmplx_width  = 4.
    if not fractal_ctx.cmplx_height:
        fractal_ctx.cmplx_height = 3.5 

    if not fractal_ctx.max_iter:
        fractal_ctx.max_iter   = 512
    if not fractal_ctx.escape_rad:    
        fractal_ctx.escape_rad = 32768. 

    if not fractal_ctx.cmplx_center :
        print(" * Warning, no center specific, setting to -.749706+0.0314565j")
        fractal_ctx.cmplx_center = complex(-.749696000010025+0.031456625003j)

    if view_ctx.duration or view_ctx.fps:
        print("Warning : duration and FPS not used in snapshot mode")

def parse_options():
    global fractal_ctx

    argv = sys.argv[1:]

    
    opts, args = getopt.getopt(argv, "pd:m:f:w:h:c:a:",
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
                                "keyframe=",
                                "dive",
                                "burn",
                                "banner",
                                "cache",
                                "smooth"])

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode()
        if opt in ['-a', '--algo']:
            module = importlib.import_module(arg) 
            fractal_ctx.algo = module._instance(fractal_ctx)

    # default to mandelbrot if nothing else is specified
    if not fractal_ctx.algo:
        module = importlib.import_module("smooth") 
        fractal_ctx.algo = module._instance(fractal_ctx)

    # hacky way to pull name from object. Assumes name looks 
    # something like  <smooth.Smooth object at 0x108e92820>
    print("+ Using algo %s"%(str(fractal_ctx.algo)[1:].split('.')[0]))

    fractal_ctx.algo_name = str(fractal_ctx.algo)[1:].split('.')[0]
    
    fractal_ctx.algo.set_default_params()    

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
        elif opt in ['--keyframe']:
            fractal_ctx.keyframe = int(arg) 
        elif opt in ['--cache']:
            fractal_ctx.cache = fc.FractalCache(fractal_ctx) 
        elif opt in ['--center']:
            fractal_ctx.cmplx_center = complex(arg) 
        elif opt in ['--dive']:
            fractal_ctx.dive = True
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
            view_ctx.vfilename = arg
        elif opt in ['--mpeg']:
            view_ctx.vfilename = arg

        
    fractal_ctx.algo.parse_options(opts, args)

if __name__ == "__main__":

    print("++ fractal.py version %s" % (FRACTAL_VER))
    
    parse_options()

    view_ctx.setup()
    view_ctx.run()

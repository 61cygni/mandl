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
#  python3 fractal.py --algo=julia --julia-walk="[0.355+0.355j, 0+0.8j,0.355+0.355j]"
#  python3 fractal.py --algo=mandelbrot --color=list --img-w=600 --img-h=400 --dive
#
# --

import time
import getopt
import sys
import math
import importlib

import decimal

import numpy as  np

import moviepy.editor as mpy
from PIL import Image, ImageDraw, ImageFont

# -- our files

from globalconfig  import *
from sharedcontext import *

import fractalcache   as fc
import fractalpalette as fp


FRACTAL_VER = "0.1"

class FractalContext:
    """
    Parameters used across frames for a given animation 
    """

    def __init__(self, ctxf = None, ctxc = None, mp = None):

        self.ver = FRACTAL_VER # used to version cash

        self.img_width  = 0  # int : Width of Image in pixels
        self.img_height = 0  # int

        self.cmplx_width  = hpf(0) # width of visualization in complex plane
        self.cmplx_height = hpf(0) 

        self.magnification = 1.0 # track how far we've zoomed in 

        # center of image
        self.c_real = None
        self.c_Imag = None

        # should be in the algos
        self.max_iter      = 0      # int max iterations before bailing
        self.escape_rad    = 0.     # float radius mod Z hits before it "escapes" 

        self.samples       = 0.     # number of samples to take per pixel

        self.scaling_factor = .90 #  float amount to zoom each epoch
        self.num_epochs     = 0   #  int, nuber of epochs into the dive

        self.advance = 0   # Call advance for this many frames prior to rendering

        self.snapshot       = False # Generate a single, high res shotb
        self.exploere       = False # pop up a visual explorer  

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
        self.burn_in_front_size = 12

        self.sleep   = 0

        self.cache   = None

        self.verbose = 0 # how much to print about progress



    def calc_cur_frame(self, snapshot_filename = None):        

        # --
        # Calculate box in complex plane from center point
        # --

        re_start = hpf(self.c_real - (self.cmplx_width / hpf(2.)))
        re_end =   hpf(self.c_real + (self.cmplx_width / hpf(2.)))

        im_start = hpf(self.c_imag - (self.cmplx_height / hpf(2.)))
        im_end   = hpf(self.c_imag + (self.cmplx_height / hpf(2.)))

        if self.verbose > 0:
            print("FractalContext starting epoch %d re range %f %f im range %f %f center %s + %s i .... " %\
                  (self.num_epochs, re_start, re_end, im_start, im_end, str(self.c_real), str(self.c_imag)), end = " ")


        # Used to create a histogram of the frequency of iteration
        # depths retured by the mandelbrot calculation. Helpful for 
        # color selection since many iterations never come up so you
        # loose fidelity by not focusing on those heavily used

        values = {}

        if self.keyframe or snapshot_filename:
            self.pre_time = time.perf_counter()  

        values = self.algo.calc_cur_frame(self.img_width, self. img_height,re_start, re_end, im_start, im_end)         

            #if self.keyframe or snapshot_filename:
            #    print(".",end="")
            #    sys.stdout.flush()

        if self.keyframe or snapshot_filename:
            print("+ frame generation time : %f"%(time.perf_counter() - self.pre_time))
            sys.stdout.flush()

        return values


    def draw_image_PIL(self, values, snapshot_filename):    

        im   = None
        draw = None

        if type(values) == type({}): 
            im = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
            draw = ImageDraw.Draw(im)

            if self.keyframe or snapshot_filename:
                self.pre_time = time.perf_counter()  
                print("Calculating colors [", end="")
                sys.stdout.flush()
            
            for x in range(0, self.img_width):
                for y in range(0, self.img_height):
                    
                    color = (128,128,128)
                    if (x,y) in values: 
                        color = self.algo.map_value_to_color(values[(x,y)])

                    # Plot the point
                    draw.point([x, y], color) 
                print(".", end="")
                sys.stdout.flush()

            if self.keyframe or snapshot_filename:
                print("%f]"%(time.perf_counter() - self.pre_time))
                sys.stdout.flush()
        else: # assume it's an image
            im = values
            draw = ImageDraw.Draw(im)
            

        # set keyframe before we do a burnin ... otherwise we see the
        # old text when we zoom in

        if self.keyframe and self.num_epochs % self.keyframe == 0:
            self.cur_keyframe = None
        else:
            self.cur_keyframe = im.copy()


        self.handle_burn_in(draw)    

        #print("Finished iteration RErange %f:%f (re width: %f)"%(RE_START, RE_END, RE_END - RE_START))
        #print("Finished iteration IMrange %f:%f (im height: %f)"%(IM_START, IM_END, IM_END - IM_START))

        # Code to track the center of the image. Helps to debug zooming, find coordinates etc. leaving in for
        # future debugging help /mc
        #
        #x0 = int(self.img_width / 2.0) - 4
        #y0 = int(self.img_height / 2.0) - 4
        #x1 = x0 + 8
        #y1 = y0 + 8
        #draw.ellipse((x0,y0,x1,y1), fill=(255,0,0), width=3)

        return im    

    def handle_burn_in(self, draw):
        if not self.burn_in:
            return

        re_start = hpf(self.c_real - (self.cmplx_width / hpf(2.)))
        re_end =   hpf(self.c_real + (self.cmplx_width / hpf(2.)))

        im_start = hpf(self.c_imag - (self.cmplx_height / hpf(2.)))
        im_end   = hpf(self.c_imag + (self.cmplx_height / hpf(2.)))

        dive_str = ""
        if self.dive:
            dive_str="epoch: %d "%(self.num_epochs)

        burn_in_text_1 = u"%sre len %.20e im len %.20e" %\
            (dive_str, re_end - re_start, im_end - im_start)
        burn_in_text_2 = u"center %.20e + %.20e i" %\
            (self.c_real, self.c_imag)

        location = (10,10)
        margin = 5 

        burn_in_font = ImageFont.truetype('fonts/cour.ttf', self.burn_in_front_size)
        burn_in_size = burn_in_font.getsize(burn_in_text_1)

        # create rectangle for 2 lines
        rec_x1 = location[0] - margin
        rec_y1 = location[1] - margin
        rec_x2 = burn_in_size[0]     + margin * 2
        rec_y2 = burn_in_size[1]*2   + margin * 4

        draw.rectangle(((rec_x1, rec_y1), (rec_x2, rec_y2)), fill="black")

        # First Line
        draw.text(location, burn_in_text_1, 'white', burn_in_font)

        # second line 
        location_2 = (location[0], location[1] + burn_in_size[1] + margin*2)
        burn_in_font = ImageFont.truetype('fonts/cour.ttf', self.burn_in_front_size)
        burn_in_size = burn_in_font.getsize(burn_in_text_2)
        draw.text(location_2, burn_in_text_2, 'white', burn_in_font)

    def zoom_and_crop_keyframe(self):    
        center_x = self.img_width  / 2
        center_y = self.img_height / 2
        scaling2 = self.scaling_factor / 2.0

        img = self.cur_keyframe.crop((center_x - (self.img_width * scaling2), center_y - (self.img_height * scaling2),
                                      center_x + (self.img_width * scaling2), center_y + (self.img_height * scaling2)))
        resize_img = img.resize((self.img_width, self.img_height), resample = Image.LANCZOS)
        return resize_img 

    def next_epoch(self, t, snapshot_filename = None):
        """Called for each frame of the animation. Will calculate
        current view, and then animate next step"""
    
        values = None

        if self.keyframe and self.cur_keyframe: 
            values = self.zoom_and_crop_keyframe()
        
        if not values and self.cache: 
            values = self.cache.read_cache()
            if values:
                self.algo.cache_loaded(values)

        if not values: 
            # primary calculation function
            values = self.calc_cur_frame(snapshot_filename)

            if self.cache:
                self.cache.write_cache(values)


        self.algo.pre_image_hook()
        im = self.draw_image_PIL(values, snapshot_filename)


        # -- 
        # Do next step in animation
        # -- 

        self.algo.animate_step(t)

        if self.verbose > 0:
            print("Done]")

        self.algo.per_frame_reset()

        if self.sleep:
            print("Sleeping for %ds"%(self.sleep))
            time.sleep(self.sleep)
    
        if snapshot_filename:
            return im.save(snapshot_filename,"gif")
        else:    
            return np.array(im)
        

    def __repr__(self):
        return """\
[FractalContext Img W:{w:d} Img H:{h:d} Cmplx W:{cw:.20f}
Cmplx H:{ch:.20f} c_real:{cr:s} c_imag:{ci:s} Scaling:{s:f} Epochs:{e:d} Max iter:{mx:d} Samples:{sm:d}]\
""".format(
        w=self.img_width,h=self.img_height,cw=self.cmplx_width,ch=self.cmplx_height,
        cr=str(self.c_real),ci=str(self.c_imag),s=self.scaling_factor,e=self.num_epochs,mx=self.max_iter,
        sm=self.samples); 

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


    def create_snapshot(self):    
    
        self.ctx.next_epoch(-1,self.vfilename)

    # --
    # Do any setup needed prior to running the calculatiion loop 
    # --

    def setup(self):

        # if we're doing a julia set walk, we need to animate. So 
        if fractal_ctx.algo_name == "julia":
            if fractal_ctx.algo.julia_list != None:
                self.ctx.dive = True
                self.ctx.c_real = hpf(0)
                self.ctx.c_imag = hpf(0)

        if not self.ctx.dive and not self.ctx.animate:
            set_snapshot_mode()
        elif self.ctx.dive:    
            set_dive_mode()

        print(self)
        print(self.ctx)

        self.ctx.algo.setup()

        if self.ctx.cache:
            self.ctx.cache.setup()

        # set burn in font size based on image size
        if self.ctx.img_width <= 480:
            self.ctx.burn_in_front_size = 8 
        elif self.ctx.img_width <= 640:
            self.ctx.burn_in_front_size = 10 
        elif self.ctx.img_width <= 1024:
            self.ctx.burn_in_front_size = 14 
        elif self.ctx.img_width <= 2048:
            self.ctx.burn_in_front_size = 24 
        elif self.ctx.img_width <= 4048:
            self.ctx.burn_in_front_size = 36 
            

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

    fractal_ctx.cmplx_width  = hpf(3.)
    fractal_ctx.cmplx_height = hpf(3. * (fractal_ctx.img_height / fractal_ctx.img_width) )

    fractal_ctx.scaling_factor = .75

    view_ctx.duration       = 4
    view_ctx.fps            = 4

def set_snapshot_mode():
    global fractal_ctx

    print(" + Running in snapshot mode ")

    fractal_ctx.snapshot  = True

    if not fractal_ctx.img_width:
        fractal_ctx.img_width  = 3840 
    if not fractal_ctx.img_height:     
        fractal_ctx.img_height = 2160 
    if not fractal_ctx.cmplx_width:
        fractal_ctx.cmplx_width  = hpf(4.)
    if not fractal_ctx.cmplx_height:
        fractal_ctx.cmplx_height = hpf(4. * (fractal_ctx.img_height / fractal_ctx.img_width))

    if not fractal_ctx.max_iter:
        fractal_ctx.max_iter   = 2048
    if not fractal_ctx.escape_rad:    
        fractal_ctx.escape_rad = 2. 
        #fractal_ctx.escape_rad = 32768. 
    if not fractal_ctx.samples:
        fractal_ctx.samples =65 

    if not fractal_ctx.c_real :
        if fractal_ctx.algo_name == "julia":
            print(" * Warning no center specified, setting to 0+0j")
            fractal_ctx.c_real = hpf(0) 
            fractal_ctx.c_imag = hpf(0) 
        else:    
            print(" * Warning no center specified, setting to -1+0j")
            fractal_ctx.c_real = hpf(-1.)
            fractal_ctx.c_imag = hpf(0.)

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
        fractal_ctx.cmplx_width  = hpf(4.)
    if not fractal_ctx.cmplx_height:
        fractal_ctx.cmplx_height = hpf(4. * (fractal_ctx.img_height / fractal_ctx.img_width))

    if not fractal_ctx.max_iter:
        fractal_ctx.max_iter   = 512
    if not fractal_ctx.escape_rad:    
        fractal_ctx.escape_rad = 2. 
        #fractal_ctx.escape_rad = 32768. 
    
    if not fractal_ctx.samples:
        fractal_ctx.samples = 9

    if fractal_ctx.c_real == None:
        print(" * Warning, no center specified, setting to -.749706+0.0314565j")
        fractal_ctx.c_real = hpf(-.749696000010025)
        fractal_ctx.c_imag = hpf(0.031456625003)

def parse_options():
    global fractal_ctx

    argv = sys.argv[1:]

    
    opts, args = getopt.getopt(argv, "pd:m:f:w:h:a:",
                               ["preview",
                                "algo=",
                                "duration=",
                                "max-iter=",
                                "sample=",
                                "img-w=",
                                "img-h=",
                                "res=",
                                "cmplx-w=",
                                "cmplx-h=",
                                "scaling-factor=",
                                "advance=",
                                "fps=",
                                "gif=",
                                "mpeg=",
                                "verbose=",
                                "julia-c=",
                                "julia-walk=",
                                "real=",
                                "imag=",
                                "palette-test=",
                                "color=",
                                "keyframe=",
                                "precision=",
                                "sample=",
                                "setcolor=",
                                "numprocs=",
                                "sleep=",
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
    

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration  = float(arg) 
            fractal_ctx.duration = float(arg) 
        elif opt in ['-m', '--max-iter']:
            fractal_ctx.max_iter = int(arg)
        elif opt in ['--sample']:
            fractal_ctx.samples = int(arg)
        elif opt in ['-w', '--img-w']:
            fractal_ctx.img_width = int(arg)
        elif opt in ['-h', '--img-h']:
            fractal_ctx.img_height = int(arg)
        elif opt in ['--res']:
            if str(arg) == "1k":
                fractal_ctx.img_width  = 1024
                fractal_ctx.img_height = 768
            elif str(arg) == "2k":
                fractal_ctx.img_width  = 2048
                fractal_ctx.img_height = 1536 
            elif str(arg) == "4k":
                fractal_ctx.img_width  = 3840
                fractal_ctx.img_height = 2160
            elif str(arg) == "8k":
                fractal_ctx.img_width  = 7680  
                fractal_ctx.img_height = 4320
            elif str(arg) == "12k":
                fractal_ctx.img_width  = 12288 
                fractal_ctx.img_height = 6480
            elif str(arg) == "16k":
                fractal_ctx.img_width  = 15360 
                fractal_ctx.img_height = 8640
            else:
                print("Error: unknown resolution (1k, 4k, 8k, 12k, and 16k supported)")
                sys.exit(0)

        elif opt in ['--cmplx-w']:
            fractal_ctx.cmplx_width = hpf(arg)
        elif opt in ['--cmplx-h']:
            fractal_ctx.cmplx_height = hpf(arg)
        elif opt in ['--real']:
            fractal_ctx.c_real = hpf(arg)
        elif opt in ['--imag']:
            fractal_ctx.c_imag = hpf(arg)
        elif opt in ['-h', '--img-h']:
            fractal_ctx.img_height = int(arg)
        elif opt in ['--scaling-factor']:
            fractal_ctx.scaling_factor = float(arg)
        elif opt in ['--advance']:
            fractal_ctx.advance = int(arg)
        elif opt in ['-f', '--fps']:
            view_ctx.fps  = int(arg)
            fractal_ctx.fps = int(arg)
        elif opt in ['--keyframe']:
            fractal_ctx.keyframe = int(arg) 
        elif opt in ['--sleep']:
            fractal_ctx.sleep = int(arg) 
        elif opt in ['--cache']:
            fractal_ctx.cache = fc.FractalCache(fractal_ctx) 
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

    fractal_ctx.algo.set_default_params()    
    fractal_ctx.algo.parse_options(opts, args)

if __name__ == "__main__":

    print("++ fractal.py version %s" % (FRACTAL_VER))
    
    parse_options()

    view_ctx.setup()
    view_ctx.run()

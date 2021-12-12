# - -
# File: ldnative.py
#
# Drive file for ldnative.c which is a C version of the
# mandelbrot using smoothing and native long doubles for speed. 
#
# -- 
import math
import sys
import os
import os.path

import subprocess
import multiprocessing

from algo import Algo
from algo import hpf

from PIL import Image

NativeLong_EXE = "./ldnative"
Gen_DIR        = "ldmfiles"

class LDNative(Algo):
    
    def __init__(self, context):
        super(LDNative, self).__init__(context) 
        self.exe   = NativeLong_EXE
        self.dir   = "./"+Gen_DIR+"/"
        self.color = (.1,.2,.3) 
        self.numprocs = 0
        #self.color = (.0,.6,1.0) 

    def parse_options(self, opts, args):    

        for opt,arg in opts:
            if opt in ['--setcolor']: # take colors 
                self.color = eval(arg) 
            elif opt in ['--numprocs']:
                self.numprocs = int(arg) 

        print('+ color to %s'%(str(self.color)))
        #print('+ number of samples %d'%(c_sample))

    def set_default_params(self):

        # set a more interesting point if we're going to be doing a dive    
        if self.context.dive and not self.context.c_real: 
            self.context.c_real = hpf(-0.235125)
            self.context.c_imag = hpf(0.827215)
        if not self.context.escape_rad:        
            self.context.escape_rad   = 256.
        if not self.context.max_iter:        
            self.context.max_iter     = 512

    def calc_pixel(self, c):
        assert 0

    def calc_cur_frame(self, img_width, img_height, x, xx, xxx, xxxx):

        filenames = []
        cmds      = []
        procs     = []

        c_real = self.context.c_real
        c_imag = self.context.c_imag
        c_w    = self.context.cmplx_width

        for i in range(0,self.numprocs):
            fn = self.dir+"ldm%d.png"%(i)
            filenames.append(fn)
            #cmd_args =  self.exe+" -w %d -h %d -n %d -c %d -i %s -x %.20f -y %.20f -l %.20f"%\
            cmd_args =  self.exe+"  -v -w %d -h %d -n %d -c %d -i %s -x %.20f -y %.20f -l %.20f -r %f -g %f -b %f"%\
                                 (img_width, img_height, self.numprocs, i+1, fn, c_real, c_imag, c_w, self.color[0], self.color[1], self.color[2])
            cmds.append(cmd_args)

        
        for cmd in cmds:
            print(" + [ldnative] executing command "+cmd);
            proc = subprocess.Popen(cmd, shell=True)
            procs.append(proc)

        for proc in procs:
            proc.wait()

        concat = Image.new('RGB', (img_width, img_height))
        i = 0
        for fn in filenames:
            im = Image.open(fn)
            concat.paste(im, (0, int(img_height/self.numprocs)*i))
            i = i + 1

        return concat 

    def _map_to_color(self, val):
        magnification = 1. / self.context.cmplx_width
        if magnification <= 100:
            magnification = 100 
        denom = math.log(math.log(magnification))

        c1 = 0.
        c2 = 0.
        c3 = 0.

        # (yellow blue 0,.6,1.0)
        c1 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[0]);
        c1 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[0]);
        c2 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[1]);
        c2 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[1]);
        c3 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[2]);
        c3 +=  0.5 + 0.5*math.cos( 3.0 + val*0.15 + self.color[2]);
        c1int = int(255.*((c1/4.) * 3.) / denom)
        c2int = int(255.*((c2/4.) * 3.) / denom)
        c3int = int(255.*((c3/4.) * 3.) / denom)
        return (c1int,c2int,c3int)

    def map_value_to_color(self, val):

        magnification = 1. / self.context.cmplx_width
        if magnification <= 100:
            magnification = 100 
        denom = math.log(math.log(magnification))

        if self.color:
            c1 = self._map_to_color(val)
            return c1 
        else:        
            #magnification = 1. / self.context.cmplx_width
            cint = int((val * 3.) / denom)
            return (cint,cint,cint)
        

    def animate_step(self, t):
        self.zoom_in()

    def setup(self):
        if self.numprocs == 0:
            cpus = multiprocessing.cpu_count()
            if cpus == 1:
                self.numprocs = cpus
            else:
                self.numprocs = int(cpus / 2)
            print(" + [ldnative] num procs not specified, setting to %d"%(self.numprocs)) 

        if not os.path.isfile(self.exe):
            print(" * [ldnative] Error : native executable %s not found "%(self.exe))
            sys.exit(0)
        if not os.path.isdir(self.dir):
            print(" + [ldnative] Directory %s not found, creating "%(self.dir))
            os.mkdir(self.dir)

def _instance(context):
    return LDNative(context)

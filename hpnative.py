# - -
# File: hpnative.py 
#
# Drive file for hpnative.c which is a C version of the
# mandelbrot that uses libbf high precision float library
#
# -- 
import math
import sys
import os
import os.path

import decimal
import subprocess
import multiprocessing

from algo import Algo

from PIL import Image

PRECISION      = 300
NativeLong_EXE = "./hpnative"
Gen_DIR        = "hpfiles"

hpf = decimal.Decimal
decimal.getcontext().prec = PRECISION 


c_width  = hpf(0.)
c_height = hpf(0.)
c_real   = hpf(0.)
c_imag   = hpf(0.)
magnification = hpf(0.)

scaling_factor = 0.
num_epochs     = 0

class HPNative(Algo):
    
    def __init__(self, context):
        super(HPNative, self).__init__(context) 
        self.exe   = NativeLong_EXE
        self.dir   = "./"+Gen_DIR+"/"
        self.color = (.1,.2,.3) 
        self.numprocs = 0
        #self.color = (.0,.6,1.0) 

    def parse_options(self, opts, args):    
        for opt,arg in opts:
            if opt in ['--nocolor']:
                self.color = None 
            if opt in ['--numprocs']:
                self.numprocs = int(arg) 
            if opt in ['--setcolor']: # XXX TODO
                pass
                #self.color = (.1,.2,.3)   # dark
                #self.color = (.0,.6,1.0) # blue / yellow

    def set_default_params(self):

        # set a more interesting point if we're going to be doing a dive    
        if self.context.dive and not self.context.cmplx_center: 
            self.context.cmplx_center = self.context.ctxc(-0.235125,0.827215)
        if not self.context.escape_rad:        
            self.context.escape_rad   = 256.
        if not self.context.max_iter:        
            self.context.max_iter     = 2048

    def calc_pixel(self, c):
        assert 0

    def calc_cur_frame(self, img_width, img_height, x, xx, xxx, xxxx):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        filenames = []
        cmds      = []
        procs     = []

        for i in range(0,self.numprocs):
            fn = self.dir+"hpm%d.png"%(i)
            filenames.append(fn)
            cmd_args =  self.exe+" -v -w %d -h %d -n %d -b %d -i %s -p %d -m %d -x \"%s\" -y \"%s\" -l \"%s\""%\
                                 (img_width, img_height, self.numprocs, i+1, fn, PRECISION, self.context.max_iter, \
                                 str(c_real), str(c_imag), str(c_width) )
            cmds.append(cmd_args)

        
        for cmd in cmds:
            print(" + [hpnative] executing command "+cmd);
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

    def zoom_in(self, iterations=1):
        global c_width
        global c_height
        global scaling_factor
        global magnification
        global num_epochs

        while iterations > 0:
            c_width   *= hpf(scaling_factor)
            c_height  *= hpf(scaling_factor)
            magnification *= scaling_factor
            iterations -= 1

            self.context.num_epochs += 1

    def setup(self):
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        c_width  = hpf(self.context.cmplx_width)
        c_height = hpf(self.context.cmplx_height)
        c_real = hpf('-1.769383179195515018213847286085473782905747263654751437465528216527888191264756458836163446389529667304485825781820303157487491238')
        c_imag = hpf('0.00423684791873677221492650717136799707668267091740375727945943565011234400080554515730243099502363650631353268335965257182300494805')
        #c_real = hpf('-1')
        #c_imag = hpf('0')

        scaling_factor = self.context.scaling_factor
        magnification = self.context.magnification
        num_epochs = self.context.num_epochs

        if self.numprocs == 0:
            cpus = multiprocessing.cpu_count()
            if cpus == 1:
                self.numprocs = cpus
            else:
                self.numprocs = int(cpus / 2)
            print(" + [hpnative] num procs not specified, setting to %d"%(self.numprocs)) 

        if not os.path.isfile(self.exe):
            print(" * [hpnative] Error : native executable %s not found "%(self.exe))
            sys.exit(0)
        if not os.path.isdir(self.dir):
            print(" + [hpnative] Directory %s not found, creating "%(self.dir))
            os.mkdir(self.dir)

def _instance(context):
    return HPNative(context)

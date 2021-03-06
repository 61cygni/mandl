# - -
# File: mpfrnative.py 
#
# Drive file for mpfrnative.c which is a C version of the
# mandelbrot that uses mpfr high precision float library
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

from globalconfig import *

NativeLong_EXE = "./mpfrnative"
Gen_DIR        = "mpfrfiles"

c_width  = hpf(0.)
c_height = hpf(0.)
c_real   = hpf(0.)
c_imag   = hpf(0.)
magnification = hpf(0.)

scaling_factor = 0.
num_epochs     = 0

class MPFRNative(Algo):
    
    def __init__(self, context):
        super(MPFRNative, self).__init__(context) 
        self.exe   = NativeLong_EXE
        self.dir   = "./"+Gen_DIR+"/"
        self.color = (.1,.2,.3) 
        self.numprocs  = 0
        self.precision = 0
        self.samples  = 17
        #self.color = (.0,.6,1.0) 

    def parse_options(self, opts, args):    
        for opt,arg in opts:
            if opt in ['--numprocs']:
                self.numprocs = int(arg) 
            elif opt in ['--precision']:
                self.precision = int(arg) 
            elif opt in ['--setcolor']: # XXX TODO
                self.color = eval(arg) 
                #self.color = (.1,.2,.3)   # dark
                #self.color = (.0,.6,1.0) # blue / yellow

        self.samples = self.context.samples 

        if self.precision == 0:
            self.precision = GLOBAL_PRECISION
            decimal.getcontext().prec = self.precision 

        print('+ color to %s'%(str(self.color)))
        print('+ number of samples %d'%(self.samples))
        print('+ precision %d'%(self.precision))


    def set_default_params(self):

        # set a more interesting point if we're going to be doing a dive    
        if self.context.dive and not self.context.c_real: 
            self.context.c_real = hpf(-0.235125)
            self.context.c_imag = hpf(0.827215)
        if not self.context.escape_rad:        
            self.context.escape_rad   = 256.
        if not self.context.max_iter:        
            self.context.max_iter     = 8000

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

        m_i    = self.context.max_iter
        c_real = self.context.c_real
        c_imag = self.context.c_imag
        c_w    = self.context.cmplx_width
        p      = self.precision

        for i in range(0,self.numprocs):
            fn = self.dir+"mpfr%d.png"%(i)
            filenames.append(fn)
            cmd_args =  self.exe+" -m %d -p %d -v -w %d -h %d -n %d -c %d -i %s  -x %s -y %s -l %s -s %d -r %f -g %f -b %f"%\
                                 (m_i, p, img_width, img_height, self.numprocs, i+1, fn, str(c_real), str(c_imag), str(c_w), self.samples, self.color[0], self.color[1], self.color[2])
            cmds.append(cmd_args)

        
        for cmd in cmds:
            print(" + [mpfrnative] executing command "+cmd);
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
        global c_width
        global c_height
        global c_real
        global c_imag
        global scaling_factor
        global magnification
        global num_epochs

        # Used for testing
        # c_width  = hpf(self.context.cmplx_width)
        # c_height = hpf(self.context.cmplx_height)
        # c_real = hpf('-1.769383179195515018213847286085473782905747263654751437465528216527888191264756458836163446389529667304485825781820303157487491238')
        # c_imag = hpf('0.00423684791873677221492650717136799707668267091740375727945943565011234400080554515730243099502363650631353268335965257182300494805')
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
            print(" + [mpfrnative] num procs not specified, setting to %d"%(self.numprocs)) 

        if not os.path.isfile(self.exe):
            print(" * [mpfrnative] Error : native executable %s not found "%(self.exe))
            sys.exit(0)
        if not os.path.isdir(self.dir):
            print(" + [mpfrnative] Directory %s not found, creating "%(self.dir))
            os.mkdir(self.dir)

def _instance(context):
    return MPFRNative(context)

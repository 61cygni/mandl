# --
# File: fractalcache.py
# Sun Jul 18 18:02:48 PDT 2021 
#
# --

import os

import hmac
import hashlib
import struct
import pickle


FRACTL_CACHE_DIR = "./.fractal_cache/"

class Frame:

    def __init__(self, ver, cw, ch, center, escape_r, m_iter, values, histogram):
        self.ver = ver
        self.cw  = cw
        self.ch  = ch
        self.center    = center
        self.escape_r  = escape_r
        self.m_iter    = m_iter
        self.values    = values
        self.histogram = histogram


class FractalCache:

    def __init__(self, mandl_ctx, root_dir = FRACTL_CACHE_DIR):
        self.ctx = mandl_ctx
        self.root_dir = FRACTL_CACHE_DIR

        self.subdir = None

    def create_subdir_name(self):
        return self.root_dir + "/" + str(self.ctx.img_width)+"x"+str(self.ctx.img_height)

    def create_file_name(self):
        # create filename from a deterministic hash of the important
        # context
        ba = struct.pack("ffffffi",
                              float(self.ctx.ver),
                              self.ctx.cmplx_width,
                              self.ctx.cmplx_height,
                              self.ctx.cmplx_center.real,
                              self.ctx.cmplx_center.imag,
                              self.ctx.escape_rad,
                              self.ctx.max_iter)

        h = hmac.new(ba, digestmod=hashlib.sha1)
        return h.hexdigest()
        
    def setup(self):
        print("+ using cache %s" % (self.root_dir))

        if not os.path.isdir(self.root_dir):
            print("  -> directory not found ...  creating ")
            os.mkdir(self.root_dir)

        self.subdir = self.create_subdir_name() 
        if not os.path.isdir(self.subdir):
            print("  ->  subdir %s not found ...  creating "%(self.subdir))
            os.mkdir(self.subdir)

    def check_cache(self):

        filename = self.subdir + "/" + self.create_file_name()
        if not os.path.isfile(filename):
            print("  ->  cache file %s not found ...  "%(filename))
            return False
        return True    

    def write_cache(self, values, histogram):
        
        filename = self.subdir + "/" + self.create_file_name()
        print("+  writing frame to cache file %s ...  "%(filename))
        frame = Frame(float(self.ctx.ver),
                      self.ctx.cmplx_width,
                      self.ctx.cmplx_height,
                      self.ctx.cmplx_center,
                      self.ctx.escape_rad,
                      self.ctx.max_iter,
                      values,
                      histogram)

        with open(filename, 'wb') as fd:
            pickle.dump(frame,fd)

    def read_cache(self):
        
        if not self.check_cache():
            return None, None

        filename = self.subdir + "/" + self.create_file_name()

        print("+  frame from cache file %s ...  "%(filename))
        frame = None 

        with open(filename, 'rb') as fd:
            frame = pickle.load(fd)

        assert frame.ver == float(self.ctx.ver)
        assert frame.cw  == self.ctx.cmplx_width 
        assert frame.ch  == self.ctx.cmplx_height 
        assert frame.center   == self.ctx.cmplx_center 
        assert frame.escape_r == self.ctx.escape_rad 
        assert frame.m_iter   == self.ctx.max_iter 

        return frame.values,frame.histogram

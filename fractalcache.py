# --
# File: fractalcache.py
# Sun Jul 18 18:02:48 PDT 2021 
#
# --

import os
import sys

import hmac
import hashlib
import struct
import pickle

from decimal import Decimal

CACHE_VER = 0.11

class FrameInfo:
    def __init__(self, mesh_width, mesh_height, center, complex_real_width, complex_imag_width, escape_r, max_escape_iter,  raw_values=None, raw_histogram=None, smooth_values=None, smooth_histogram=None):
        self.mesh_width         = mesh_width
        self.mesh_height        = mesh_height
        self.center             = center
        self.complex_real_width = complex_real_width
        self.complex_imag_width = complex_imag_width
        self.escape_r           = escape_r
        self.max_escape_iter    = max_escape_iter

        self.raw_values         = raw_values
        self.raw_histogram      = raw_histogram
        self.smooth_values      = smooth_values
        self.smooth_histogram   = smooth_histogram

    def emptyCopy(self):
        """ Looks like storing strings of everything makes us pickle-proof? """
        return FrameInfo(str(self.mesh_width), str(self.mesh_height), str(self.center), str(self.complex_real_width), str(self.complex_imag_width), str(self.escape_r), str(self.max_escape_iter))
        #return FrameInfo(self.mesh_width, self.mesh_height, self.center, self.complex_real_width, self.complex_imag_width, self.escape_r, self.max_escape_iter)

    def pickleCopy(self):
        return FrameInfo(str(self.mesh_width), str(self.mesh_height), str(self.center), str(self.complex_real_width), str(self.complex_imag_width), str(self.escape_r), str(self.max_escape_iter), self.raw_values, self.raw_histogram, self.smooth_values, self.smooth_histogram)

class Frame:
    """
    Two different results caches available, one for uniform meshes 
    at "<timeline.sharedCachePath>", and a project-specific one for 
    non-uniform meshes at "<timeline.projectFolderName>_cache".

    The project-specific cache uses just frame numbers for cache file names.
    The shared cache uses a hash of this frame's FrameInfo for cache file names.

    A shared results cache called 'shared_cache' would have directories like:
    shared_cache/v_0.11/native/mandelbrot/deadbeef
    shared_cache/v_0.11/flint/mandelbrot/deadbeef
    shared_cache/v_0.11/native/julia/feedbabe
    shared_cache/v_0.11/flint/julia/feedbabe

    And the project called 'demo1' would have non-uniform meshes cached at:
    demo1/v_0.11/native/mandelbrot/123.pik
    demo1/v_0.11/flint/mandelbrot/123.pik
    demo1/v_0.11/native/julia/12.pik
    demo1/v_0.11/flint/julia/13.pik
    """
    def __init__(self, timeline, dive_mesh, frame_number=-1, raw_values=None, raw_histogram=None, smooth_values=None, smooth_histogram=None):
        self.cache_version = CACHE_VER
        self.timeline      = timeline
        self.dive_mesh     = dive_mesh
        self.frame_number  = frame_number

        self.frame_info    = FrameInfo(self.dive_mesh.meshWidth, self.dive_mesh.meshHeight, self.dive_mesh.center, self.dive_mesh.realMeshGenerator.baseWidth, self.dive_mesh.imagMeshGenerator.baseWidth, self.dive_mesh.escapeRadius, self.dive_mesh.maxEscapeIterations, raw_values, raw_histogram, smooth_values, smooth_histogram)

    def create_results_file_name(self):
        return os.path.join(self.create_results_subpath(), self.create_results_file_identifier())

    def create_results_file_identifier(self):
        if self.dive_mesh.isUniform():
            ba = None
            ba = pickle.dumps(self.frame_info.emptyCopy())
            h = hmac.new(ba, digestmod=hashlib.sha1)
            return h.hexdigest()
        else:
            return u"%d.pik" % self.frame_number

    def create_results_subpath(self, mkdir_if_needed=False):
        if self.dive_mesh.isUniform():
            root_cache_path = self.timeline.sharedCachePath
        else:
            root_cache_path = u"%s_cache" % self.timeline.projectFolderName

        version_subdir = u"v_%s" % str(self.cache_version)
        results_subpath = os.path.join(root_cache_path, version_subdir, self.timeline.mathSupport.precisionType, self.timeline.fractalType) 
        if mkdir_if_needed and not os.path.exists(results_subpath):
            os.makedirs(results_subpath)
        return results_subpath
   
    def write_results_cache(self):
        self.create_results_subpath(mkdir_if_needed=True) # Just for side-effect folder creation

        filename = self.create_results_file_name()
        #print("+  writing frame to cache file %s ...  "%(filename))

        # Probably a mistake to write a no-data cache file, so panic.
        if self.frame_info.raw_values is None or self.frame_info.smooth_values is None:
            raise ValueError("Aborting cache file write of missing data to \"%s\"" % filename)

        with open(filename, 'wb') as fd:
            pickle.dump(self.frame_info.pickleCopy(),fd)

    def read_results_cache(self):
        filename = self.create_results_file_name()
        if not os.path.exists(filename) or not os.path.isfile(filename):
            return 

        #print("+  frame from cache file %s ...  "%(filename))
        frame_data = None 

        with open(filename, 'rb') as fd:
            frame_data = pickle.load(fd)
  
#        assert frame_data.mesh_width == self.frame_info.mesh_width
#        assert frame_data.mesh_height == self.frame_info.mesh_height
#        assert frame_data.center == self.frame_info.center
#        assert frame_data.complex_real_width == self.frame_info.complex_real_width
#        assert frame_data.complex_imag_width == self.frame_info.complex_imag_width
#        assert frame_data.escape_r == self.frame_info.escape_r
#        assert frame_data.max_escape_iter == self.frame_info.max_escape_iter

        self.frame_info.raw_values         = frame_data.raw_values
        self.frame_info.raw_histogram      = frame_data.raw_histogram
        self.frame_info.smooth_values      = frame_data.smooth_values
        self.frame_info.smooth_histogram   = frame_data.smooth_histogram

        return 

    def remove_from_results_cache(self):
        cache_file_name = self.create_results_file_name()
        if os.path.exists(cache_file_name) and os.path.isfile(cache_file_name):
            os.remove(cache_file_name)
            self.raw_values = None
            self.raw_histogram = None
            self.smooth_values = None
            self.smooth_histogram = None


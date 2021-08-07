# --
# File: mandeldistance.py
#
#
# --

from algo import Algo

import math
import numpy as np

from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from algo import EscapeFrameInfo, EscapeAlgo
import fractalpalette as fp


def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)


class MandelDistance(EscapeAlgo):
    def __init__(self, dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params={}):
        super().__init__(dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params)

        self.algorithm_name = 'mandeldistance'

    def calculate_results(self):
        mesh_array = self.dive_mesh.generateMesh()
        math_support = self.dive_mesh.mathSupport

        #mandelbrot_function = np.vectorize(math_support.orig_mandelbrotDistanceEstimate)
#        mandelbrot_function = np.vectorize(math_support.mandelbrotDistanceEstimate)
#        self.cache_frame.frame_info.raw_values = mandelbrot_function(mesh_array, self.escape_radius, self.max_escape_iterations)


        mandelbrot_function = np.vectorize(math_support.mandelbrotDistanceEstimate)
        (pixel_values_2d, distances) = mandelbrot_function(mesh_array, self.escape_radius, self.max_escape_iterations)

        rescaleFunction = np.vectorize(math_support.rescaleForRange)
        (pixel_values_2d_smoothed) = rescaleFunction(distances, pixel_values_2d, self.max_escape_iterations, self.dive_mesh.realMeshGenerator.baseWidth)

        self.cache_frame.frame_info.raw_values = pixel_values_2d
        self.cache_frame.frame_info.smooth_values = pixel_values_2d_smoothed    

        return

    def generate_image(self):
        # Capturing the transpose of our array, because it looks like I mixed
        # up rows and cols somewhere along the way.
        if self.use_smoothing == True:
            pixel_values_2d = self.cache_frame.frame_info.smooth_values.T
        else:
            pixel_values_2d = self.cache_frame.frame_info.raw_values.T

        ##pixel_values_2d = self.cache_frame.frame_info.raw_values.T
        #print("shape of things to come: %s" % str(pixel_values_2d.shape))


# TODO: Really, width and height are all kinda incorrect here - 
# gotta spend some TLC on the array shape and transpose.
        (image_width, image_height) = pixel_values_2d.shape
        im = Image.new('RGB', (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, image_width):
            for y in range(0, image_height):
                color = self.map_value_to_color(pixel_values_2d[x,y])

                # Plot the point
                draw.point([x, y], color) 

        if self.burn_in == True:
            meta = self.get_frame_metadata()
            if meta:
                burn_in_text = u"%d" % (meta['frame_number'])
                self.burn_text_to_drawing(burn_in_text, draw)

        return im    

    def map_value_to_color(self, raw_val):
        cint = int(float(raw_val) * 255)
        return (cint,cint,cint)

#    def map_value_to_color(self, raw_val):
#        ## Was an error, but when val was negative, this blew up.
#        val = float(raw_val)
#        if val < 0.0:
#            val = 0.0
#        zoo = .1 
#        zoom_level = 1. / (self.dive_mesh.imagMeshGenerator.baseWidth)
#        d = clamp( pow(zoom_level * val/zoo,0.1), 0.0, 1.0 );
#        #if math.isnan(d):
#        #    print("NAN val: %s zoom_level: %s width: %s" % (str(val), str(zoom_level), self.dive_mesh.imagMeshGenerator.baseWidth))
#        cint = int(float(d)*255)
#        # Forced float() here because baseWidth is maybe a special math subtype
#
#        return (cint,cint,cint)



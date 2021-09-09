# --
# File: julia_solo.py
#
# 
# Interesting julia-center points:
# -.8+.145j
#
# --

import os
import pickle

import math
import numpy as np

from PIL import Image, ImageDraw

from julia_solo import JuliaSolo
import fractalpalette as fp

class JuliaSmooth(JuliaSolo):

    def __init__(self, dive_mesh, frame_number, output_folder_name, extra_params={}):
        super().__init__(dive_mesh, frame_number, output_folder_name, extra_params)

        self.algorithm_name = 'julia_smooth'

        # TODO: really should be handled by a palette
        self.color = (.0,.6,1.0) # blue / yellow

    def process_counts(self):
        smoothing_function = np.vectorize(self.dive_mesh.mathSupport.smoothAfterCalculation)
        self.processed_array = smoothing_function(self.last_values_array, self.counts_array, self.max_escape_iterations, self.escape_radius)

    def generate_image(self):
        # Capturing the transpose of our array, because it looks like I mixed
        # up rows and cols somewhere along the way.
        if self.use_smoothing == True:
            pixel_values_2d = self.cache_frame.frame_info.smooth_values.T
        else:
            pixel_values_2d = self.cache_frame.frame_info.raw_values.T
        #print("shape of things to come: %s" % str(pixel_values_2d.shape))

# TODO: Really, width and height are all kinda incorrect here - 
# gotta spend some TLC on the array shape and transpose.
        (image_width, image_height) = pixel_values_2d.shape
        im = Image.new('RGB', (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, image_width):
            for y in range(0, image_height):
                color = self.palette.map_value_to_color(pixel_values_2d[x,y])

                # Plot the point
                draw.point([x, y], color) 

        if self.burn_in == True:
            meta = self.get_frame_metadata()
            if meta:
                burn_in_text = u"%d center: %s\n    realw: %s imagw: %s" % (meta['frame_number'], meta['mesh_center'], meta['complex_real_width'], meta['complex_imag_width'])
                self.burn_text_to_drawing(burn_in_text, draw)

        return im    

    def generate_image(self):
        (image_height, image_width) = self.processed_array.shape
        im = Image.new('RGB', (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        # Note: Image's width,height is backwards from numpy's size (rows, cols)
        for x in range(0, image_width):
            for y in range(0, image_height):
                color = self.map_value_to_color(self.processed_array[y,x])

                # Plot the point
                draw.point([x, y], color)

        if self.burn_in == True:
            meta = self.get_metadata()
            if meta:
                burn_in_text = u"%d" % (meta['frame_number'])
                self.burn_text_to_drawing(burn_in_text, draw)

        image_filename_base = u"%d.tiff" % self.frame_number
        self.output_image_file_name = os.path.join(self.output_folder_name, image_filename_base)
        im.save(self.output_image_file_name)

    def map_value_to_color(self, val):
        # TODO: should make this a special rotating palette,
        # or bound the values before doing a lookup to a palette.
        if math.isnan(val):
            return (0,0,0)

        if self.color:
            # (yellow blue 0,.6,1.0)
            c1 = 1 + math.cos(3.0 + val*0.15 + self.color[0])
            c2 = 1 + math.cos(3.0 + val*0.15 + self.color[1])
            c3 = 1 + math.cos(3.0 + val*0.15 + self.color[2])

            if c1 <= 0 or math.isnan(c1):
                c1int = 0
            else:
                c1int = int(255.*((c1/4.) * 3.) / 1.5)
            if c2 <= 0 or math.isnan(c2):
                c2int = 0
            else:
                c2int = int(255.*((c2/4.) * 3.) / 1.5)
            if c3 <= 0 or math.isnan(c3):
                c3int = 0
            else:
                c3int = int(255.*((c3/4.) * 3.) / 1.5)

            return (c1int,c2int,c3int)
        else:
            c1 = 1 + math.cos(3.0 + val*0.15)
            cint = int(255.*((c1/4.) * 3.) / 1.5)
            return (cint,cint,cint)



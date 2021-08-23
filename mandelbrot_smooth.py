# --
# File: mandelbrot_smooth.py
#
# Subclass of MandelbrotSolo, which implements a process_counts() 
# function that populates 'processed_array' with processed values, 
# instead of defaulting to the counts array.
#
# --

import os

import math
import numpy as np

from PIL import Image, ImageDraw

from mandelbrot_solo import MandelbrotSolo

class MandelbrotSmooth(MandelbrotSolo):

# TODO: Yeah, really should be a palette
#
#    @staticmethod
#    def options_list():
#        whole_list = MandelbrotSolo.options_list()
#        whole_list.extend(["color="])
#        return whole_list
#
#    @staticmethod
#    def parse_options(opts):
#        options = MandelbrotSolo.parse_options(opts)
#        for opt,arg in opts:
#            if opt in ['--color']:
#                options['color'] = (.1,.2,.3)   # dark
#                #options['color'] = (.0,.6,1.0) # blue / yellow
#                #options['color'] = (.0,.6,1.0)
#
#        return options

    def __init__(self, dive_mesh, frame_number, output_folder_name, extra_params={}):
        super().__init__(dive_mesh, frame_number, output_folder_name, extra_params)

        self.algorithm_name = 'mandelbrot_smooth'

        self.color = (.0,.6,1.0) # blue / yellow
        #self.color = (.1,.2,.3) # dark

    def process_counts(self):
        smoothing_function = np.vectorize(self.dive_mesh.mathSupport.smoothAfterCalculation)
        self.processed_array = smoothing_function(self.last_values_array, self.counts_array, self.max_escape_iterations, self.escape_radius)

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


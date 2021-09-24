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

from fractalpalette import FractalPaletteWithSchemes

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

        # Hacky hard-code for now.
        # TODO: need to persist palette definitions with the timeline json.
        self.palette = FractalPaletteWithSchemes([(.0,.6,1.0), # blue-tan 
            (1.0,.4,.4), # teal-pink
            (1.0,.2,.4), # green-pink
            (.4,.2,.6), # green-purple
        ])
        self.palette_index = extra_params.get('palette_scheme_index', 0) 

    def process_counts(self):
        #print(f"process_counts says mathSupport is {self.dive_mesh.mathSupport.precisionType}")
        self.processed_array = self.dive_mesh.mathSupport.smoothAfterCalculation(self.last_values_real_array, self.last_values_imag_array, self.counts_array, self.max_escape_iterations, self.escape_radius)
        #smoothing_function = np.vectorize(self.dive_mesh.mathSupport.smoothAfterCalculation)
        #self.processed_array = smoothing_function(self.last_values_array, self.counts_array, self.max_escape_iterations, self.escape_radius)

    def generate_image(self):
        self.palette.set_scheme_index(self.palette_index) 
        super().generate_image()


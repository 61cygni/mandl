# --
# File: mandelbrot.py
#
# Basic escape iteration method for calculating the mandlebrot set
#
# Code cribbed from all over the place ... notably :
#
# https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
# http://linas.org/art-gallery/escape/escape.html
#
# Misiurewicz points also cribbed from all over 
#
# https://mrob.com/pub/muency/misiurewiczpoint.html
# https://www.youtube.com/watch?v=u1pwtSBTnPU&t=274s
#
#
# MPs:
#
# 0.4244 + 0.200759i;

import os
import pickle

import math
import numpy as np

from collections import defaultdict

from PIL import Image, ImageDraw

from algo import EscapeAlgo
import fractalpalette as fp

class MandelbrotSolo(EscapeAlgo):
    def __init__(self, dive_mesh, frame_number, output_folder_name, extra_params={}):
        super().__init__(dive_mesh, frame_number, output_folder_name, extra_params)

        self.algorithm_name = 'mandelbrot_solo'

    def generate_counts(self):
        math_support = self.dive_mesh.mathSupport

        #mandelbrot_function = np.vectorize(math_support.mandelbrot_beginning)
        mandelbrot_function = np.vectorize(math_support.mandelbrot)
        (self.counts_array, self.last_values_array) = mandelbrot_function(self.mesh_array, self.escape_radius, self.max_escape_iterations)

        counts_name_base = u"%d.counts.pik" % self.frame_number
        counts_file_name = os.path.join(self.output_folder_name, counts_name_base)
        with open(counts_file_name, 'wb') as counts_handle:
            pickle.dump(self.counts_array, counts_handle)

    def pre_image_hook(self):
        hist = defaultdict(int) 

        # numpyArray.shape returns (rows, columns)
        for y in range(0, self.mesh_array.shape[0]):
            for x in range(0, self.mesh_array.shape[1]):
                # Not using mathSupport's floor() here, because it should just be a normal-scale float
                if self.counts_array[y,x] < self.max_escape_iterations:
                    #print("x: %d, y: %d, val: %s, floor: %s" % (x,y,str(counts_array[y,x]), str(math.floor(counts_array[y,x]))))
                    hist[math.floor(self.counts_array[y,x])] += 1

        self.palette.histogram = hist
        self.palette.calc_hues(self.max_escape_iterations)

    def generate_image(self):
        (image_height, image_width) = self.processed_array.shape
        im = Image.new('RGB', (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        # Note: Image's width,height is backwards from numpy's size (rows, cols)
        for x in range(0, image_width):
            for y in range(0, image_height):
                color = self.palette.map_value_to_color(self.processed_array[y,x])

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

    def ending_hook(self):
        self.palette.per_frame_reset()



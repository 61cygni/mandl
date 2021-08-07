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

import math
import numpy as np

from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from algo import EscapeFrameInfo, EscapeAlgo
import fractalpalette as fp


class MandelbrotSolo(EscapeAlgo):
    def __init__(self, dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params={}):
        super().__init__(dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params)

        self.algorithm_name = 'mandelbrot_solo'

    def generate_results(self):
        """
        Special no-cache implementation of the main results sequence
        """
        self.cache_frame = self.build_cache_frame()
        self.calculate_results()

        return

    def calculate_results(self):
        mesh_array = self.dive_mesh.generateMesh()
        math_support = self.dive_mesh.mathSupport

        mandelbrot_function = np.vectorize(math_support.mandelbrot)
        (pixel_values_2d, last_zees) = mandelbrot_function(mesh_array, self.escape_radius, self.max_escape_iterations)

        hist = defaultdict(int) 

        for x in range(0, mesh_array.shape[0]):
            for y in range(0, mesh_array.shape[1]):
                # Not using mathSupport's floor() here, because it should just be a normal-scale float
                if pixel_values_2d[x,y] < self.max_escape_iterations:
                    #print("x: %d, y: %d, val: %s, floor: %s" % (x,y,str(pixel_values_2d[x,y]), str(math_support.floor(pixel_values_2d[x,y]))))
                    hist[math.floor(pixel_values_2d[x,y])] += 1

        self.cache_frame.frame_info.raw_values = pixel_values_2d
        self.cache_frame.frame_info.raw_histogram = hist

        return

    def pre_image_hook(self):
        self.palette.histogram = self.cache_frame.frame_info.raw_histogram
        self.palette.calc_hues(self.max_escape_iterations)

    def generate_image(self):
        pixel_values_2d = self.cache_frame.frame_info.raw_values.T

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
                burn_in_text = u"%d" % (meta['frame_number'])
                self.burn_text_to_drawing(burn_in_text, draw)

        return im    

    def ending_hook(self):
        self.palette.per_frame_reset()



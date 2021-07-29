import math
import numpy as np

from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from algo import JuliaFrameInfo, JuliaAlgo

import fractalpalette as fp

class Julia(JuliaAlgo):

    def __init__(self, dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params={}):
        super().__init__(dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params)

        self.algorithm_name = 'julia'

    def calculate_results(self):
        mesh_array = self.dive_mesh.generateMesh()
        math_support = self.dive_mesh.mathSupport

        julia_function = np.vectorize(math_support.julia)
        (pixel_values_2d, last_zees) = julia_function(self.julia_center, mesh_array, self.escape_radius, self.max_escape_iterations)

        smoothing_function = np.vectorize(math_support.smoothAfterCalculation)
        pixel_values_2d_smoothed = smoothing_function(last_zees, pixel_values_2d, self.max_escape_iterations)

        hist = defaultdict(int) 
        hist_smoothed = defaultdict(int) 

        for x in range(0, mesh_array.shape[0]):
            for y in range(0, mesh_array.shape[1]):
                # Not using mathSupport's floor() here, because it should just be a normal-scale float
                if pixel_values_2d[x,y] < self.max_escape_iterations:
                    #print("x: %d, y: %d, val: %s, floor: %s" % (x,y,str(pixel_values_2d[x,y]), str(math_support.floor(pixel_values_2d[x,y]))))
                    hist[math.floor(pixel_values_2d[x,y])] += 1
                if pixel_values_2d_smoothed[x,y] < self.max_escape_iterations:
                    hist_smoothed[math.floor(pixel_values_2d_smoothed[x,y])] += 1

        self.cache_frame.frame_info.raw_values = pixel_values_2d
        self.cache_frame.frame_info.raw_histogram = hist
        self.cache_frame.frame_info.smooth_values = pixel_values_2d_smoothed
        self.cache_frame.frame_info.smooth_histogram = hist_smoothed

    def pre_image_hook(self):
        # Capturing the transpose of our array, because it looks like I mixed
        # up rows and cols somewhere along the way.
        if self.use_smoothing == True:
            self.palette.histogram = self.cache_frame.frame_info.smooth_histogram
        else:
            self.palette.histogram = self.cache_frame.frame_info.raw_histogram

        # TODO: Hey, calc_hues should get a different range of bins if
        # smoothed values are floats!
        self.palette.calc_hues(self.max_escape_iterations)

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

                burn_in_location = (10,10)
                burn_in_margin = 5 
                burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
                burn_in_size = burn_in_font.getsize_multiline(burn_in_text)
                draw.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
                draw.text(burn_in_location, burn_in_text, 'white', burn_in_font)

        return im    

    def ending_hook(self):
        self.palette.per_frame_reset()
        


# --
# File: smooth.py
#
# Implementation of the smoothing algorithm for iteration escape method
# of drawing mandelbrot as describe and implemented by Inigo Quilez
#
# https://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
# https://www.shadertoy.com/view/4df3Rn coloring and smoothing working
#
#
# Seashell cove - -0.745+0.186j
#
# -- 
import math
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from algo import Algo, EscapeFrameInfo, EscapeAlgo


class Smooth(EscapeAlgo):

    @staticmethod
    def parse_options(opts):
        """ Smooth ignores the parameter --smooth, because it's redundantish """
        options = EscapeAlgo.parse_options(opts)

        for opt,arg in opts:
            if opt in ['--color']:
                options['color'] = (.1,.2,.3)   # dark
                #options['color'] = (.0,.6,1.0) # blue / yellow
                #options['color'] = (.0,.6,1.0)

        return options

    def __init__(self, dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params={}):
        super().__init__(dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params)

        self.algorithm_name = 'smooth'
        self.color = extra_params.get('color', None)

    def calculate_results(self):
        mesh_array = self.dive_mesh.generateMesh()
        math_support = self.dive_mesh.mathSupport

        mandelbrot_function = np.vectorize(math_support.mandelbrot)
        (pixel_values_2d, last_zees) = mandelbrot_function(mesh_array, self.escape_radius, self.max_escape_iterations)

        smoothing_function = np.vectorize(math_support.smoothAfterCalculation)
        pixel_values_2d_smoothed = smoothing_function(last_zees, pixel_values_2d, self.max_escape_iterations, self.escape_radius)

        self.cache_frame.frame_info.raw_values = pixel_values_2d
        self.cache_frame.frame_info.smooth_values = pixel_values_2d_smoothed

        return

    def generate_image(self):
        # Capturing the transpose of our array, because it looks like I mixed
        # up rows and cols somewhere along the way.
        # Using smooth only here, because that's what this is.
        pixel_values_2d = self.cache_frame.frame_info.smooth_values.T

# TODO: Really, width and height are all kinda incorrect here - 
# gotta spend some TLC on the array shape and transpose.
        (image_width, image_height) = pixel_values_2d.shape
        im = Image.new('RGB', (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, image_width):
            for y in range(0, image_height):
                # NOTE: This is the difference - not using palette, but
                # instead, a (kinda locally) calculated range
                color = self.map_value_to_color(pixel_values_2d[x,y])

                # Plot the point
                draw.point([x, y], color) 

        if self.burn_in == True:
            meta = self.get_frame_metadata()
            if meta:
                burn_in_text = u"%d" % (meta['frame_number'])

                burn_in_location = (10,10)
                burn_in_margin = 5 
                burn_in_font = ImageFont.truetype('fonts/cour.ttf', 8)
                burn_in_size = burn_in_font.getsize_multiline(burn_in_text)
                draw.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
                draw.text(burn_in_location, burn_in_text, 'white', burn_in_font)

        return im    

    def map_value_to_color(self, val):

        magnification = 1. / self.dive_mesh.imagMeshGenerator.baseWidth
        if magnification <= 100:
            magnification = 100 

        denom = float(self.dive_mesh.mathSupport.justTwoLogs(magnification))

        if self.color:
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
        else:        
            #magnification = 1. / self.context.cmplx_width
            cint = int((val * 3.) / denom)
            return (cint,cint,cint)
        


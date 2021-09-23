# --
# File: mandeldistance.py
#
#
# --

import os
import pickle

import numpy as np

from PIL import Image, ImageDraw # Should be removable if palette performed the lookup

from mandelbrot_solo import MandelbrotSolo

class MandelDistance(MandelbrotSolo):
    def __init__(self, dive_mesh, frame_number, output_folder_name, extra_params={}):
        super().__init__(dive_mesh, frame_number, output_folder_name, extra_params)

        self.algorithm_name = 'mandeldistance'

    def generate_counts(self):
        """ 
        Originally thought this could use 'normal' mandelbrot, but keeping
        track of the derivative is important for the distance estimate, so
        we use the math_support's distance esitmate function instead.
        """
        #mandelbrot_function = np.vectorize(self.dive_mesh.mathSupport.mandelbrotDistanceEstimate)
        #(self.counts_array, self.last_values_array) = mandelbrot_function(self.mesh_array, self.escape_radius, self.max_escape_iterations)
        (self.counts_array, self.last_values_real_array, self.last_values_imag_array) = self.dive_mesh.mathSupport.mandelbrotDistanceEstimate(self.mesh_real_array, self.mesh_imag_array, self.escape_radius, self.max_escape_iterations)

        counts_name_base = u"%d.counts.pik" % self.frame_number
        counts_file_name = os.path.join(self.output_folder_name, counts_name_base)
        with open(counts_file_name, 'wb') as counts_handle:
            pickle.dump(self.counts_array, counts_handle)

#    def process_counts(self):
#        smoothing_function = np.vectorize(self.dive_mesh.mathSupport.rescaleForRange)
#        self.processed_array = smoothing_function(self.last_values_array, self.counts_array, self.max_escape_iterations, self.dive_mesh.realMeshGenerator.baseWidth)

# I guess with the new distance calc, we DO need to rescale the color range.
#    def pre_image_hook(self):
#        """ 
#        Don't actually want to update the palette right now for
#        calculating the distance estimate color.
#        """
#        pass


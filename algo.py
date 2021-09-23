

import os
import pickle

import fractalcache as fc
import fractalpalette as fp
import divemesh as mesh

from PIL import Image, ImageDraw, ImageFont # EscapeAlgo uses for burn-in

# Should probably import Timeline, but not technically needed, since
# it's cyclic?

class Algo(object):
    """
    Algo is responsible for generating a single, entire, image 
    frame from a specified mesh.
    
    This means the algo is responsible for generating and caching its own
    intermediate results.

    The sequence in run() is the core of the processing architecture.

    Trying something out: presuming that Algos will all write out 
    intermediate files, without extra parameters to turn that behavior
    off.  This is only really visible in this base class in mesh_setup(), 
    in which you can't turn off the mesh pickle file writing.

    In general, because this is a customizable sequence processor,
    subclasses should NOT call superclass implementations, because they're
    probably providing more specific behavior.
    """

    @staticmethod
    def options_list():
        """ 
        Algorithm implementations can fill a dictionary with key/value
        pairs to pass algorithm-specific parameters from the command line
        into the per-frame algorithm instantiation

        To do this, the list of available options are returned here, and
        then iteration over the getopts values is done in parse_options()
        """
        return []    

    @staticmethod
    def load_options_with_math_support(opts, math_support):
        return {}

    def __init__(self, dive_mesh, frame_number, output_folder_name, extra_params={}):
        self.algorithm_name = None 

        self.dive_mesh = dive_mesh
        self.frame_number = frame_number
        self.output_folder_name = output_folder_name

        # Algo should fill in these values
        self.mesh_array = None
        self.mesh_real_array = None
        self.mesh_imag_array = None
        self.counts_array = None
        self.last_values_array = None
        self.last_values_real_array = None
        self.last_values_imag_array = None
        self.processed_array = None
        self.output_image_file_name = None

    def get_metadata(self):
        return {'frame_number' : self.frame_number,
            'fractal_type': self.algorithm_name,
            'precision_type': self.dive_mesh.mathSupport.precisionType,
            'mesh_center': str(self.dive_mesh.getCenter()),
            'complex_real_width' : str(self.dive_mesh.realMeshGenerator.baseWidth),
            'complex_imag_width' : str(self.dive_mesh.imagMeshGenerator.baseWidth),
            'mesh_is_uniform' : str(self.dive_mesh.isUniform())}

    def run(self):
        """ 
        Frame generation happens in phases, with intermediate hooks available.
        """
        self.beginning_hook()
        self.mesh_setup()
        self.generate_counts()
        self.pre_process_hook()
        self.process_counts()
        self.pre_image_hook()
        self.generate_image()
        self.ending_hook()
        return self.output_image_file_name 

    def beginning_hook(self):
        pass

    def mesh_setup(self):
        """ 
        Might not be important to split mesh array creation into
        its owns step, but it does add flexibility to Algo subclasses
        to handle this differently, or maybe to not save results out
        to file.
        """
        # Originally relied on the directory structure to exist, but after
        # clearing intermediates a few times, it got annoying, so let's just
        # cross our fingers that this all ends up where we hoped it would.
        if not os.path.exists(self.output_folder_name):
            os.makedirs(output_folder_name)

        mesh_base_name = u"%d.mesh.pik" % self.frame_number
        mesh_file_name = os.path.join(self.output_folder_name, mesh_base_name)
        with open(mesh_file_name, 'wb') as mesh_handle:
            pickle.dump(self.dive_mesh, mesh_handle)

        self.mesh_real_array = self.dive_mesh.generateRealMesh()
        self.mesh_imag_array = self.dive_mesh.generateImagMesh()

    def generate_counts(self):
        """ Business end of getting results for a mesh """
        raise NotImplementedError('Subclass must implement generate_counts()') 

    def pre_process_hook(self):
        pass

    def process_counts(self):
        """ 
        The main thing process_counts() does is make sure that 
        'processed_array' is loaded with information for generate_image()
        to run with.

        This base-class implementation just points one to the other.
        """
        self.processed_array = self.counts_array

    def pre_image_hook(self):
        pass

    def generate_image(self):
        """
        Apart from actually generating an image, another important thing
        that generate_image() does is sets output_image_file_name to the
        appropriate file name that we generated.
        """
        raise NotImplementedError('Subclass must implement generate_image()') 

    def ending_hook(self):
        pass

class EscapeAlgo(Algo):

    @staticmethod
    def options_list():
        whole_list = Algo.options_list()

        whole_list.extend(["escape-radius=", 
            "max-escape-iterations=",
            "burn",
            "palette=",
        ]) 

        return whole_list

    @staticmethod
    def load_options_with_math_support(opts, math_support):
        options = Algo.load_options_with_math_support(opts, math_support)

        for opt,arg in opts:
            if opt in ['--escape-radius']:
                options['escape_radius'] = float(arg)
            elif opt in ['--max-escape-iterations']:
                options['max_escape_iterations'] = int(arg)
            elif opt in ['--burn']:
                options['burn_in'] = True
            elif opt in ['--palette']:
                options['palette'] = arg

        return options

    def __init__(self, dive_mesh, frame_number, output_folder_name, extra_params={}):
        super().__init__(dive_mesh, frame_number, output_folder_name, extra_params)

        # Load, with optional default values
        self.escape_radius = extra_params.get('escape_radius', 10.0)
        self.max_escape_iterations = int(extra_params.get('max_escape_iterations', 255))
        self.burn_in = extra_params.get('burn_in', False)
        self.palette = extra_params.get('palette', fp.FractalPalette())

    def burn_text_to_drawing(self, burn_in_text, drawing):
        burn_in_location = (10,10)
        burn_in_margin = 5 
        burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
        burn_in_size = burn_in_font.getsize_multiline(burn_in_text)
        drawing.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
        drawing.text((burn_in_location[0]-2, burn_in_location[1] - 2), burn_in_text, 'white', burn_in_font)
        
class JuliaAlgo(EscapeAlgo):

    @staticmethod
    def options_list():
        whole_list = EscapeAlgo.options_list()

        whole_list.extend(["julia-center="]) 

        return whole_list

    @staticmethod
    def load_options_with_math_support(opts, math_support):
        # Considered loading this with default values, but didn't
        # want to compete with the defaults in __init__()?
        options = EscapeAlgo.load_options_with_math_support(opts, math_support)

        for opt,arg in opts:
            if opt in ['--julia-center']:
                options['julia_center'] = math_support.createComplex(arg)

        return options

    def __init__(self, dive_mesh, frame_number, output_folder_name, extra_params={}):
        super().__init__(dive_mesh, frame_number, output_folder_name, extra_params)

        #self.julia_center = extra_params.get('julia_center', self.dive_mesh.mathSupport.createComplex(0,0)
        self.julia_center = extra_params.get('julia_center', self.dive_mesh.mathSupport.createComplex(-.8,.145))

        #print("settled on %s" % self.julia_center)



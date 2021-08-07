
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

    Frame generation happens in two phases, with intermediate hooks available.
    - beginning hook
    - generate results
    - pre-image hook
    - generate image
    - ending hook
    """

    @staticmethod
    def parse_options(opts):
        """ 
        Algorithm implementations can fill a dictionary with key/value
        pairs to pass algorithm-specific parameters from the command line
        into the per-frame algorithm instantiation
        """
        return {}

    def __init__(self, dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params={}):
        self.algorithm_name = None 

        self.dive_mesh = dive_mesh
        self.frame_number = frame_number

        self.project_folder_name = project_folder_name
        self.shared_cache_path = shared_cache_path

        self.build_cache = build_cache
        self.invalidate_cache = invalidate_cache

        self.cache_frame = None

    def build_cache_frame(self):
        raise NotImplementedError('Subclass must implement build_cache_frame()') 

    def get_frame_metadata(self):
        return {'frame_number' : self.frame_number,
            'fractal_type': self.algorithm_name,
            'precision_type': self.dive_mesh.mathSupport.precisionType,
            'mesh_center': str(self.dive_mesh.center),
            'complex_real_width' : str(self.dive_mesh.realMeshGenerator.baseWidth),
            'complex_imag_width' : str(self.dive_mesh.imagMeshGenerator.baseWidth), 
            'mesh_is_uniform' : str(self.dive_mesh.isUniform())}

    def beginning_hook(self):
        pass

    def generate_results(self):
        """
        Load or calculate the frame data.

        Probably need to move this to a hierarchical subclass
        eventually, but for now, the cache objects hold up to
        2 arrays and 2 histograms, so it can be used for all 
        our Algos.
        """
        self.cache_frame = self.build_cache_frame()

        if self.invalidate_cache == True:
            self.cache_frame.remove_from_results_cache()

        self.cache_frame.read_results_cache()

        if not self.cache_frame.frame_info.data_is_loaded():
            self.calculate_results()

            # Fresly calculated results get saved if we're building the cache
            if self.build_cache == True:
                self.cache_frame.write_results_cache()

        return

    def calculate_results(self):
        """ Business end of getting results for a mesh """
        raise NotImplementedError('Subclass must implement calculate_results()') 

    def pre_image_hook(self):
        pass

    def generate_image(self):
        pass

    def write_image_to_file(self, image):
        return self.cache_frame.write_image_to_file(image)

    def ending_hook(self):
        pass

class EscapeFrameInfo(fc.FrameInfo):
    def __init__(self, mesh_width, mesh_height, center, complex_real_width, complex_imag_width, escape_r, max_escape_iter, raw_values=None, raw_histogram=None, smooth_values=None, smooth_histogram=None):
        super().__init__(mesh_width, mesh_height, center, complex_real_width, complex_imag_width, raw_values, raw_histogram, smooth_values, smooth_histogram)

        self.escape_r           = escape_r
        self.max_escape_iter    = max_escape_iter

    def empty_copy(self):
        """ Looks like storing strings of everything makes us pickle-proof? """
        return EscapeFrameInfo(str(self.mesh_width), str(self.mesh_height), str(self.center), str(self.complex_real_width), str(self.complex_imag_width), str(self.escape_r), str(self.max_escape_iter))

    def pickle_copy(self):
        return EscapeFrameInfo(str(self.mesh_width), str(self.mesh_height), str(self.center), str(self.complex_real_width), str(self.complex_imag_width), str(self.escape_r), str(self.max_escape_iter), self.raw_values, self.raw_histogram, self.smooth_values, self.smooth_histogram)


class EscapeAlgo(Algo):

    @staticmethod
    def parse_options(opts):
        options = Algo.parse_options(opts)

        for opt,arg in opts:
            if opt in ['--escape-radius']:
                options['escape_radius'] = float(arg)
            elif opt in ['--max-escape-iterations']:
                options['max_escape_iterations'] = int(arg)
            elif opt in ['--smooth']:
                options['use_smoothing'] = True
            elif opt in ['--burn']:
                options['burn_in'] = True
            elif opt in ['--palette']:
                options['palette'] = arg

        return options

    def __init__(self, dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params={}):
        super().__init__(dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params)

        # Load, with optional default values
        self.escape_radius = extra_params.get('escape_radius', 2.0)
        self.max_escape_iterations = extra_params.get('max_escape_iterations', 255)
        self.use_smoothing = extra_params.get('use_smoothing', False)
        self.burn_in = extra_params.get('burn_in', False)
        self.palette = extra_params.get('palette', fp.FractalPalette())

    def build_cache_frame(self):
        frame_info = EscapeFrameInfo(self.dive_mesh.meshWidth, self.dive_mesh.meshHeight, self.dive_mesh.center, self.dive_mesh.realMeshGenerator.baseWidth, self.dive_mesh.imagMeshGenerator.baseWidth, self.escape_radius, self.max_escape_iterations)
        return fc.Frame(project_folder_name=self.project_folder_name, shared_cache_path=self.shared_cache_path, algorithm_name=self.algorithm_name, dive_mesh=self.dive_mesh, frame_info=frame_info, frame_number=self.frame_number)

    def get_frame_metadata(self):
        metadata = super().get_frame_metadata()
        metadata['escape_radius'] = self.escape_radius
        metadata['max_escape_iterations'] = self.max_escape_iterations
        return metadata

    def burn_text_to_drawing(self, burn_in_text, drawing):
        burn_in_location = (10,10)
        burn_in_margin = 5 
        burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
        burn_in_size = burn_in_font.getsize_multiline(burn_in_text)
        drawing.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
        drawing.text((burn_in_location[0]-2, burn_in_location[1] - 2), burn_in_text, 'white', burn_in_font)
        
class JuliaFrameInfo(EscapeFrameInfo):
    def __init__(self, mesh_width, mesh_height, center, complex_real_width, complex_imag_width, escape_r, max_escape_iter,  julia_center, raw_values=None, raw_histogram=None, smooth_values=None, smooth_histogram=None):
        super().__init__(mesh_width, mesh_height, center, complex_real_width, complex_imag_width, escape_r, max_escape_iter, raw_values, raw_histogram, smooth_values, smooth_histogram)

        self.julia_center = julia_center

    def empty_copy(self):
        """ Looks like storing strings of everything makes us pickle-proof? """
        return JuliaFrameInfo(str(self.mesh_width), str(self.mesh_height), str(self.center), str(self.complex_real_width), str(self.complex_imag_width), str(self.escape_r), str(self.max_escape_iter), str(self.julia_center))

    def pickle_copy(self):
        return JuliaFrameInfo(str(self.mesh_width), str(self.mesh_height), str(self.center), str(self.complex_real_width), str(self.complex_imag_width), str(self.escape_r), str(self.max_escape_iter), str(self.julia_center), self.raw_values, self.raw_histogram, self.smooth_values, self.smooth_histogram)

class JuliaAlgo(EscapeAlgo):

    @staticmethod
    def parse_options(opts):
        # Considered loading this with default values, but didn't
        # want to compete with the defaults in __init__()?
        options = EscapeAlgo.parse_options(opts)

        for opt,arg in opts:
            if opt in ['--julia-center']:
                options['julia_center'] = arg

        return options

    def __init__(self, dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params={}):
        super().__init__(dive_mesh, frame_number, project_folder_name, shared_cache_path, build_cache, invalidate_cache, extra_params)

        #self.julia_center = extra_params.get('julia_center', self.dive_mesh.mathSupport.createComplex(0,0)
        self.julia_center = extra_params.get('julia_center', self.dive_mesh.mathSupport.createComplex(-.8,.145))

        #print("settled on %s" % self.julia_center)

    def build_cache_frame(self):
        frame_info = JuliaFrameInfo(self.dive_mesh.meshWidth, self.dive_mesh.meshHeight, self.dive_mesh.center, self.dive_mesh.realMeshGenerator.baseWidth, self.dive_mesh.imagMeshGenerator.baseWidth, self.escape_radius, self.max_escape_iterations, self.julia_center)
        return fc.Frame(project_folder_name=self.project_folder_name, shared_cache_path=self.shared_cache_path, algorithm_name=self.algorithm_name, dive_mesh=self.dive_mesh, frame_info=frame_info, frame_number=self.frame_number)

    def get_frame_metadata(self):
        metadata = super().get_frame_metadata()
        metadata['julia_center'] = self.julia_center
        return metadata

    def beginning_hook(self):
        pass



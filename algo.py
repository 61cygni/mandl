
class Algo(object):

    def __init__(self, context):
        self.context = context
        self.algo_specific_cache = None

    def parse_options(self, opts, args):    
        pass

    def set_default_params(self):
        pass

    def setup(self):
        pass

    def animate_step(self, t):
        pass

    def pre_image_hook(self):
        pass

    def per_frame_reset(self):
        pass

    def cache_loaded(self, values):
        pass 

    def zoom_in(self, iterations=1):
        while iterations:
            self.context.cmplx_width   *= self.context.scaling_factor
            self.context.cmplx_height  *= self.context.scaling_factor
            self.context.magnification *= self.context.scaling_factor
            self.context.num_epochs += 1
            iterations -= 1

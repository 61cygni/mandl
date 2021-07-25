
class Algo(object):

    def __init__(self, context):
        self.context = context


    def parse_options(self, opts, args):    
        pass

    def setup(self):
        pass

    def animate_step(self, t):
        pass

    def pre_image_hook(self):
        pass

    def per_frame_reset(self):
        pass

    def zoom_in(self, iterations=1):
        while iterations:
            self.context.cmplx_width  *= self.context.scaling_factor
            self.context.cmplx_height *= self.context.scaling_factor
            self.context.num_epochs += 1
            iterations -= 1

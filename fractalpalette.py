# --
# File: fractalpalette.py
#
# Logical for color palettes
#
# --

import math
import numpy as  np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class FractalPalette:
    """
    Color gradient
    """

    # Color in RGB 
    def __init__(self):
        self.gradient_size = 1024
        self.palette   = []
        self.histogram = []


    def map_value_to_color(self, m, hues, smoothing=False):

        if len(self.palette) == 0: 
            c = 255 - int(255 * hues[math.floor(m)]) 
            return (c, c, c)
            
        color = None
        if smoothing:
            c1 = self.palette[1024 - int(1024 * hues[math.floor(m)])]
            c2 = self.palette[1024 - int(1024 * hues[math.ceil(m)])]
            color = fp.FractalPalette.linear_interpolate(c1,c2,.5) 
        else:
            color = self.palette[1024 - int(1024 * hues[math.floor(m)])]

        return color    
        

    def linear_interpolate(color1, color2, fraction):
        new_r = int(math.ceil((color2[0] - color1[0])*fraction) + color1[0])
        new_g = int(math.ceil((color2[1] - color1[1])*fraction) + color1[1])
        new_b = int(math.ceil((color2[2] - color1[2])*fraction) + color1[2])
        return (new_r, new_g, new_b)


    def create_gauss_gradient(self, c1, c2, mu=0.0, sigma=1.0): 
        
        print("+ Creating color gradient with gaussian decay")
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)

        self.palette.append(c2)

        x = 0.0
        while len(self.palette) <= self.gradient_size:
            g = gaussian(x,0,.10)
            c = FractalPalette.linear_interpolate(c1, c2, g) 
            self.palette.append(c)
            x = x + (1./(self.gradient_size+1))

    def create_exp_gradient(self, c1, c2, decay_const = 1.01):    

        print("+ Creating color gradient with exponential decay")
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)


        x = 0.0

        while len(self.palette) <= self.gradient_size:
            fraction = math.pow(math.e,-15.*x)
            c = FractalPalette.linear_interpolate(c1, c2, fraction)
            self.palette.append(c)
            x = x + (1. / 1025) 


    def create_exp2_gradient(self, c1, c2, decay_const = 1.01):    

        print("+ Creating color gradient with varying exponential decay")
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)


        x = 0.0
        c = c1
        # Do a very quick decent for the first 1/16 
        while len(self.palette) <= float(self.gradient_size)/32.:
            fraction = math.pow(math.e,-30.*x)
            c = FractalPalette.linear_interpolate((255,255,255), c1, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 

        last_c = c
        x = 0.0
        # Do another quick decent back to first color for the next 1/16 
        while len(self.palette) <= 2.*(float(self.gradient_size) / 16.):
            fraction = math.pow(math.e,-15.*x)
            c = FractalPalette.linear_interpolate((255,255,255), last_c, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 

        last_c = c
        x = 0.0
        # Do another quick decent back to first color for the next 1/16 
        while len(self.palette) <= 2.*(float(self.gradient_size) / 16.):
            fraction = math.pow(math.e,-5.*x)
            c = FractalPalette.linear_interpolate((255,255,255), last_c, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 

        last_c = c
        # For the remaining go back to white 
        x = 0.0
        while len(self.palette) <= self.gradient_size :
            fraction = math.pow(math.e,-2.*x)
            c = FractalPalette.linear_interpolate((255,255,255),last_c, fraction)
            self.palette.append(c)
            x = x + (1. / float(self.gradient_size)) 


    def create_normal_gradient(self, c1, c2, decay_const = 1.05):    
        
        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)

        fraction = 1.
        while len(self.palette) <= self.gradient_size:
            c = FractalPalette.linear_interpolate(c1, c2, fraction)
            self.palette.append(c)
            fraction = fraction / decay_const
            

    # Create 255 value gradient
    # Use the following trivial linear interpolation algorithm
    # (color2 - color1) * fraction + color1
    def create_gradient_from_list(self, color_list = [(255,255,255),(0,0,0),(255,255,255),(0,0,0),(255,255,255),(0,0,0),(241, 247, 215),(255,204,204),(204,204,255),(255,255,204),(255,255,255)]):

        print("+ Creating color gradient from color list")

        if len(self.palette) != 0:
            print("Error palette already created")
            sys.exit(0)
        
        #the first few colors are critical, so just fill by hand.
        self.palette.append((0,0,0))
        self.palette.append((0,0,0))
        self.palette.append(FractalPalette.linear_interpolate((0,0,0),(255,255,255),.2))
        self.palette.append(FractalPalette.linear_interpolate((0,0,0),(255,255,255),.4))
        self.palette.append(FractalPalette.linear_interpolate((0,0,0),(255,255,255),.6))
        self.palette.append(FractalPalette.linear_interpolate((0,0,0),(255,255,255),.8))

        # The magic number 6 here just denotes the previous colors we
        # filled by hand
        section_size = int(float(self.gradient_size-6)/float(len(color_list)-1))

        for c in range(0, len(color_list) - 1): 
            for i in range(0, section_size+1): 
                fraction = float(i)/float(section_size)
                new_color = FractalPalette.linear_interpolate(color_list[c], color_list[c+1], fraction)
                self.palette.append(new_color)
        while len(self.palette) < self.gradient_size:
            c = self.palette[-1]
            self.palette.append(c)
        #assert len(self.palette) == self.gradient_size    

    def make_frame(self, t):    

        IMG_WIDTH=1024
        IMG_HEIGHT=100
        
        im = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        color_iter = 0
        for x in range(0,IMG_WIDTH):
            color = self.palette[color_iter]
            for y in range(0,IMG_HEIGHT):
                draw.point([x, y], color) 
            color_iter += 1 
        

        return np.array(im)

    def __iter__(self):
        return self.palette

    def __getitem__(self, index):
             return self.palette[index]

    def display(self):
        clip = mpy.VideoClip(self.make_frame, duration=64)
        clip.preview(fps=1) #fps 1 is really all that works
## FractalPalette        

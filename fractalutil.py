# --
# File: util.py
#
# Helper functions used throughout
# --


def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def squared_modulus(z):
    return ((z.real*z.real)+(z.imag*z.imag))

# --
# Super basic linear interp method the expects colors and fraction as 
# floats
# --

def linear_interpolate_f(color1, color2, fraction):
    return ((color2 - color1)*fraction) + color1

# -- 
# smoothstep 
#
# Performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
# This is useful in cases where a threshold function with a smooth transition is
# desired.  
#
# Cribbed from : https://thebookofshaders.com/glossary/?search=smoothstep
# --

def smoothstep_f(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);

# --
# Determine whether C is in the two major bulbs. If so, we don't need to
# calculate the pixel
# --
def inside_M1_or_M2(c):
    c2 = squared_modulus(c) 

    # skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
    if  256.0*c2*c2 - 96.0*c2 + 32.0*c.real - 3.0 < 0.0: 
        return True 
    # skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
    if 16.0*(c2+2.0*c.real+1.0) - 1.0 < 0.0: 
        return True 

    return False    


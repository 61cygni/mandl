# --
# File: sharedcontext.py
# Date: Sun Dec 26 10:31:10 PST 2021 
#
# Shared context between UI and driver for creating animations
#
# --

# --
# global::RunContext
# --

class RunContext:

    def __init__(self):
        self.algo     = "" 
        self.image_w  = 0 
        self.image_h  = 0
        self.red      = 0. 
        self.green    = 0. 
        self.blue     = 0.
        self.samples  = 0
        self.max_iter = 0
        self.julia_c  = complex(0)
        self.c_width  = hpf(0)
        self.c_height = hpf(0)
        self.c_real   = hpf(0)
        self.c_imag   = hpf(0)

# --
# Class Waypoint
#
# Saved points for an animation 
# --

class Waypoint:
    def __init__(self, context, duration, image):
        self.context  = context
        self.duration = duration
        self.image    = image


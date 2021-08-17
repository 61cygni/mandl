# mandl


# Constructing A Dive

## Exploration

Establish at least two frames to dive between.
 - real and imaginary starting widths
    (real -> horiz, imag -> vert, or was it the other way?)
 - real and imaginary ending widths 
    (second width is redundant if aspect ratio is constant)
 - single center point, used as middle of both start and end frames

While exploring, write out points of interest, creating a waypoint list.
The first and last waypoints are the default endpoints for an 
editing project.

## Editing

Need at least 2 waypoints to establish the 'outer' parameters for
a dive animation.  

In terms of an epoch-based definition, these are the starting widths 
and the appropriate combination of zoom factor, framerate,
and duration, that achieve the window widths at the start and the end.

In terms of mesh exploration, these are two meshes that define the starting
and ending widths, and either a framerate or a duration.

## Statistics

For an edit, would like to have statistics gathered across the frames
that show:
 - How many pixels are which values (counts and/or hists)
 - Some kinds of entropy estimates for each frame
 - Pattern matching results, against a library of shapes

## Sync

Timing of color palette and drawing algorithm type should be flexible
enough that we can sync it to a music track, or to our defined waypoints.

Sound generations from a sequence might require timing adjustments for
the animation, to make a rhythm or a tone adjustment more consistent.
e.g. if we have a good start of a beat, but the 4th bar is messy, we could
rush or delay that phrase to make it fit more consistently.

Sound sync will probably require a similar kind of adjustment, except the
satisfaction is backwards, where we take a rhythm or a tone definition,
and bend the temporal locations that we've targeted as waypoints to
line up with the beat or tone.



# Notes on Architecture and Classes

FractalContext
 - Keeps and manages overall parameters.
 - Instantiates algorithm and precision-aware subclasses.
 - Holds the Timeline for the animation.
 - Responsible for generating images for frame numbers.


MediaView
 - Configures encoding parameters for output rendering.


DiveTimeline
 - Keeps framerate and image/mesh size.
 - Keeps a sequential list of DiveTimelineSpans, which define animation parameters through their keyframes.
 - Instantiates an appropriate Algo for every frame requested.


Algo
EscapeAlgo(Algo) and EscapeFrameInfo(FrameInfo)
JuliaAlgo(EscapeAlgo) and JuliaFrameInfo(FrameInfo)
 - Chaperone for algorithm-specific intermediates.
 - Holds both algorithm invocations, and algorithm-specific pre and post processing hooks.
 - Responsible for intermediates caching.


MathSupport
 - Implementations of library-specific calculations.
 - Interpolators and multipliers for different numeric types.
 - Native Python implementation in the base class.


# Optimization Discussion

## Inner loop multiplications optimizations at high bit depths

Text taken from:
https://randomascii.wordpress.com/2011/08/13/faster-fractals-through-algebra/

When using floating-point math the speed of addition, subtraction, and multiplication are generally identical. However floating-point math has only about 52 bits of precision which is woefully inadequate for serious fractal exploration. All of my fractal programs have featured high-precision math routines to allow virtually unlimited zooming. Fractal eXtreme supports up to 7,680 bits of precision. While this isn’t truly unlimited it is close enough because fractal calculation time generally increases as the cube of the zoom level, and even on the fastest PCs available it exceeds most people’s patience between 500 and 2,000 bits of precision.

The speed of squaring and multiplication is critical because in high-precision math they are O(n^2) operations and everything else is O(n). As ‘n’ (the number of 32-bit or 64-bit words) increases, the time to do the multiplications becomes the only thing that matters. Okay, at extremely high precisions multiplication doesn’t have to be O(n^2), but the alternatives are a lot of work for uncertain gain, so I’m ignoring them.

While multiplication and squaring are both O(n^2) it turns out that squaring can be done roughly twice as fast as multiplication. Half of the partial results when squaring are needed twice, but only need to be calculated once.

All the information necessary to find the algebraic optimization is now available.

The observation (which came from somebody I know only through e-mail) was that the Mandelbrot calculations can be done using three squaring operations rather than two squares and a multiply. At the limit this gives approximately a one-third speedup!

The one multiply was being used to calculate zr * zi. If we instead calculate (zr + zi)^2 then we get zr^2 + 2*zr*zi + zi^2. Since we’ve already calculated zr^2 and zi^2 we can just subtract them off and, voila, multiplication through squaring:


====




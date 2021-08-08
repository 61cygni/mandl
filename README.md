# mandl

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




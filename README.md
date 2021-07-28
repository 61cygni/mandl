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



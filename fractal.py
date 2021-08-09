# --
# File: mandelbrot.py
# 
# Driver file for playing around with the Mandelbrot set 
#
# Code cribbed from all over the place ... notably :
#
# https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
# http://linas.org/art-gallery/escape/escape.html
#
# Misiurewicz points also cribbed from all over 
#
# https://mrob.com/pub/muency/misiurewiczpoint.html
# https://www.youtube.com/watch?v=u1pwtSBTnPU&t=274s
#
#
# MPs:
#
# 0.4244 + 0.200759i;
#
# --

import getopt
import sys
import math
import os

import pickle

import multiprocessing # Can't actually make this work yet - gonna need pickling?

from collections import defaultdict

import numpy as  np

import moviepy.editor as mpy
from scipy.stats import norm

from moviepy.audio.tools.cuts import find_audio_period

from PIL import Image, ImageDraw, ImageFont

# -- our files
import fractalcache   as fc
import fractalpalette as fp
import fractalmath    as fm
import divemesh       as mesh

from algo import Algo # Abstract base class import, because we rely on it.
from julia import Julia 
from mandelbrot import Mandelbrot
from mandelbrot_solo import MandelbrotSolo
from mandeldistance import MandelDistance
from smooth import Smooth

MANDL_VER = "0.1"

class FractalContext:
    """
    The context for a single dive
    """

    def __init__(self, math_support=fm.DiveMathSupport()):
        self.math_support = math_support

        self.img_width  = 0 # int : Wide of Image in pixels
        self.img_height = 0 # int

        self.cmplx_width = self.math_support.createFloat(0.0)
        self.cmplx_height = self.math_support.createFloat(0.0)
        self.cmplx_center = self.math_support.createComplex(0.0) # center of image in complex plane

        self.max_iter      = 0  # int max iterations before bailing
        self.escape_rad    = 2. # float radius mod Z hits before it "escapes" 

        self.scaling_factor = 0.0 #  float amount to zoom each epoch

        self.set_zoom_level = 0   # Zoom in prior to the dive
        self.clip_start_frame = -1
        self.clip_frame_count = 1 
        self.write_video = True

        self.smoothing      = False # bool turn on color smoothing
        self.snapshot       = False # Generate a single, high res shotb

        self.julia_list   = None # Used just for timeline construction

        self.palette = None

        # Shifting from the FractalContext being the oracle of frame information, to the Timeline being the oracle.
        # Rather than keeping 'current frame' info in the context, we just keep the timeline, and
        # query it for frame-specific parameters to render with.
        self.timeline = None

        self.project_name = 'default_project'
        self.shared_cache_path = 'shared_cache'
        self.build_cache = False
        self.invalidate_cache = False

        self.algorithm_map = {'julia' : Julia, 
                'mandelbrot' : Mandelbrot,
                'mandelbrot_solo' : MandelbrotSolo,
                'mandeldistance' : MandelDistance,
                'smooth': Smooth}
        self.algorithm_name = None
        self.algorithm_extra_params = {} # Keeps command-line params for later use

        self.verbose = 0 # how much to print about progress

    def render_frame_number(self, frame_number):
        extra_params = {}
        if self.algorithm_name in self.algorithm_extra_params:
            extra_params = self.algorithm_extra_params[self.algorithm_name]

        #dive_mesh realizes more info is needed, so stashes it into its extraParams
        # Gotta retrieve that in calculateResults, right?
        # So, extra_params needs to be looked at, and algo params need to be set by the values, right?
        # Algorithm gets instantiated with all its parts in place...
        # We've gotta allow mesh's extra params to add to and override the overall algorithm's extra params?

        dive_mesh = self.timeline.getMeshForFrame(frame_number)
        #print("extra params from mesh: %s" % str(dive_mesh.extraParams))
        extra_params.update(dive_mesh.extraParams) # Allow per-frame info to overwrite algorithm info?

        display_frame_number = frame_number
        if self.clip_start_frame != -1:
            display_frame_number += self.clip_start_frame

        #print("extra params for \"%s\" instantiation: %s" % (self.algorithm_name, str(extra_params)))
        #print(".") # Just to keep the time-per-frame calculations from being overwritten in the terminal
        frame_algorithm = self.algorithm_map[self.algorithm_name](dive_mesh=dive_mesh, frame_number=display_frame_number, project_folder_name=self.project_name, shared_cache_path=self.shared_cache_path, build_cache=self.build_cache, invalidate_cache=self.invalidate_cache, extra_params=extra_params)

        frame_algorithm.beginning_hook()
        
        frame_algorithm.generate_results()
        
        frame_algorithm.pre_image_hook()

        frame_image = frame_algorithm.generate_image()

        image_file_name = frame_algorithm.write_image_to_file(frame_image)

        frame_algorithm.ending_hook()

        return image_file_name

    def __repr__(self):
        return """\
[FractalContext Img W:{w:d} Img H:{h:d} Cmplx W:{cw:s}
Cmplx H:{ch:s} Complx Center:{cc:s} Scaling:{s:f} Max iter:{mx:d}]\
""".format(
        w=self.img_width,h=self.img_height,cw=str(self.cmplx_width),ch=str(self.cmplx_height),
        cc=str(self.cmplx_center),s=self.scaling_factor,mx=self.max_iter); 

class MediaView: 
    """
    Handle displaying to gif / mp4 / screen etc.  
    """
    def make_frame(self, t):
        return np.array(self.ctx.render_frame_number(self.frame_number_from_time(t)))

    def __init__(self, duration, fps, ctx):
        self.duration  = duration
        self.fps       = fps
        self.ctx       = ctx
        self.banner    = False 
        self.vfilename = None 

    def frame_number_from_time(self, t):
        return math.floor(self.fps * t) 

    def time_from_frame_number(self, frame_number):
        return frame_number / self.fps

    def intro_banner(self):
        # Generate a text clip
        w,h = self.ctx.img_width, self.ctx.img_height
        banner_text = u"%dx%d center %s duration=%d fps=%d" %\
                       (w, h, str(self.ctx.cmplx_center), self.duration, self.fps)


        txt = mpy.TextClip(banner_text, font='Amiri-regular',
                           color='white',fontsize=12)

        txt_col = txt.on_color(size=(w + txt.w,txt.h+6),
                          color=(0,0,0), pos=(1,'center'), col_opacity=0.6)

        txt_mov = txt_col.set_position((0,h-txt_col.h)).set_duration(4)

        return mpy.CompositeVideoClip([self.clip,txt_mov]).subclip(0,self.duration)

    def create_snapshot(self):    
    
        if not self.vfilename:
            self.vfilename = "snapshot.gif"
        
        self.ctx.next_epoch(-1,self.vfilename)


    # --
    # Do any setup needed prior to running the calculatiion loop 
    # --

    def setup(self):

        print(self)
        #print(self.ctx)


#    def runTimeline(self, timeline):
#        movieClip = mpy.VideoClip(self.make_frame, 

    def run(self):

        if self.ctx.snapshot == True:
            self.create_snapshot()
            return

        self.ctx.timeline =  self.construct_simple_timeline()
        # Duration may be less than overall, if this is a sub-clip, so
        # figure out what our REAL duration is.
        frame_file_names = []
        timeline_frame_count = self.ctx.timeline.getTotalSpanFrameCount()
        for curr_frame_number in range(timeline_frame_count):
            frame_file_names.append(self.ctx.render_frame_number(curr_frame_number))

        timeline_duration = self.time_from_frame_number(timeline_frame_count)

        self.clip = mpy.ImageSequenceClip(frame_file_names, fps = self.ctx.timeline.framerate)

        if self.ctx.write_video == True:
            if self.banner:
                self.clip = self.intro_banner()
    
    #        if not self.vfilename:
    #            self.clip.preview(fps=1) #fps 1 is really all that works
            if self.vfilename.endswith(".gif"):
                #print("fps: %s" % str(self.fps))
                #self.clip.write_gif(self.vfilename, fps=self.fps)
                print("fps: %s" % str(self.ctx.timeline.framerate))
                self.clip.write_gif(self.vfilename, fps=self.ctx.timeline.framerate)
            elif self.vfilename.endswith(".mp4"):
                self.clip.write_videofile(self.vfilename,
                                      fps=self.fps, 
                                      audio=False, 
                                      codec="mpeg4")
            else:
                print("Error: file extension not supported, must be gif or mp4")
                sys.exit(0)


    def construct_simple_timeline(self):
        """
        Transitional. 

        Basically, let's construct a timeline from one set of start/end points,
        as defined by the current context.
        """
        overall_zoom_factor = self.ctx.scaling_factor
        overall_frame_count = self.duration * self.fps  
        # Integers only, but be generous and round up?
        if math.floor(overall_frame_count) != overall_frame_count:
            overall_frame_count = math.floor(overall_frame_count) + 1

        clip_start_frame = 0
        last_frame_number = overall_frame_count - 1
        if self.ctx.clip_start_frame != -1:
            # if start frame is defined, then rely on frame count
            # also being valid
            clip_start_frame = self.ctx.clip_start_frame;
            last_frame_number = clip_start_frame + self.ctx.clip_frame_count - 1  

        # So we know the overall trajectory of window zoom, and we know
        # the frames we're actually trying to render.
        rendered_frame_count = last_frame_number - clip_start_frame + 1

        print("clip_start_frame: %d rendered_frame_count: %d" % (clip_start_frame, rendered_frame_count))

        if last_frame_number > overall_frame_count:
            raise ValueError("Can't construct timeline of %d frames, starting at frame %d, for a sequence of only %d frames (%f seconds at %f fps) (%d)" % (rendered_frame_count, clip_start_frame, overall_frame_count, self.duration, self.fps, last_frame_number))

        if clip_start_frame == 0:
            start_width_real = self.ctx.cmplx_width
            start_width_imag = self.ctx.cmplx_height
        else:
            # Start frame is 1 or greater, the exponent here should be okay.
            start_width_real = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_width, overall_zoom_factor, clip_start_frame)
            start_width_imag = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_height, overall_zoom_factor, clip_start_frame)

        end_width_real = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_width, overall_zoom_factor, last_frame_number)
        end_width_imag = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_height, overall_zoom_factor, last_frame_number)

        print("Timeline ranges: {%s,%s} -> {%s,%s} in %d frames" % (str(start_width_real), str(start_width_imag), str(end_width_real), str(end_width_imag), rendered_frame_count))

        timeline = DiveTimeline(projectFolderName=self.ctx.project_name, algorithm_name=self.ctx.algorithm_name, framerate=self.fps, frameWidth=self.ctx.img_width, frameHeight=self.ctx.img_height, mathSupport=self.ctx.math_support, sharedCachePath=self.ctx.shared_cache_path)
         
        if timeline.algorithm_name == 'julia':
            # Just evenly divide the waypoints across the time for a simple timeline
            keyframeCount = len(self.ctx.julia_list)
            # 2 keyframes over 10 frames = 10 frames per keyframe
            keyframeSpacing = math.floor(rendered_frame_count / (keyframeCount - 1)) 
            if keyframeSpacing < 1:
                raise ValueError("Can't construct julia walk with more waypoints than animation frames")

            span = DiveTimelineSpan(timeline, rendered_frame_count)
            span.addNewWindowKeyframe(0, start_width_real, start_width_imag)
            span.addNewWindowKeyframe(rendered_frame_count - 1, end_width_real, end_width_imag)
            span.addNewUniformKeyframe(0)
            span.addNewUniformKeyframe(rendered_frame_count - 1)

            span.addNewCenterKeyframe(0, self.ctx.cmplx_center)
            span.addNewCenterKeyframe(rendered_frame_count - 1, self.ctx.cmplx_center)

            currKeyframeFrameNumber = 0
            for currJuliaCenter in self.ctx.julia_list:
                # Recognize when we're at the last item, and jump that keyframe to the final frame
                if currKeyframeFrameNumber + keyframeSpacing > rendered_frame_count - 1:
                    currKeyframeNumber = rendered_frame_count - 1
                
                span.addNewComplexParameterKeyframe(currKeyframeFrameNumber, 'julia_center', currJuliaCenter, transitionIn='linear', transitionOut='linear')
                currKeyframeFrameNumber += keyframeSpacing

            timeline.timelineSpans.append(span)

        else:
            #print("Trying to make span of %d frames" % frame_count)
            span = timeline.addNewSpanAtEnd(rendered_frame_count, self.ctx.cmplx_center, start_width_real, start_width_imag, end_width_real, end_width_imag)
    
            #perspectiveFrame = math.floor(frame_count * .5)
            #span.addNewTiltKeyframe(perspectiveFrame, 4.0, 1.0) 

        return timeline

    def __repr__(self):
        return """\
[MediaView duration {du:f} FPS:{f:s} Output:{vf:s}]\
""".format(du=self.duration,f=str(self.fps),vf=str(self.vfilename))


class DiveTimeline: 
    """
    Representation of an edit timeline. This maps parameters to specific frame numbers.
    A timeline also is the keeper of framerate for a frame sequence. (I think?)

    For now, the timeline also maintains the calculation cache for every frame, though
    this may eventually be the responsibility of a 'Project'

    Overview of the sequencing classes
    ----------------------------------
    DiveTimeline
    DiveTimelineSpan (Basis for setting keyframes)
    DiveSpanKeyframe
    - DiveSpanCenterKeyframe
    - DiveSpanWindowKeyframe
    - DiveSpanUniformKeyframe
    - DiveSpanTiltKeyframe
   
    There are currently only 3 'tracks' of keyframes (for complex 
    center, window base sizes, and perspective).  Keyframes 
    currently all live on integer frame numbers.
    """

    def __init__(self, projectFolderName, algorithm_name, framerate, frameWidth, frameHeight, mathSupport, sharedCachePath):
        
        self.projectFolderName = projectFolderName
        self.sharedCachePath = sharedCachePath

        self.algorithm_name = algorithm_name

        self.framerate = float(framerate)
        self.frameWidth = int(frameWidth)
        self.frameHeight = int(frameHeight)

        self.mathSupport = mathSupport

        # No definition made yet for edit gaps, so let's just enforce adjacency of ranges for now.
        self.timelineSpans = []

    def getTotalSpanFrameCount(self):
        seenFrameCount = 0
        for currSpan in self.timelineSpans:
            seenFrameCount += currSpan.frameCount
        return seenFrameCount

    def addNewSpanAtEnd(self, frameCount, center, startWidthReal, startWidthImag, endWidthReal, endWidthImag):
        """
        Constructs a new span, and adds it to the end of the existing span list

        Also adds center keyframes (that is, keyframes at the end of the span, which
        set the values for the complex center of the image), and window width keyframes 
        to the start and end of the new span.

        Apparently also adding perspective keyframes too.
        """
        span = DiveTimelineSpan(self, frameCount)
        span.addNewCenterKeyframe(0, center, 'quadratic-to', 'quadratic-to')
        span.addNewCenterKeyframe(frameCount - 1, center, 'quadratic-to', 'quadratic-to')
        span.addNewWindowKeyframe(0, startWidthReal, startWidthImag)
        span.addNewWindowKeyframe(frameCount - 1, endWidthReal, endWidthImag)

        # Setting uniform perspective this way kinda taped on as a solution, but not yet 
        # sure how to more gracefully set up perspective keyframes.
        span.addNewUniformKeyframe(0)
        span.addNewUniformKeyframe(frameCount-1)

        self.timelineSpans.append(span)

        return span

    def getSpanForFrameNumber(self, frameNumber):
        """
        Seems like I should cache or memoize this, to keep from searching for every frame,
        or at least binary search it, but I'm allergic to optimizing before profiling, 
        so 'slow' search it is for now.
        
        10 frames -> {0,9}
        3 frames -> {10,12}
        10 frames -> {13,22}
        """
        nextSpanFirstFrame = 0
        for currSpan in self.timelineSpans:
            nextSpanFirstFrame += currSpan.frameCount
            if frameNumber < nextSpanFirstFrame:
                currSpan.lastObservedStartFrame = nextSpanFirstFrame - currSpan.frameCount
                return currSpan

        return None # Went past the end without finding a valid span, so it's too high a frame number

    def getMeshForFrame(self, frameNumber):
        """
        Calculate a discretized 2d plane of complex points based on spans and keyframes in this timeline.

        # Haven't revisited this logic since building the MeshGenerator objects...
        Order of operations:
         1.) base_complex_center
         2.) distortions on base_complex_center (probably never really want this, but hey, it makes sense here)
         3.) MeshGenerator distortions?
         4.) overall distortions on the calculated 2D mesh

         Procedurally calculate the discretized 2D plane of complex points, based on all
         the known keyframes and modifiers for a given frame number...

         # complexCenter, complexWidth, imaginaryWidth 
         
        """
        # First step is to figure out which span the target frame belongs to
        targetSpan = self.getSpanForFrameNumber(frameNumber)
        if not targetSpan:
            raise IndexError("Frame number '%d' is out of range for this timeline" % frameNumber)

        # Within this span, find the closest upstream and closest downstream keyframes
        # Pretty sure we require least 2 keyframes defined for every span (start and end), so 
        # this should work out.
        localFrameNumber = frameNumber - targetSpan.lastObservedStartFrame
        (previousCenterKeyframe, nextCenterKeyframe) = targetSpan.getKeyframesClosestToFrameNumber('center', localFrameNumber) 
        #print("centers %s -> %s" % (str(previousCenterKeyframe.center), str(nextCenterKeyframe.center)))

        meshCenterValue = targetSpan.interpolateCenterValueBetweenKeyframes(localFrameNumber, previousCenterKeyframe, nextCenterKeyframe)
        #print("  interpolatedCenter: %s" % meshCenterValue)

        (previousWindowKeyframe, nextWindowKeyframe) = targetSpan.getKeyframesClosestToFrameNumber('window', localFrameNumber) 
        #print("windows %s,%s -> %s,%s" % (str(previousWindowKeyframe.realWidth), str(previousWindowKeyframe.imagWidth), str(nextWindowKeyframe.realWidth), str(nextWindowKeyframe.imagWidth)))

        (baseWidthReal, baseWidthImag) = targetSpan.interpolateWindowValuesBetweenKeyframes(localFrameNumber, previousWindowKeyframe, nextWindowKeyframe)

        #print("  interpolatedWidths: %s, %s" % (str(baseWidthReal), str(baseWidthImag)))

        (previousPerspectiveKeyframe, nextPerspectiveKeyframe) = targetSpan.getKeyframesClosestToFrameNumber('perspective', localFrameNumber)

        # More complicated with perspective keyframes, right?
        # Which mesh generator do we use - Uniform or Tilt?

        # Might have to interpolate the (widthFactor,heightFactor) of a tilt keyframe...
        # If both keyframes are uniform, then we don't actually interpolate
        # Hacky isisntance, but whatcha gonna do?
        previousIsUniform = isinstance(previousPerspectiveKeyframe, DiveSpanUniformKeyframe)
        nextIsUniform = isinstance(nextPerspectiveKeyframe, DiveSpanUniformKeyframe)
        if previousIsUniform and nextIsUniform:
            # Might feel like "baseImagWidth" is a typo (because it's distributed vertically), but
            # it's the 'imaginary width', even though we use it as the vertical element in the final mesh
            realMeshGenerator = mesh.MeshGeneratorUniform(mathSupport=self.mathSupport, varyingAxis='width', valuesCenter=meshCenterValue.real, baseWidth=baseWidthReal)
            imagMeshGenerator = mesh.MeshGeneratorUniform(mathSupport=self.mathSupport, varyingAxis='height', valuesCenter=meshCenterValue.imag, baseWidth=baseWidthImag)
        else:
            (widthTiltFactor, heightTiltFactor) = self.interpolateTiltFactorsBetweenPerspectiveKeyframes(localFrameNumber, previousPerspectiveKeyframe, nextPerspectiveKeyframe)

            # Tilt factor is the multiplier applied to the range.
            realMeshGenerator = mesh.MeshGeneratorTilt(mathSupport=self.mathSupport, varyingAxis='width', valuesCenter=meshCenterValue.real, baseWidth=baseWidthReal, tiltFactor=widthTiltFactor)
            imagMeshGenerator = mesh.MeshGeneratorTilt(mathSupport=self.mathSupport, varyingAxis='height', valuesCenter=meshCenterValue.imag, baseWidth=baseWidthImag, tiltFactor=heightTiltFactor)
 
        extraFrameParams = {}
        parameterKeyframePairs = targetSpan.getParameterKeyframePairsClosestToFrameNumber('complex', localFrameNumber)
        # parameterKeyframePairs['paramName'] -> [(previousKeyframe, nextKeyframe) , (previousKeyframe, nextKeyframe)...]
        for currParamName, currKeyframePairs in parameterKeyframePairs.items():
            for (leftKeyframe, rightKeyframe) in currKeyframePairs:
                extraFrameParams[currParamName] = targetSpan.interpolateComplexBetweenParameterKeyframes(localFrameNumber, leftKeyframe, rightKeyframe) 
            
        parameterKeyframePairs = targetSpan.getParameterKeyframePairsClosestToFrameNumber('float', localFrameNumber)
        # parameterKeyframePairs['paramName'] -> [(previousKeyframe, nextKeyframe) , (previousKeyframe, nextKeyframe)...]
        for currParamName, currKeyframePairs in parameterKeyframePairs.items():
            for (leftKeyframe, rightKeyframe) in currKeyframePairs:
                extraFrameParams[currParamName] = self.interpolateFloatBetweenParameterKeyframes(localFrameNumber, leftKeyframe, rightKeyframe) 

        diveMesh = mesh.DiveMesh(self.frameWidth, self.frameHeight, meshCenterValue, realMeshGenerator, imagMeshGenerator, self.mathSupport, extraFrameParams)
        #print (diveMesh)
        return diveMesh

    def interpolateTiltFactorsBetweenPerspectiveKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        First part of this repeats some hacky isinstance stuff.

        But this time, we know they BOTH won't be uniform, else we never would try to
        interpolate between them.
        """
        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        # Hacky isisntance, but whatcha gonna do?
        leftIsUniform = isinstance(leftKeyframe, DiveSpanUniformKeyframe)
        rightIsUniform = isinstance(rightKeyframe, DiveSpanUniformKeyframe)

        leftFrameNumber = leftKeyframe.lastObservedFrameNumber
        rightFrameNumber = rightKeyframe.lastObservedFrameNumber

        # Use 1.0 as default widthFactor and heightFactor for uniform keyframes.
        if leftIsUniform:
            transitionType = rightKeyframe.transitionIn
            leftWidthValue = 1.0
            leftHeightValue = 1.0
            rightWidthValue = rightKeyframe.widthFactor
            rightHeightValue = rightKeyframe.heightFactor 
        elif rightIsUniform:
            transitionType = leftKeyframe.transitionOut
            leftWidthValue = leftKeyframe.widthFactor
            leftHeightValue = leftKeyframe.heightFactor
            rightWidthValue = 1.0
            rightHeightValue = 1.0
        else: # both are not uniform
            # TODO: probably should enforce same transition type or crash here, but
            # I don't feel like it at the moment.
            transitionType = leftKeyframe.transitionOut
            leftWidthValue = leftKeyframe.widthFactor
            leftHeightValue = leftKeyframe.heightFactor
            rightWidthValue = rightKeyframe.widthFactor
            rightHeightValue = rightKeyframe.heightFactor

        widthTiltFactor = self.mathSupport.interpolate(transitionType, leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
        heightTiltFactor = self.mathSupport.interpolate(transitionType, leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)

        return(widthTiltFactor, heightTiltFactor)

# Maybe these belong in timeline span?
#    def getFrameNumberForTimecode(self, timecode):
#        return math.floor(
#    def getTimecodeForFrameNumber(self, frame_number):

    def __repr__(self):
        return """\
[DiveTimeline Project:{proj} framerate:{f}]\
""".format(proj=self.title,f=self.framerate)

class DiveTimelineSpan:
    """
    # ?(Can be used to observe/calculate n-1 zoom factors.)?
    """
    def __init__(self, timeline, frameCount):
        self.timeline = timeline
        self.frameCount = int(frameCount)

        # Only a single 'track' for each keyframe type to begin with, represented
        # just as a keyframe lookup for each track.  Being able to stack multiples of
        # similar-typed keyframes will probably be helpful in the long run.
        self.centerKeyframes = {}
        self.windowKeyframes = {}
        self.perspectiveKeyframes = {}

        self.complexParameterKeyframes = defaultdict(dict)
        self.floatParameterKeyframes = defaultdict(dict)
        # parameterKeyframes['julia_center'][25] = complex(0,0)
        # parameterKeyframes['julia_center'][40] = complex(0,0)

        # Currently, not allowing keyframes to exist outside of the span, even though
        # that is often helpful for defining pleasing transitions.
        # Currently, also not allowing keyframes to exist at non-frame targets, which
        # might lead to some alignment frustrations, because sub-frame calculations
        # are probably kinda important.

        self.lastObservedStartFrame = 0 # To stash frame offset when extracted from Timeline

    ####
    # TODO: All of these helper functions need to perform a modification of upstream and downsatream
    # keyframes when forcing addition, to make sure the transition types are all in order.
    # When creating a new keyframe, default is to inherit the lead-in and lead-out interpolators of the existing span.
    #
    # So, if it's:
    #  |           (lin)             |  (default all linear)
    #
    #  |  +(unspec)K(unspec)         |
    #  |    (lin)  K (lin)           | 
    #
    #  |  +(log-to)K(unspec)         |
    #  | (log-to)  K    (lin)        |
    #
    #  |  +(log-to)K(log-from)       |
    #  | (log-to)  K    (log-from)   |
    #
    # When trying to fill the range, both adjacent keyframes will agree (because it's a span property).
    # I think this means when setting a keyframe, the upstream and downstream keyframes are ALWAYS SET
    # to be consistent with the newly inserted keyframe. (either that, or create single-point-of-reference span structure?)
    #
    # BaseRangeSpan(startKey, endKey, type=lin)
    # +Keyframe(log-from) =>
    # BaseRangeSpan(startKey, keyframe, type=lin) + BaseRangeSpan(keyframe, endkey, type=log-from)
    #
    # Now, some asshole wants to define a 'speed' for a section...
    # This will drag some keyframes along, up to a point, where it can.
    # It might even just warn if reverse is observed?
    #  
    # Looks like we need special handling for when setting a keyframe on top of an existig keyframe, right?
    # If existing keyframe at the new keyframe frane number, and new keyframe is unspecified, then
    # keep the interpolators as-is.

    def addNewCenterKeyframe(self, frameNumber, centerValue, transitionIn='quadratic-to', transitionOut='quadratic-to'):
        # TODO: should probably gracefully handle stomping an existing keyframe, right?
        newKeyframe = DiveSpanCenterKeyframe(self, centerValue, transitionIn, transitionOut)
        self.centerKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def addNewWindowKeyframe(self, frameNumber, realWidth, imagWidth, transitionIn='root-to', transitionOut='root-to'):
        newKeyframe = DiveSpanWindowKeyframe(self, realWidth, imagWidth, transitionIn, transitionOut)
        self.windowKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def addNewUniformKeyframe(self, frameNumber, transitionIn='quadratic-to', transitionOut='quadratic-to'):
        newKeyframe = DiveSpanUniformKeyframe(self, transitionIn, transitionOut)
        self.perspectiveKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def addNewTiltKeyframe(self, frameNumber, widthFactor, heightFactor, transitionIn='quadratic-to-from', transitionOut='quadratic-to-from'):
        newKeyframe = DiveSpanTiltKeyframe(self, widthFactor, heightFactor, transitionIn, transitionOut)
        self.perspectiveKeyframes[frameNumber] = newKeyframe
        return newKeyframe

    def addNewComplexParameterKeyframe(self, frameNumber, name, value, transitionIn='linear', transitionOut='linear'):
        # TODO: Would like to be unable to assign keyframes with identical
        # names to both the complex and float parameter sets.
        newKeyframe = DiveSpanParameterKeyframe(self, value, transitionIn, transitionOut)
        self.complexParameterKeyframes[name][frameNumber] = newKeyframe
        # parameterKeyframes['julia_center'][40] = complex(0,0)
        return newKeyframe

    def addNewFloatParameterKeyframe(self, frameNumber, name, value, transitionIn='linear', transitionOut='linear'):
        newKeyframe = DiveSpanParameterKeyframe(self, value, transitionIn, transitionOut)
        self.floatParameterKeyframes[name][frameNumber] = newKeyframe
        # parameterKeyframes['a_value'][40] = 1.234
        return newKeyframe

    def getKeyframesClosestToFrameNumber(self, keyframeType, frameNumber):
        """
        Returns a tuple of keyframes, which are the nearest left and right keyframes for the frameNumber.
        When the frameNumber directly has a keyframe, the same keyframe is returned for both values.

        Important to remember to bless the 'lastObservedFrameNumber' into the keyframe.
        """
        typeOptions = ['center', 'window', 'perspective']
        if keyframeType not in typeOptions:
            raise ValueError("keyframeType must be one of (%s)" % ", ".join(typeOptions))

        if frameNumber >= self.frameCount:
            raise IndexError("Requested %s keyframe frame number '%d' is out of range for a span that's '%d' frames long" % (keyframeType, frameNumber, self.frameCount))

        keyframeHash = None
        if keyframeType == 'perspective':
            keyframeHash = self.perspectiveKeyframes
        elif keyframeType == 'window':
            keyframeHash = self.windowKeyframes
        else: # keyframeType == 'center':
            keyframeHash = self.centerKeyframes

        # Direct hit 
        # (really should have non-integer locations for keyframes, shouldn't we?)
        if frameNumber in keyframeHash:
            targetKeyframe = keyframeHash[frameNumber]
            targetKeyframe.lastObservedFrameNumber = frameNumber
            return (targetKeyframe, targetKeyframe)

        previousKeyframe = None
        nextKeyframe = None
        for currFrameNumber in sorted(keyframeHash.keys()):
            if currFrameNumber <= frameNumber:
                previousKeyframe = keyframeHash[currFrameNumber]
                previousKeyframe.lastObservedFrameNumber = currFrameNumber
            if currFrameNumber > frameNumber:
                nextKeyframe = keyframeHash[currFrameNumber]
                nextKeyframe.lastObservedFrameNumber = currFrameNumber
                break # Past the sorted range, so done looking

        return (previousKeyframe, nextKeyframe)

    def getParameterKeyframePairsClosestToFrameNumber(self, keyframeType, frameNumber):
        typeOptions = ['float', 'complex']
        if keyframeType not in typeOptions:
            raise ValueError("keyframeType must be one of (%s)" % ", ".join(typeOptions))

        if frameNumber >= self.frameCount:
            raise IndexError("Requested %s keyframe frame number '%d' is out of range for a span that's '%d' frames long" % (keyframeType, frameNumber, self.frameCount))

        outerKeyframeHash = None
        if keyframeType == 'complex':
            outerKeyframeHash = self.complexParameterKeyframes
        else: #keyframeType == 'float':
            outerKeyframeHash = self.floatParameterKeyframes


        answerKeyframes = defaultdict(list) 
        # answerKeyframes['paramName'] -> [(previousKeyframe, nextKeyframe) , (previousKeyframe, nextKeyframe)...]

        for currParamName, keyframeHash in outerKeyframeHash.items():
            #print("Looking at %s" % currParamName)
            # Direct hit 
            if frameNumber in keyframeHash:
                targetKeyframe = keyframeHash[frameNumber]
                targetKeyframe.lastObservedFrameNumber = frameNumber
                answerKeyframes[currParamName].append((targetKeyframe, targetKeyframe))
                continue
    
            previousKeyframe = None
            nextKeyframe = None
            for currFrameNumber in sorted(keyframeHash.keys()):
                if currFrameNumber <= frameNumber:
                    previousKeyframe = keyframeHash[currFrameNumber]
                    previousKeyframe.lastObservedFrameNumber = currFrameNumber
                if currFrameNumber > frameNumber:
                    nextKeyframe = keyframeHash[currFrameNumber]
                    nextKeyframe.lastObservedFrameNumber = currFrameNumber
                    break # Past the sorted range, so done looking
  
            if previousKeyframe != None and nextKeyframe != None:
                answerKeyframes[currParamName].append((previousKeyframe, nextKeyframe))

        #print("found: %s" % str(answerKeyframes))
        return answerKeyframes

    def interpolateCenterValueBetweenKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        Relies heavily on the stashed/cached 'lastObservedFrameNumber' value of a keyframe
        """
        #print("interpolating %s -> %s at frame %s" % (str(leftKeyframe), str(rightKeyframe), str(frameNumber)))

        # Recognize when left and right are the same, and dont' calculate anything.
        if leftKeyframe == rightKeyframe:
            return leftKeyframe.center

        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        # May want to consider 'close' rather than equal for the center equivalence check.
        if leftKeyframe.center == rightKeyframe.center:
            return leftKeyframe.center

        # Enforce that left keyframe's transitionOut should match right keyframe's transitionIn
        if leftKeyframe.transitionOut != rightKeyframe.transitionIn:
            raise ValueError("Keyframe transition types mismatched for frame number '%d'" % frameNumber)
        transitionType = leftKeyframe.transitionOut

        # Python scopes seep like this, right? Just use the value later?
        #
        # And I kept all these separate, because I'm prety sure there will be more
        # interpolation-specific parameters needed when all's said and done.
        interpolatedReal = self.timeline.mathSupport.interpolate(transitionType, leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
        interpolatedImag = self.timeline.mathSupport.interpolate(transitionType, leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)

        return self.timeline.mathSupport.createComplex(interpolatedReal, interpolatedImag)

    def interpolateWindowValuesBetweenKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        Relies heavily on the stashed/cached 'lastObservedFrameNumber' value of a keyframe
        """
        # Recognize when left and right are the same, and dont' calculate anything.
        if leftKeyframe == rightKeyframe:
            return (leftKeyframe.realWidth, leftKeyframe.imagWidth)

        # Enforce that left keyframe's transitionOut should match right keyframe's transitionIn
        if leftKeyframe.transitionOut != rightKeyframe.transitionIn:
            raise ValueError("Keyframe transition types mismatched for frame number '%d'" % frameNumber)

        transitionType = leftKeyframe.transitionOut
        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        interpolatedRealWidth = leftKeyframe.realWidth
        interpolatedImagWidth = leftKeyframe.imagWidth

        # And I kept all these separate, because I'm prety sure there will be more
        # interpolation-specific parameters needed when all's said and done.
        interpolatedRealWidth = self.timeline.mathSupport.interpolate(transitionType, leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
        interpolatedImagWidth = self.timeline.mathSupport.interpolate(transitionType, leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)

        #print("window interpolates from: (%s,%s) to: (%s,%s) as: (%s,%s)" % (leftKeyframe.realWidth, leftKeyframe.imagWidth, rightKeyframe.realWidth, rightKeyframe.imagWidth, interpolatedRealWidth, interpolatedImagWidth))

        return (interpolatedRealWidth, interpolatedImagWidth)

    def interpolateComplexBetweenParameterKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        Relies heavily on the stashed/cached 'lastObservedFrameNumber' value of a keyframe
        """
        #print("interpolating %s -> %s at frame %s" % (str(leftKeyframe), str(rightKeyframe), str(frameNumber)))

        # Recognize when left and right are the same, and dont' calculate anything.
        if leftKeyframe == rightKeyframe:
            return leftKeyframe.value

        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        # May want to consider 'close' rather than equal for the center equivalence check.
        if leftKeyframe.value == rightKeyframe.value:
            return leftKeyframe.value

        # Enforce that left keyframe's transitionOut should match right keyframe's transitionIn
        if leftKeyframe.transitionOut != rightKeyframe.transitionIn:
            raise ValueError("Keyframe transition types mismatched for frame number '%d'" % frameNumber)
        transitionType = leftKeyframe.transitionOut

        # Python scopes seep like this, right? Just use the value later?
        #
        # And I kept all these separate, because I'm prety sure there will be more
        # interpolation-specific parameters needed when all's said and done.
        interpolatedReal = self.timeline.mathSupport.interpolate(transitionType, leftKeyframe.lastObservedFrameNumber, leftKeyframe.value.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.value.real, frameNumber)
        interpolatedImag = self.timeline.mathSupport.interpolate(transitionType, leftKeyframe.lastObservedFrameNumber, leftKeyframe.value.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.value.imag, frameNumber)

        return self.timeline.mathSupport.createComplex(interpolatedReal, interpolatedImag)

    def interpolateFloatBetweenParameterKeyframes(self, frameNumber, leftKeyframe, rightKeyframe):
        """
        Relies heavily on the stashed/cached 'lastObservedFrameNumber' value of a keyframe
        """
        #print("interpolating %s -> %s at frame %s" % (str(leftKeyframe), str(rightKeyframe), str(frameNumber)))

        # Recognize when left and right are the same, and dont' calculate anything.
        if leftKeyframe == rightKeyframe:
            return leftKeyframe.value

        if frameNumber < leftKeyframe.lastObservedFrameNumber or frameNumber > rightKeyframe.lastObservedFrameNumber:
            raise IndexError("Frame number '%d' isn't between 2 keyframes at '%d' and '%d'" % (frameNumber, leftKeyframe.lastObservedFrameNumber, rightKeyframe.lastObservedFrameNumber))

        # May want to consider 'close' rather than equal for the center equivalence check.
        if leftKeyframe.value == rightKeyframe.value:
            return leftKeyframe.value

        # Enforce that left keyframe's transitionOut should match right keyframe's transitionIn
        if leftKeyframe.transitionOut != rightKeyframe.transitionIn:
            raise ValueError("Keyframe transition types mismatched for frame number '%d'" % frameNumber)
        transitionType = leftKeyframe.transitionOut

        # Python scopes seep like this, right? Just use the value later?
        #
        # And I kept all these separate, because I'm prety sure there will be more
        # interpolation-specific parameters needed when all's said and done.
        interpolatedReal = self.timeline.mathSupport.interpolate(transitionType, leftKeyframe.lastObservedKeyframeNumber, leftKeyframe.value, rightKeyframe.lastObservedFrameNumber, rightKeyframe.value, frameNumber)

        return interpolatedReal


    def __repr__(self):
        return """\
[DiveTimelineSpan {framecount} frames]\
""".format(framecount=self.frameCount)

class DiveSpanKeyframe:
    def __init__(self, span, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        self.span = span 

        transitionOptions = ['quadratic-to', 'quadratic-from', 'quadratic-to-from', 'log-to', 'root-to', 'linear']
        if transitionIn not in transitionOptions:
            raise ValueError("transitionIn must be one of (%s)" % ", ".join(transitionOptions))
        if transitionOut not in transitionOptions:
            raise ValueError("transitionOut must be one of (%s)" % ", ".join(transitionOptions))
        
        self.transitionIn = transitionIn
        self.transitionOut = transitionOut

        self.lastObservedFrameNumber = 0 # For stashing frame numbers in

    def __repr__(self):
        return """\
[DiveSpanKeyframe, {inType} -> frame {frame} -> {outType}]\
""".format(inType=self.transitionIn, frame=self.lastObservedFrameNumber, outType=self.transitionOut)

class DiveSpanCenterKeyframe(DiveSpanKeyframe):
    def __init__(self, span, center, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        super().__init__(span, transitionIn, transitionOut)
        self.center = center

class DiveSpanWindowKeyframe(DiveSpanKeyframe):
    def __init__(self, span, realWidth, imagWidth, transitionIn='root-to', transitionOut='root-to'):
        super().__init__(span, transitionIn, transitionOut)
        self.realWidth = realWidth
        self.imagWidth = imagWidth

class DiveSpanUniformKeyframe(DiveSpanKeyframe):
    def __init__(self, span, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        super().__init__(span, transitionIn, transitionOut)

class DiveSpanTiltKeyframe(DiveSpanKeyframe):
    def __init__(self, span, widthFactor, heightFactor, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        super().__init__(span, transitionIn, transitionOut)
        self.widthFactor = widthFactor
        self.heightFactor = heightFactor

class DiveSpanParameterKeyframe(DiveSpanKeyframe):
    def __init__(self, span, value, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        super().__init__(span, transitionIn, transitionOut)
        self.value = value

####
# Big bunch of 'still thinking about' comments here.  Probably should have kept them on my own branch.
###
#
# Conceptual priority of items in a timeline goes something like...
#
# Frame Numbers  
#  Frames are the base grid, so nothing can 'shift' a frame number to a higher or lower value.
#
# DiveSpan
#   Spans define the start and end base values (pre-modifications) of a dive animation
#
# DiveSpanCenterKeyframe and DiveSpanWindowKeyframe
#   Within spans, target values to hit for a frame
#
# I think that's the end of the highest-order items.  Modifications up to this point have to
# all be complete and calculated before the next items can be calculated/analyzed.
#
# For window rages, whether set as keyframes, or interpolated between, you can observe
# the effective zoom factor between any two frames, based on the transition from frame's 
# values to the next

#
# Idea behind ZoomFactorKeyframe
#
#  Attaches to the timeline based on width ranges.  Still haven't solved several
#  aspects of this half-baked idea yet.
#  Main goal is to be able to set a speed factor for a range of frames.
#
#  If a keyframe sets an axis 
#  range for a specific frame number, then the two surrounding frame-spans up to the nearest adjacent
#  range keyframes need to be stretched or squished so that the product of all the zoom factors
#  between keyframes achieves the ranges set in the adjacent range keyframes.  Or more simply,
#  if you add a range keyframe, then the surrounding zoom factors are adjusted so existing 
#  range keyframe targets aren't changed.
#  The default ramp type for zoom factor interpolation is linear.  The logic behind this is that
#  if the factor is linearly monotonically increasing, then the feel is acceleration.  So linear
#  interpolation of zoom factors results in a ramp of speed between keyframes.  I imagine that more
#  explicit curve control for zoom factor interpolation will be desirable.
#
#  [[Linear interpolation does seem to make it harder to set a constant speed though, doesn't it?]]
#  [[Does it work to add a keyframe as "set a constant zoom factor between all these frames"?]]
#

##
# A - attempt at constructing a 'move this image to this frame' kind of behavior 
##
# Take the calculated range values at frame + 4, and shift them to the target frame.
# Other stuff has to follow along... and attempting to write this is showing me what's left dangling...
#
# Quick observation:
# We'll come at this edit by looking at an existing render, and wanting a modification.
# The existing render has specific base information at a frame we're observing.
# This construction should be resilient to upstream base information modifications
# (i.e. if something 'before' is changed, this edit should still work about the same way)

# And this really has to be range-dependent, despite my trying to dodge that over and over and over.
# I keep trying to say "the base is rigidly calculated"
# And keep realizing that speed manipulations for a visible range MUST be dependent on some
# flexible attachment of the effect to a range of base values.
 
# Maybe the deal is, that range manipulations are all calculated in advance, and in the specific order of
# their declaration?  Like, there's no complicated layering of range manipulations, only
# a procedural description of all the range manipulations, which is realized into the baked-in
# base information for every frame, before axes are generated or meshes are manipulated.

# Adding set values non-monotonically, will basically play segments in reverse, right?
# start = 1.0, end = .0001, 100 frames long
# +keyframe in place at frame 10 (so frames 1-10 are equivalent to the default base ranges)
# +keyframe at frame 15, back to 1.0 (so plays start->key->start->end)
# (also, frames 15-100 are equivalent to what used to happen in frames 1-100)
#
# So now, want to set a 'speed' for 15->20
# Or better, want to set the reverse speed to faster...
# But I shouldn't be allowed to say "play a frame range faster", should I?
# Because it's dependent on the interpolated values that it has to hit.
# But I need to be able to say for this UNINTERRUPTED frame span (could verify on application?!), 
# Set my zoom factors to BLAH.
# A speed manipulation is a localized effect, that will change the (?upstream and?) downstream interpolations.
# The probem is, we maybe need to figure out what those ranges are iteratively, before allowing 
# it to 'lock' into place?

#targetFrameNumber = 2
#sampleFromFrameNumber = targetFrameNumber + 4
#
# Looks like that when we're sampling the 'current' shape for a future frame number, 
# we need to force that number into a keyframe, if it's part of an existing manipulation?
#if mainSpan.frameNumberHasParametricEffects(sampleFromFrameNumber):
#if mainSpan.frameNumberHasBaseModifiers(sampleFromFrameNumber):
#    # Creates a 'current values' keyframe, unrolls parametric changes to get to concrete values
#    mainSpan.addKeyframeInPlace(sampleFromFrameNumber) 
#
#addBaseRangeKeyframeInPlace
#
#focusRangeWidth, focusRangeHeight = mainSpan.getRangeShapeForFrameNumber(sampleFromFrameNumber)

# When we set a window keyframe, we've changed the 'effective zoom factor' of all upstream and downstream frames.
# The range of the 'effective speed change' is bounded by the next closest keyframes.

## 
# B - attempt at constructing a "speed up these 4 frames" kinda behavior
##
#
#targetFrameNumber = 3
#targetFrameCount = 4
#startingRangeWidth, startingRangeHeight = mainSpan.getRangeShapeForFrameNumber(targetFrameNumber)
#endingRangeWidth, endingRangeHeight = mainSpan.getRangeShapeForFrameNumber(targetFrameNumber + targetFrameCount - 1)

# No speed assignments that span across existing base keyframes allowed.

# Seems like this is getting close.
# The thing is, for a given range (whether forward or backward), I can't manipulate the speed in a way that
# breaks the base keyframe endpoints.
#
# Maybe that's it...
# No 'speed' assignments possible, only equivalent 'base keyframe' assignments.
# So I can't say 'play from this base range to that base range faster,
#
# Really want speed to ramp as a default, and make it so sudden jumps in speed
# are more difficult to achieve by default.

# THAT'S RIGHT  - I forgot, that speed modifiers don't hang on frame numbers, but they're applied 
# on regions within the visible ranges...
# BUT THAT WON'T WORK as a timeline thing, because if we reverse, then we don't want the 
# speed modifier to have an effect for all the matching ranges.

# Maybe it's that at a given keyframe, BOTH range targets, and a zoom factor can be applied?

## Maybe "speed change" is a function that does a bunch of math for you?
#mainSpan.addRangeSpeedKeyframe(startFrameNumber, leadingTransitionFrames=2, trailingTransitionFrames=2, startingZoomFactor * 2, 
#
#mainSpan.addRangeSpeedModifier(startFrameNumber, leadingTransitionFrames=2, trailingTransitionFrames=2, startingZoomFactor * 2)
#
#internally, it says:
#firstFrameZoomFactor = mainSpan.getZoomFactorBetweenFrameNumbers(startFrameNumber, startFrameNumber + 1)
#?rangeKeyframeAxisGenerator()?
#...generate keyframes for target ranges, based on  this... somehow...



# --
# Default settings for the dive. All of these can be overridden from the
# command line
# --
def set_demo1_params(fractal_ctx, view_ctx):
    print("+ Running in demo mode - loading default mandelbrot dive params")
    #fractal_ctx.img_width  = 1024
    #fractal_ctx.img_height = 768 
    #fractal_ctx.img_width  = 320
    #fractal_ctx.img_height = 240 

    fractal_ctx.img_width  = 160 
    fractal_ctx.img_height = 120 

    #fractal_ctx.img_width  = 80 
    #fractal_ctx.img_height = 60 
    #fractal_ctx.img_width  = 16 
    #fractal_ctx.img_height = 12 

    cmplx_width_str = '5.0'
    cmplx_height_str = '3.5'
    #cmplx_width_str = '.001'
    #cmplx_height_str = '.00075'
    #cmplx_width_str = '2.0'
    #cmplx_height_str = str(1024*2/768) 

    # For debugging, this window (and a couple zooms after) is where things
    # tend to go wrong when precision is getting clipped off
    #cmplx_width_str = '6.736977389253725e-08'
    #cmplx_height_str = '4.7158841724776074e-08'
    fractal_ctx.cmplx_width  = fractal_ctx.math_support.createFloat(cmplx_width_str)
    fractal_ctx.cmplx_height = fractal_ctx.math_support.createFloat(cmplx_height_str)


    # This is close t Misiurewicz point M32,2
    # fractal_ctx.cmplx_center = fractal_ctx.ctxc(-.77568377, .13646737)
    #center_real_str = '-1.769383179195515018213'
    #center_imag_str = '0.00423684791873677221'
    #fractal_ctx.cmplx_center = fractal_ctx.math_support.createComplex(center_real_str, center_imag_str)
    #center_str = '-1.769383179195515018213+0.00423684791873677221j'


    center_real_str = '-1.7693831791955150182138472860854737829057472636547514374655282165278881912647564588361634463895296673044858257818203031574874912384217194031282461951137475212550848062085787454772803303225167998662391124184542743017129214423639793169296754394181656831301342622793541423768572435783910849972056869527305207508191441734781061794290699753174911133714351734166117456520272756159178932042908932465102671790878414664628213755990650460738372283470777870306458882898202604001744348908388844962887074505853707095832039410323454920540534378406083202543002080240776000604510883136400112955848408048692373051275999457470473671317598770623174665886582323619043055508383245744667325990917947929662025877792679499645660786033978548337553694613673529685268652251959453874971983533677423356377699336623705491817104771909424891461757868378026419765129606526769522898056684520572284028039883286225342392455089357242793475261134567912757009627599451744942893765395578578179137375672787942139328379364197492987307203001409779081030965660422490200242892023288520510396495370720268688377880981691988243756770625044756604957687314689241825216171368155083773536285069411856763404065046728379696513318216144607821920824027797857625921782413101273331959639628043420017995090636222818019324038366814798438238540927811909247543259203596399903790614916969910733455656494065224399357601105072841234072044886928478250600986666987837467585182504661923879353345164721140166670708133939341595205900643816399988710049682525423837465035288755437535332464750001934325685009025423642056347757530380946799290663403877442547063918905505118152350633031870270153292586262005851702999524577716844595335385805548908126325397736860678083754587744508953038826602270140731059161305854135393230132058326419325267890909463907657787245924319849651660028931472549400310808097453589135197164989941931054546261747594558823583006437970585216728326439804654662779987947232731036794099604937358361568561860539962449610052967074013449293876425609214167615079422980743121960127425155223407999875999884'
    center_imag_str = '0.00423684791873677221492650717136799707668267091740375727945943565011234400080554515730243099502363650631353268335965257182300494805538736306127524814939292355930892834392050796724887904921986666045576626946900666103494014904714323725586979789908520656683202658064024115300378826789786394641622035341055102900456305723718684527210377325846307917512628774672005693326232806953822796755832517188873479124361430989485495501124096329421682827330693532171505367455526637382706988583456915684673202462211937384523487065290004627037270912806345336469007546411109669407622004367957958476890043040953462048335322273359167297049252960438077167010004209439515213189081508634843224000870136889065895088138204552309352430462782158649681507477960551795646930149740918234645225076516652086716320503880420325704104486903747569874284714830068830518642293591138468762031036739665945023607640585036218668993884533558262144356760232561099772965480869237201581493393664645179292489229735815054564819560512372223360478737722905493126886183195223860999679112529868068569066269441982065315045621648665342365985555395338571505660132833205426100878993922388367450899066133115360740011553934369094891871075717765803345451791394082587084902236263067329239601457074910340800624575627557843183429032397590197231701822237810014080715216554518295907984283453243435079846068568753674073705720148851912173075170531293303461334037951893251390031841730968751744420455098473808572196768667200405919237414872570568499964117282073597147065847005207507464373602310697663458722994227826891841411512573589860255142210602837087031792012000966856067648730369466249241454455795058209627003734747970517231654418272974375968391462696901395430614200747446035851467531667672250261488790789606038203516466311672579186528473826173569678887596534006782882871835938615860588356076208162301201143845805878804278970005959539875585918686455482194364808816650829446335905975254727342258614604501418057192598810476108766922935775177687770187001388743012888530139038318783958771247007926690'

# Shorter, for debugging
#    center_real_str = '-1.76938317919551501821384728608547378290574726365475143746552821652788819126475645883616344638952966730448582578182030315748749123842171940312824619511374752125508480620857874547728033032251679986623911241845427430171292144236397931692967543941816568313013426227935414237685724357839108499720568695273052075081914417347810617942906997531749111337143517341661174565202727561591789320429089324651026717908784146646282137559906504607383722834707778703064588828982026040017443489083888449628870745058537'
#    center_imag_str = '0.004236847918736772214926507171367997076682670917403757279459435650112344000805545157302430995023636506313532683359652571823004948055387363061275248149392923559308928343920507967248879049219866660455766269469006661034940149047143237255869797899085206566832026580640241153003788267897863946416220353410551029004563057237186845272103773258463079175126287746720056933262328069538227967558325171888734791243614309894854955011240963294216828273306935321715053674555266373827069885834569156846732024622119'


    #center_str = '-0.05+0.6805j'
    #fractal_ctx.cmplx_center = fractal_ctx.math_support.createComplex(center_str)
    fractal_ctx.cmplx_center = fractal_ctx.math_support.createComplex(center_real_str, center_imag_str)

    fractal_ctx.project_name = 'demo1'
    #fractal_ctx.scaling_factor = .8
    fractal_ctx.scaling_factor = .5

    fractal_ctx.write_video = False


    # Looks like 23.976/8 fps (almost 3fps), duration=60 (which could be 180 frames) 
    # max-iter=4096 
    # Bit depth was only ~400
    # Ended up reaching e-51
    # Most iteration answers were around 1200-1300?


    # Trying to double that... Basically, can just double duration?
    # And then, can start off at frame 175?
    #
    # Looks like 23.976/8 fps (almost 3fps), duration=120 (which could be 360 frames) 
    # Which is 45 frames per process, for 8 processes.
    # max-iter=4096 
    # Let's stack bit depth to accommodate max_iter?
    # Lower bound I'm going with is: .4438
    # So bits = 1817? Let's make it a round 1800?
    # Ended up reaching... 
    # Most iteration answers were around ?

    # Think I forgot to set bits deep enough?
    # Blew out at e-57?
    # Looks like iters were around 1500
    # Precision was (accidentally?) 400
   
    # 550 * 3.32 = 1826
    # max-iter=4096
    # Looks like 23.976/8 fps (almost 3fps), 
    # duration=90 (which could be 270 frames) 
    # lost all detail at ~227?
    # iters were hitting 4000
    # Looks like it hit e-68

    # Ok.
    # 8192 iter
    # ~3621 bits => 1125 digits
    # duration=180 (~540 frames defined, just giving myself runway)
    # Gonna run frames 200+?

    # Frame 240 is hitting e-72 widths, 4400 iters
    # Looks like window definition has ~953 digits of unused precision?
    # Lol.

    # Hit ~e-91 at frame 303
    # iters sitting around 5300


    # Shaving off 400 digits of precision... (to 725 digits = 2407 bits)
    # Also trying only 6 simultaneous, frames 300-350

    # frame 350 hit ~e-107
    # Iters around 6300

    # Shaving more precision off - 625 digits

    # Blew out around 385?
    # Hitting 7900 iters
    # Hitting e-114, so is using something like 300 positions total?

    # K, for frames 375+, gonna go with...
    # doubling iters again,
    # Adding another 100 positions? (went to 825 positions)

    # Finally timed that run
    # 50 frames (375-425)
    # 2048*8 max iter (== 16384 iters)
    # 825 digits (2739 bits)
    # real    61m38.026s
    # user    514m30.844s
    # sys     0m9.203s
    # Which is ~74 seconds per frame.
    # ~10,400 iters used
    # ~e-129 widths

    # Moving endpoint to 320 seconds? (~957 frames?)
    # Going up to 2048 * 10 (= 20,480 iters)
    # Looks like the last increment used 15 more digits... 
    # So, Maybe add 100 positions?
    # That'll be 925 digits (3071 bits)
    # real    30m22.199s
    # user    381m58.938s
    # sys     0m12.344s
    # hit e-135 at 450
    # using 11,000 iters


    # Frames 450-475 (no big changes for this segment)
    # real    38m16.362s
    # user    445m59.469s
    # sys     0m12.938s
    # hit e-143 at 475
    # using ~14,800 itersA
    # Looks like this gets to ~10:30 in the edge of infinity dive.
    # So, out of 2:29:04, that's about 6% of the dive?
    #
    # Gonna raise to 41k iterations (2048 * 20)
    # Gonna bump up to 1500 digits? (4830 bits?)
    # Trying to take a bigger bite, 475-900?

    # Attempt at batching to 599 failed the first time, 
    # BUT resulted in:
    # used 27k iter
    # e-180 widths
    # Looks like this was gonna work just fine...
    
    # 25 frames hit ~109 minutes (~4.5 minutes per frame)
    # (1308 minutes user time) = 52 minutes per frame.
    # Iters hit ~24k
   
    # real    142m25.250s
    # user    1797m10.031s
    # sys     0m11.922s


    #fractal_ctx.max_iter       = 128 
    #fractal_ctx.max_iter       = 255 
    #fractal_ctx.max_iter       = 512 # covers ~e-34 or so 
    #fractal_ctx.max_iter       = 1024 # covers ~e-48 or so
    #fractal_ctx.max_iter       = 2048 
    #fractal_ctx.max_iter       = 2048 * 16
    fractal_ctx.max_iter       = 2048 * 71 # About half way up EoI's dive
    #fractal_ctx.max_iter       = 2048 * 142 # Near the end of EoI's dive, maybe

    fractal_ctx.escape_rad     = 2
    #fractal_ctx.escape_rad     = 4 
    #fractal_ctx.escape_rad     = 8 
    #fractal_ctx.escape_rad     = 512 
    #fractal_ctx.escape_rad     = math.sqrt(1024.0)
    fractal_ctx.algorithm_extra_params[fractal_ctx.algorithm_name]['escape_radius'] = fractal_ctx.escape_rad 
    fractal_ctx.algorithm_extra_params[fractal_ctx.algorithm_name]['max_escape_iterations'] = fractal_ctx.max_iter 

    # Basically make this command-line, not demo?
    #fractal_ctx.clip_start_frame = 7 
    #fractal_ctx.clip_frame_count = 3 

    fractal_ctx.verbose = 3
    fractal_ctx.build_cache=False

    # FPS still isn't set quite right, but we'll get it there eventually.
    view_ctx.fps            = 23.976 / 8.0 
    #view_ctx.fps            = 23.976 / 4.0 
    #view_ctx.fps            = 23.976 / 2.0
    #view_ctx.fps            = 23.976
    #view_ctx.fps            = 29.97 / 2.0 

    #view_ctx.duration = math.ceil(6.0 / view_ctx.fps)
    #view_ctx.duration       = 4.0
    #view_ctx.duration       = 540.0
    view_ctx.duration       = 2200.0 # A bit more than EoI's dive, if a .5 zoom factor at 23,976/8 fps.
    #view_ctx.duration       = 1500.0
    #view_ctx.duration       = 30.0
    #view_ctx.duration       = 2.0
    #view_ctx.duration       = 0.25

def set_julia_walk_demo1_params(fractal_ctx, view_ctx):
    print("+ Running in demo mode - loading default julia walk params")
    fractal_ctx.img_width  = 1024
    fractal_ctx.img_height = 768 

    cmplx_width_str = '3.2'
    cmplx_height_str = '2.5'
    fractal_ctx.cmplx_width  = fractal_ctx.math_support.createFloat(cmplx_width_str)
    fractal_ctx.cmplx_height = fractal_ctx.math_support.createFloat(cmplx_height_str)


    #fractal_ctx.algorithm_extra_params['julia_center'] = fractal_ctx.math_support.createComplex(-.8,.145)
    fractal_ctx.algorithm_extra_params['julia']['julia_center'] = fractal_ctx.math_support.createComplex(-.8,.145)

    #fractal_ctx.algorithm_extra_params['julia_center'] = fractal_ctx.math_support.createComplex(0,0)
    fractal_ctx.julia_list = [fractal_ctx.math_support.createComplex(0.355,0.355), fractal_ctx.math_support.createComplex(0.0,0.8), fractal_ctx.math_support.createComplex(0.3355,0.355)] 

    fractal_ctx.cmplx_center = fractal_ctx.math_support.createComplex(0,0)

    fractal_ctx.project_name = 'julia_demo1'

    fractal_ctx.scaling_factor = 1.0

    fractal_ctx.max_iter       = 255

    fractal_ctx.escape_rad     = 2.
    #fractal_ctx.escape_rad     = 32768. 

    fractal_ctx.verbose = 3
    fractal_ctx.build_cache = True

    view_ctx.duration       = 4.0
    #view_ctx.duration = 0.5

    # FPS still isn't set quite right, but we'll get it there eventually.
    view_ctx.fps            = 23.976 / 8.0 
    #view_ctx.fps            = 23.976 / 2.0 
    #view_ctx.fps            = 29.97 / 2.0 

def set_preview_mode(fractal_ctx, view_ctx):
    print("+ Running in preview mode ")

    fractal_ctx.img_width  = 300
    fractal_ctx.img_height = 200

    fractal_ctx.cmplx_width  = fractal_ctx.math_support.createFloat(3.0)
    fractal_ctx.cmplx_height = fractal_ctx.math_support.createFloat(2.5)

    fractal_ctx.scaling_factor = .75

    #fractal_ctx.escape_rad     = 4.
    fractal_ctx.escape_rad     = 32768. 

    view_ctx.duration       = 4
    view_ctx.fps            = 4


def set_snapshot_mode(fractal_ctx, view_ctx, snapshot_filename='snapshot.gif'):
    print("+ Running in snapshot mode ")

    fractal_ctx.snapshot = True
    view_ctx.vfilename = snapshot_filename

    fractal_ctx.img_width  = 3000
    fractal_ctx.img_height = 2000 

    fractal_ctx.max_iter   = 2000

    fractal_ctx.cmplx_width  = fractal_ctx.math_support.createFloat(3.0)
    fractal_ctx.cmplx_height = fractal_ctx.math_support.createFloat(2.5)

    fractal_ctx.scaling_factor = .99 # set so we can zoom in more accurately

    #fractal_ctx.escape_rad     = 4.
    fractal_ctx.escape_rad     = 32768. 

    view_ctx.duration       = 0
    view_ctx.fps            = 0


def parse_options(fractal_ctx, view_ctx):
    argv = sys.argv[1:]
    
    opts, args = getopt.getopt(argv, "pd:m:s:f:w:h:c:a:",
                               ["preview",
                                "algo=",
                                "flint",
                                "flintcustom",
                                "gmp",
                                "demo",
                                "demo-julia-walk",
                                "duration=",
                                "fps=",
                                "clip-start-frame=",
                                "clip-frame-count=",
                                "project-name=",
                                "image-frame-path=",
                                "shared-cache-path=",
                                "build-cache",
                                "invalidate-cache",
                                "banner",
                                "max-iter=",
                                "img-w=",
                                "img-h=",
                                "cmplx-w=",
                                "cmplx-h=",
                                "center=",
                                "scaling-factor=",
                                "snapshot=",
                                "gif=",
                                "mpeg=",
                                "verbose=",
                                "palette-test=",
                                #"color=",
                                "color",
                                "julia-center=", # Julia
                                "julia-list=", # Julia
                                "burn", # Hopefully all algorithms?
                                "escape_radius", # Mandelbrot, Julia
                                "max_escape_iterations", # Mandelbrot, Julia
                                "smooth", # Mandelbrot, Julia
                                ])


    # First-pass parameters handled so others can be responsive
    # - Math support, so instantiations are properly typed
    # - Algorithm name, so additional parameters can be read
    for opt, arg in opts:
        if opt in ['--gmp']:
            fractal_ctx.math_support = fm.DiveMathSupportGmp()
        elif opt in ['--flint']:
            fractal_ctx.math_support = fm.DiveMathSupportFlint()
        elif opt in ['--flintcustom']:
            fractal_ctx.math_support = fm.DiveMathSupportFlintCustom()
        elif opt in ['-a', '--algo']:
            if str(arg) in fractal_ctx.algorithm_map:
                fractal_ctx.algorithm_name = str(arg)

    if fractal_ctx.algorithm_name is None:
        fractal_ctx.algorithm_name = 'mandelbrot'

    # Kinda a crazy invocation.  Loads algorithm-specific parameters into
    # a dictionary, based on that algorithm's static class parse function.
    fractal_ctx.algorithm_extra_params[fractal_ctx.algorithm_name] = fractal_ctx.algorithm_map[fractal_ctx.algorithm_name].parse_options(opts)
    # Theoretically possible we'll eventually want to run this for all
    # possible algorithm types, but for now, just loading for the 
    # 'active' algorithm.

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode(fractal_ctx, view_ctx)
        elif opt in ['-s', '--snapshot']:
            set_snapshot_mode(fractal_ctx, view_ctx, arg)
        elif opt in ['--demo']:
            set_demo1_params(fractal_ctx, view_ctx)
        elif opt in ['--demo-julia-walk']:
            set_julia_walk_demo1_params(fractal_ctx, view_ctx)

    palette = fp.FractalPalette() # Will get stashed for the algorithm to use

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration  = float(arg) 
        elif opt in ['-f', '--fps']:
            view_ctx.fps  = float(arg)
        elif opt in ['--clip-start-frame']:
            # For construct_simple_timeline, use of --clip-start-frame
            # makes --clip-frame-count kinda required
            fractal_ctx.clip_start_frame = int(arg)
        elif opt in ['--clip-frame-count']:
            fractal_ctx.clip_frame_count = int(arg)
        elif opt in ['-m', '--max-iter']:
            fractal_ctx.max_iter = int(arg)
        elif opt in ['-w', '--img-w']:
            fractal_ctx.img_width = int(arg)
        elif opt in ['-h', '--img-h']:
            fractal_ctx.img_height = int(arg)
        elif opt in ['--cmplx-w']:
            fractal_ctx.cmplx_width = fractal_ctx.math_support.createFloat(arg)
        elif opt in ['--cmplx-h']:
            fractal_ctx.cmplx_height = fractal_ctx.math_support.createFloat(arg)
        elif opt in ['-c', '--center']:
            fractal_ctx.cmplx_center= fractal_ctx.math_support.createComplex(arg)
        elif opt in ['--scaling-factor']:
            fractal_ctx.scaling_factor = float(arg)
        elif opt in ['-z', '--zoom']:
            fractal_ctx.set_zoom_level = int(arg)

        # Worth noting, julia-list is used only for Timeline construction, 
        # and isn't an intrisic thing to the Algo.
        elif opt in ['--julia-list']:
            raw_julia_list = eval(arg)  # expects a list of complex numbers
            if len(raw_julia_list) <= 1:
                print("Error: List of complex numbers for Julia walk must be at least two points")
                sys.exit(0)
            julia_list = []
            for currCenter in raw_julia_list:
                julia_list.append(fractal_ctx.math_support.create_complex(currCenter))
            fractal_ctx.julia_list = julia_list
        elif opt in ['--palette-test']:
            if str(arg) == "gauss":
                palette.create_gauss_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp":    
                palette.create_exp_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp2":    
                palette.create_exp2_gradient((0,0,0),(128,128,128))
            elif str(arg) == "list":    
                palette.create_gradient_from_list()
            else:
                print("Error: --palette-test arg must be one of gauss|exp|list")
                sys.exit(0)
            palette.display()
            sys.exit(0)
#        elif opt in ['--color']:
#            if str(arg) == "gauss":
#                palette.create_gauss_gradient((255,255,255),(0,0,0))
#            elif str(arg) == "exp":    
#                palette.create_exp_gradient((255,255,255),(0,0,0))
#            elif str(arg) == "exp2":    
#                palette.create_exp2_gradient((0,0,0),(128,128,128))
#            elif str(arg) == "list":    
#                palette.create_gradient_from_list()
#            else:
#                print("Error: --palette-test arg must be one of gauss|exp|list")
#                sys.exit(0)
#            fractal_ctx.palette = palette
        elif opt in ['--project-name']:
            fractal_ctx.project_name = arg
        elif opt in ['--shared-cache-path']:
            fractal_ctx.shared_cache_path = arg 
        elif opt in ['--build-cache']:
            fractal_ctx.build_cache = True
        elif opt in ['--invalidate-cache']:
            fractal_ctx.invalidate_cache = True
        elif opt in ['--banner']:
            view_ctx.banner = True
        elif opt in ['--verbose']:
            verbosity = int(arg)
            if verbosity not in [0,1,2,3]:
                print("Invalid verbosity level (%d) use range 0-3"%(verbosity))
                sys.exit(0)
            fractal_ctx.verbose = verbosity
        elif opt in ['--gif']:
            if view_ctx.vfilename != None:
                print("Error : Already specific media type %s"%(view_ctx.vfilename))
                sys.exit(0)
            view_ctx.vfilename = arg
        elif opt in ['--mpeg']:
            if view_ctx.vfilename != None:
                print("Error : Already specific media type %s"%(view_ctx.vfilename))
                sys.exit(0)
            view_ctx.vfilename = arg

    # Stash the palette as an extra parameter
    extra_params = fractal_ctx.algorithm_extra_params[fractal_ctx.algorithm_name]
    extra_params['palette'] = palette

if __name__ == "__main__":

    print("++ fractal.py version %s" % (MANDL_VER))

    fractal_ctx = FractalContext()
    view_ctx  = MediaView(16, 16, fractal_ctx)

    parse_options(fractal_ctx, view_ctx)
   
    view_ctx.setup()
    view_ctx.run()


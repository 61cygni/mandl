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

MANDL_VER = "0.1"

class MandlContext:
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
        self.clip_total_frames = 1 

        self.smoothing      = False # bool turn on color smoothing
        self.snapshot       = False # Generate a single, high res shotb

        self.fractal = 'mandelbrot' # or 'julia'
        self.julia_list   = None 

        self.palette = None
        self.burn_in = False

        # Shifting from the MandlContext being the oracle of frame information, to the Timeline being the oracle.
        # Rather than keeping 'current frame' info in the context, we just keep the timeline, and
        # query it for frame-specific parameters to render with.
        self.timeline = None

        self.project_name = 'default_project'
        self.shared_cache_path = 'shared_cache'
        self.build_cache = False
        self.invalidate_cache = False

        self.verbose = 0 # how much to print about progress

    def render_frame_number(self, frame_number, snapshot_filename=None):
        """
        Load or calculate the frame data.

        Once we have frame data, can perform histogram and coloring
        """
        (pixel_values_2d_raw, hist_raw, pixel_values_2d_smooth, hist_smooth, frame_metadata) = self.timeline.loadResultsForFrameNumber(frame_number, buildCache=self.build_cache, invalidateCache=self.invalidate_cache)

        # Capturing the transpose of our array, because it looks like I mixed
        # up rows and cols somewhere along the way.
        if self.smoothing == True:
            pixel_values_2d = pixel_values_2d_smooth.T
            hist = hist_smooth
        else:
            pixel_values_2d = pixel_values_2d_raw.T
            hist = hist_raw

        #print("shape of things to come: %s" % str(pixel_values_2d.shape))

        total = sum(hist.values())
        hues = []
        h = 0

        # calculate percent of total for each iteration
        for i in range(self.max_iter):
            if total :
                h += hist[i] / total
            hues.append(h)
        hues.append(h)

        # -- 
        # Create image for this frame
        # --

        im = self.draw_image_PIL(pixel_values_2d, hues, frame_metadata)

        #print("Finished iteration RErange %f:%f (re width: %f)"%(RE_START, RE_END, RE_END - RE_START))
        #print("Finished iteration IMrange %f:%f (im height: %f)"%(IM_START, IM_END, IM_END - IM_START))

        #if self.verbose > 0:
        #    print("Done]")
        
        if snapshot_filename:
            return im.save(snapshot_filename,"gif")
        else:    
            return np.array(im)

    def draw_image_PIL(self, values, hues, metadata=None):    

        im = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        
        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                m = values[x,y] 

                if not self.palette:
                    c = 255 - int(255 * hues[math.floor(m)]) 
                    color=(c, c, c)
                elif self.smoothing:
                    c1 = self.palette[1024 - int(1024 * hues[math.floor(m)])]
                    c2 = self.palette[1024 - int(1024 * hues[math.ceil(m)])]
                    color = fp.MandlPalette.linear_interpolate(c1,c2,.5) 
                else:
                    color = self.palette[1024 - int(1024 * hues[math.floor(m)])]

                # Plot the point
                draw.point([x, y], color) 

        #print("Finished iteration RErange %f:%f (re width: %f)"%(RE_START, RE_END, RE_END - RE_START))
        #print("Finished iteration IMrange %f:%f (im height: %f)"%(IM_START, IM_END, IM_END - IM_START))

        if self.burn_in == True and metadata != None:
            burn_in_text = u"%d center: %s\n    realw: %s imagw: %s" % (metadata['frame_number'], metadata['mesh_center'], metadata['complex_real_width'], metadata['complex_imag_width'])

            burn_in_location = (10,10)
            burn_in_margin = 5 
            burn_in_font = ImageFont.truetype('fonts/cour.ttf', 12)
            burn_in_size = burn_in_font.getsize_multiline(burn_in_text)
            draw.rectangle(((burn_in_location[0] - burn_in_margin, burn_in_location[1] - burn_in_margin), (burn_in_size[0] + burn_in_margin * 2, burn_in_size[1] + burn_in_margin * 2)), fill="black")
            draw.text(burn_in_location, burn_in_text, 'white', burn_in_font)

        return im    


    def __repr__(self):
        return """\
[MandlContext Img W:{w:d} Img H:{h:d} Cmplx W:{cw:s}
Cmplx H:{ch:s} Complx Center:{cc:s} Scaling:{s:f} Smoothing:{sm:b} Max iter:{mx:d}]\
""".format(
        w=self.img_width,h=self.img_height,cw=str(self.cmplx_width),ch=str(self.cmplx_height),
        cc=str(self.cmplx_center),s=self.scaling_factor,mx=self.max_iter,sm=self.smoothing); 

class MediaView: 
    """
    Handle displaying to gif / mp4 / screen etc.  
    """
    def make_frame(self, t):
        return self.ctx.render_frame_number(self.frame_number_from_time(t))

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
        print(self.ctx)


    def run(self):

        if self.ctx.snapshot == True:
            self.create_snapshot()
            return

        self.ctx.timeline =  self.construct_simple_timeline()
        # Duration may be less than overall, if this is a sub-clip, so
        # figure out what our REAL duration is.
        timeline_frame_count = self.ctx.timeline.getTotalSpanFrameCount()
        timeline_duration = self.time_from_frame_number(timeline_frame_count)

        self.clip = mpy.VideoClip(self.make_frame, duration=timeline_duration)

        if self.banner:
            self.clip = self.intro_banner()

        if not self.vfilename:
            self.clip.preview(fps=1) #fps 1 is really all that works
        elif self.vfilename.endswith(".gif"):
            self.clip.write_gif(self.vfilename, fps=self.fps)
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
        overall_frame_count = self.duration * self.fps  
        if math.floor(overall_frame_count) != overall_frame_count:
            overall_frame_count = math.floor(overall_frame_count) + 1

        clip_start_frame = 0
        rendered_frame_count = overall_frame_count
        last_frame_number = overall_frame_count

        overall_zoom_factor = self.ctx.scaling_factor
    
        if self.ctx.clip_start_frame == -1: # Whole clip is the span 
            start_width_real = self.ctx.cmplx_width
            start_width_imag = self.ctx.cmplx_height
        else:
            if self.ctx.clip_start_frame + self.ctx.clip_total_frames > overall_frame_count:
                raise ValueError("Can't construct timeline of %d frames for a sequence of only %d frames (%f seconds at %f fps)" % (self.ctx.clip_total_frames, overall_frame_count, self.duration, self.fps))

            clip_start_frame = self.ctx.clip_start_frame
            last_frame_number = clip_start_frame + self.ctx.clip_total_frames - 1
            rendered_frame_count = last_frame_number - clip_start_frame + 1

            # Start frame is 1 or greater (hopefully), so subtracting 1 from the exponent here should be okay.
            start_width_real = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_width, overall_zoom_factor, clip_start_frame)
            start_width_imag = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_height, overall_zoom_factor, clip_start_frame)

        # Sub-section the frames, if needed
        end_width_real = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_width, overall_zoom_factor, last_frame_number)
        end_width_imag = self.ctx.math_support.scaleValueByFactorForIterations(self.ctx.cmplx_height, overall_zoom_factor, last_frame_number)

        print("{%s,%s} -> {%s,%s} in %d frames" % (str(start_width_real), str(start_width_imag), str(end_width_real), str(end_width_imag), rendered_frame_count))

        if self.ctx.fractal == 'julia':
            timeline = DiveTimeline(projectFolderName=self.ctx.project_name, fractal='julia', framerate=self.fps, frameWidth=self.ctx.img_width, frameHeight=self.ctx.img_height, mathSupport=self.ctx.math_support, sharedCachePath=self.ctx.shared_cache_path)
            # Just evenly divide the waypoints across the time for a simple timeline
            keyframeCount = len(self.ctx.julia_list)
            # 2 keyframes over 10 frames = 10 frames per keyframe
            keyframeSpacing = math.floor(rendered_frame_count / (keyframeCount - 1)) 
            if keyframeSpacing < 1:
                raise ValueError("Can't construct julia walk with more waypoints than animation frames")

            span = DiveTimelineSpan(timeline, rendered_frame_count, self.ctx.escape_rad, self.ctx.max_iter, self.ctx.smoothing)
            span.addNewWindowKeyframe(0, start_width_real, start_width_imag)
            span.addNewWindowKeyframe(rendered_frame_count - 1, end_width_real, end_width_imag)
            span.addNewUniformKeyframe(0)
            span.addNewUniformKeyframe(rendered_frame_count - 1)

            currKeyframeFrameNumber = 0
            for currJuliaCenter in self.ctx.julia_list:
                # Recognize when we're at the last item, and jump that keyframe to the final frame
                if currKeyframeFrameNumber + keyframeSpacing > rendered_frame_count - 1:
                    currKeyframeNumber = frame_count - 1
                
                span.addNewCenterKeyframe(currKeyframeFrameNumber, currJuliaCenter, transitionIn='linear', transitionOut='linear')
                currKeyframeFrameNumber += keyframeSpacing

            timeline.timelineSpans.append(span)

        else:
            timeline = DiveTimeline(projectFolderName=self.ctx.project_name, fractal='mandelbrot', framerate=self.fps, frameWidth=self.ctx.img_width, frameHeight=self.ctx.img_height, mathSupport=self.ctx.math_support, sharedCachePath=self.ctx.shared_cache_path)
            # Here, I have ctx, which should know escapeRadius, maxIter, and shouldSmooth... 
            #print("Trying to make span of %d frames" % frame_count)
            span = timeline.addNewSpanAtEnd(rendered_frame_count, self.ctx.cmplx_center, start_width_real, start_width_imag, end_width_real, end_width_imag, self.ctx.escape_rad, self.ctx.max_iter, self.ctx.smoothing)
    
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

    def __init__(self, projectFolderName, fractal, framerate, frameWidth, frameHeight, mathSupport, sharedCachePath):
        
        self.projectFolderName = projectFolderName
        self.sharedCachePath = sharedCachePath

        fractalOptions = ['mandelbrot', 'julia']
        if fractal not in fractalOptions:
            raise ValueError("fractal must be one of (%s)" % ", ".join(fractalOptions))
        self.fractalType = fractal

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

    def addNewSpanAtEnd(self, frameCount, center, startWidthReal, startWidthImag, endWidthReal, endWidthImag, escapeRadius, maxEscapeIterations, shouldSmooth):
        """
        Constructs a new span, and adds it to the end of the existing span list

        Also adds center keyframes (that is, keyframes at the end of the span, which
        set the values for the complex center of the image), and window width keyframes 
        to the start and end of the new span.

        Apparently also adding perspective keyframes too.
        """
        span = DiveTimelineSpan(self, frameCount, escapeRadius, maxEscapeIterations, shouldSmooth)
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

    def loadResultsForFrameNumber(self, frameNumber, buildCache=True, invalidateCache=False):
        """ Multi-cache-aware data loading or calculating """
        diveMesh = self.getMeshForFrame(frameNumber)
        cacheFrame = fc.Frame(self, diveMesh, frameNumber)

        if invalidateCache == True:
            cacheFrame.remove_from_results_cache()

        cacheFrame.read_results_cache()

        if cacheFrame.frame_info.raw_values is None or cacheFrame.frame_info.raw_histogram is None or cacheFrame.frame_info.smooth_values is None or cacheFrame.frame_info.smooth_histogram is None:
            #print("+  calculating epoch results")
            (rawValues, rawHistogram, smoothValues, smoothHistogram) = self.calculateResultsForDiveMesh(diveMesh)
            cacheFrame.frame_info.raw_values = rawValues
            cacheFrame.frame_info.raw_histogram = rawHistogram
            cacheFrame.frame_info.smooth_values = smoothValues
            cacheFrame.frame_info.smooth_histogram = smoothHistogram

            # Fresly calculated results get saved if we're building the cache
            if buildCache == True:
                cacheFrame.write_results_cache()

        frame_metadata = {'frame_number' : frameNumber,
            'fractal_type': self.fractalType,
            'precision_type': self.mathSupport.precisionType,
            'mesh_center': str(diveMesh.center),
            'complex_real_width' : str(diveMesh.realMeshGenerator.baseWidth),
            'complex_imag_width' : str(diveMesh.imagMeshGenerator.baseWidth), 
            'escape_radius' : str(diveMesh.escapeRadius),
            'mesh_is_uniform' : str(diveMesh.isUniform()),
            'max_escape_iterations' : str(diveMesh.maxEscapeIterations)}

        return (cacheFrame.frame_info.raw_values, cacheFrame.frame_info.raw_histogram, cacheFrame.frame_info.smooth_values, cacheFrame.frame_info.smooth_histogram, frame_metadata)

    def calculateResultsForDiveMesh(self, diveMesh):
        mesh = diveMesh.generateMesh()

        if self.fractalType == 'julia':
            juliaFunction = np.vectorize(self.mathSupport.julia)
            (pixel_values_2d, lastZees) = juliaFunction(diveMesh.center, mesh, diveMesh.escapeRadius, diveMesh.maxEscapeIterations)
        else: # self.FractalType == 'mandelbrot'
            mandelbrotFunction = np.vectorize(self.mathSupport.mandelbrot)
            (pixel_values_2d, lastZees) = mandelbrotFunction(mesh, diveMesh.escapeRadius, diveMesh.maxEscapeIterations)

        smoothingFunction = np.vectorize(self.mathSupport.smoothAfterCalculation)
        pixel_values_2d_smoothed = smoothingFunction(lastZees, pixel_values_2d, diveMesh.maxEscapeIterations)

        hist = defaultdict(int) 
        hist_smoothed = defaultdict(int) 

        show_row_progress = False
        for x in range(0, mesh.shape[0]):
            for y in range(0, mesh.shape[1]):
                # Not using mathSupport's floor() here, because it should just be a normal-scale float
                if pixel_values_2d[x,y] < diveMesh.maxEscapeIterations:
                    #print("x: %d, y: %d, val: %s, floor: %s" % (x,y,str(pixel_values_2d[x,y]), str(self.mathSupport.floor(pixel_values_2d[x,y]))))
                    hist[math.floor(pixel_values_2d[x,y])] += 1
                if pixel_values_2d_smoothed[x,y] < diveMesh.maxEscapeIterations:
                    hist_smoothed[math.floor(pixel_values_2d_smoothed[x,y])] += 1


#        pixel_values_2d = np.zeros((mesh.shape[0], mesh.shape[1]), dtype=np.uint32)
#        pixel_values_2d_smoothed = np.zeros((mesh.shape[0], mesh.shape[1]), dtype=np.float)
#        hist = defaultdict(int) 
#        hist_smoothed = defaultdict(int) 
#
#        show_row_progress = True
#        for x in range(0, mesh.shape[0]):
#            for y in range(0, mesh.shape[1]):
#                if self.fractalType == 'julia':
#                    (pixel_values_2d[x,y], lastZee) = self.mathSupport.julia(diveMesh.center, mesh[x,y], diveMesh.escapeRadius, diveMesh.maxEscapeIterations)
#                else: # self.FractalType == 'mandelbrot'
#                    (pixel_values_2d[x,y], lastZee) = self.mathSupport.mandelbrot(mesh[x,y], diveMesh.escapeRadius, diveMesh.maxEscapeIterations)
#
#                pixel_values_2d_smoothed[x,y] = self.mathSupport.smoothAfterCalculation(lastZee, pixel_values_2d[x,y], diveMesh.maxEscapeIterations)
#
#                # Not using mathSupport's floor() here, because it should just be a normal-scale float
#                if pixel_values_2d[x,y] < diveMesh.maxEscapeIterations:
#                    #print("x: %d, y: %d, val: %s, floor: %s" % (x,y,str(pixel_values_2d[x,y]), str(self.mathSupport.floor(pixel_values_2d[x,y]))))
#                    hist[math.floor(pixel_values_2d[x,y])] += 1
#                if pixel_values_2d_smoothed[x,y] < diveMesh.maxEscapeIterations:
#                    hist_smoothed[math.floor(pixel_values_2d_smoothed[x,y])] += 1
#
#            if show_row_progress == True:
#                print("%d-" % x, end="")
#                sys.stdout.flush()

        return (pixel_values_2d, hist, pixel_values_2d_smoothed, hist_smoothed)

        ####
        # Graveyard of failed attempts at further vectorizing this, maybe there's a clue in here
        # somewhere...
        ####

        # In search of efficient ways to apply the map, and getting stuck with various issues
        # like pickling, which are keeping me from using multiprocessing.Pool

# Seemed to go exponential run time for some bizarre reason
#            # Probably not necessary, but lining up the 2-element subarray
#            pixel_inputs_1d = pixel_values_2d.reshape((mesh.shape[0] * mesh.shape[1]))
#
#            # Pretty sure this is mistakenly doing an n! pass, or something just as ridiculous.
#            #print("shape of pixel_inputs_1d: %s" % str(pixel_inputs_1d.shape))
#            pixel_values_1d = np.array([self.mathSupport.mandelbrot(complex_value, diveMesh.escapeRadius, diveMesh.maxEscapeIterations, diveMesh.shouldSmooth) for complex_value in pixel_inputs_1d])
#    
#            pixel_values_2d = pixel_values_1d.reshape((mesh.shape[0], mesh.shape[1]))
#            #pixel_values_2d = np.squeeze(pixel_values_2d, axis=2) # Incantation to remove a sub-array level
#            #print("shape of pixel_values_2d: %s" % str(pixel_values_2d.shape))

        #nope
        #theFunction = np.vectorize(self.mandelbrot_flint)
        #pixel_values_1d = theFunction(pixel_inputs_1d)

        #pixel_inputs_1d = pixel_inputs.reshape(1,self.img_width * self.img_height)

        #pixel_values_1d = map(self.mandelbrot_flint, pixel_inputs_1d)
        #pixel_values_1d = np.array(list(map(self.mandelbrot_flint, pixel_inputs_1d)))
        #print("shape of pixel_values_1d: %s" % str(pixel_values_1d.shape))

        # Can't pikcle... hmm
        #mandelpool = multiprocessing.Pool(processes = 1)
        #pixel_values_1d = mandelpool.map(self.mandelbrot_flint, pixel_inputs_1d)
        #mandelpool.close()
        #mandelpool.join()

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
   
        # Passing lots into the dive mesh.  Notably, some info about the DiveTimelineSpan that was
        # responsible for creating this mesh.  Might want to store an actual reference to the object, but
        # doesn't seem needed yet?
        diveMesh = mesh.DiveMesh(self.frameWidth, self.frameHeight, meshCenterValue, realMeshGenerator, imagMeshGenerator, self.mathSupport, targetSpan.escapeRadius, targetSpan.maxEscapeIterations, targetSpan.shouldSmooth)
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

        if transitionType == 'log-to':
            widthTiltFactor = self.mathSupport.interpolateLogTo(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateLogTo(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'root-to':
            widthTiltFactor = self.mathSupport.interpolateRootTo(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateRootTo(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'linear':
            widthTiltFactor = self.mathSupport.interpolateLinearTo(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateLinearTo(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'quadratic-to':
            widthTiltFactor = self.mathSupport.interpolateQuadraticEaseOut(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateQuadraticEaseOut(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        elif transitionType == 'quadratic-from':
            widthTiltFactor = self.mathSupport.interpolateQuadraticEaseIn(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateQuadraticEaseIn(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)
        else: # transitionType == 'quadratic-to-from'
            widthTiltFactor = self.mathSupport.interpolateQuadraticEaseInOut(leftFrameNumber, leftWidthValue, rightFrameNumber, rightWidthValue, frameNumber)
            heightTiltFactor = self.mathSupport.interpolateQuadraticEaseInOut(leftFrameNumber, leftHeightValue, rightFrameNumber, rightHeightValue, frameNumber)

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
    def __init__(self, timeline, frameCount, escapeRadius, maxEscapeIterations, shouldSmooth):
        self.timeline = timeline
        self.frameCount = int(frameCount)

        self.escapeRadius = escapeRadius
        self.maxEscapeIterations = maxEscapeIterations
        self.shouldSmooth = shouldSmooth

        # Only a single 'track' for each keyframe type to begin with, represented
        # just as a keyframe lookup for each track.  Being able to stack multiples of
        # similar-typed keyframes will probably be helpful in the long run.
        self.centerKeyframes = {}
        self.windowKeyframes = {}
        self.perspectiveKeyframes = {}
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
        if transitionType == 'log-to':
            interpolatedReal = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'root-to':
            interpolatedReal = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'linear':
            interpolatedReal = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'quadratic-to':
            interpolatedReal = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        elif transitionType == 'quadratic-from':
            interpolatedReal = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
        else: # transitionType == 'quadratic-to-from':
            interpolatedReal = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.real, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.real, frameNumber)
            interpolatedImag = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.center.imag, rightKeyframe.lastObservedFrameNumber, rightKeyframe.center.imag, frameNumber)
            
        interpolatedCenter = self.timeline.mathSupport.createComplex(interpolatedReal, interpolatedImag)
        return interpolatedCenter

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
        if transitionType == 'log-to':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateLogTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'root-to':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateRootTo(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'linear':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateLinear(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'quadratic-to':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateQuadraticEaseOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        elif transitionType == 'quadratic-from':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateQuadraticEaseIn(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)
        else: # transitionType == 'quadratic-to-from':
            interpolatedRealWidth = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.realWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.realWidth, frameNumber)
            interpolatedImagWidth = self.timeline.mathSupport.interpolateQuadraticEaseInOut(leftKeyframe.lastObservedFrameNumber, leftKeyframe.imagWidth, rightKeyframe.lastObservedFrameNumber, rightKeyframe.imagWidth, frameNumber)

        return (interpolatedRealWidth, interpolatedImagWidth)

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
def set_demo1_params(mandl_ctx, view_ctx):
    print("+ Running in demo mode - loading default mandelbrot dive params")
    mandl_ctx.img_width  = 1024
    mandl_ctx.img_height = 768 

    cmplx_width_str = '5.0'
    cmplx_height_str = '3.5'
    mandl_ctx.cmplx_width  = mandl_ctx.math_support.createFloat(cmplx_width_str)
    mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(cmplx_height_str)

    # This is close t Misiurewicz point M32,2
    # mandl_ctx.cmplx_center = mandl_ctx.ctxc(-.77568377, .13646737)
    #center_real_str = '-1.769383179195515018213'
    #center_imag_str = '0.00423684791873677221'
    #mandl_ctx.cmplx_center = mandl_ctx.math_support.createComplex(center_real_str, center_imag_str)
    center_str = '-1.769383179195515018213+0.00423684791873677221j'
    mandl_ctx.cmplx_center = mandl_ctx.math_support.createComplex(center_str)

    mandl_ctx.project_name = 'demo1'

    mandl_ctx.scaling_factor = .90

    mandl_ctx.max_iter       = 255

    mandl_ctx.escape_rad     = 2.
    #mandl_ctx.escape_rad     = 32768. 

    mandl_ctx.verbose = 3
    mandl_ctx.burn_in = True
    mandl_ctx.build_cache=True

    view_ctx.duration       = 2.0
    #view_ctx.duration       = 0.25

    # FPS still isn't set quite right, but we'll get it there eventually.
    view_ctx.fps            = 23.976 / 2.0 
    #view_ctx.fps            = 29.97 / 2.0 

def set_julia_walk_demo1_params(mandl_ctx, view_ctx):
    print("+ Running in demo mode - loading default julia walk params")
    mandl_ctx.img_width  = 1024
    mandl_ctx.img_height = 768 

    cmplx_width_str = '3.2'
    cmplx_height_str = '2.5'
    mandl_ctx.cmplx_width  = mandl_ctx.math_support.createFloat(cmplx_width_str)
    mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(cmplx_height_str)

    mandl_ctx.fractal = 'julia'
    mandl_ctx.julia_list = [mandl_ctx.math_support.createComplex(0.355,0.355), mandl_ctx.math_support.createComplex(0.0,0.8), mandl_ctx.math_support.createComplex(0.3355,0.355)] 

    mandl_ctx.cmplx_center = mandl_ctx.math_support.createComplex(0,0)

    mandl_ctx.project_name = 'julia_demo1'

    mandl_ctx.scaling_factor = 1.0

    mandl_ctx.max_iter       = 255

    mandl_ctx.escape_rad     = 2.
    #mandl_ctx.escape_rad     = 32768. 

    mandl_ctx.verbose = 3
    mandl_ctx.burn_in = True
    mandl_ctx.build_cache = True

    view_ctx.duration       = 2.0

    # FPS still isn't set quite right, but we'll get it there eventually.
    view_ctx.fps            = 23.976 / 2.0 
    #view_ctx.fps            = 29.97 / 2.0 

def set_preview_mode(mandl_ctx, view_ctx):
    print("+ Running in preview mode ")

    mandl_ctx.img_width  = 300
    mandl_ctx.img_height = 200

    mandl_ctx.cmplx_width  = mandl_ctx.math_support.createFloat(3.0)
    mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(2.5)

    mandl_ctx.scaling_factor = .75

    #mandl_ctx.escape_rad     = 4.
    mandl_ctx.escape_rad     = 32768. 

    view_ctx.duration       = 4
    view_ctx.fps            = 4


def set_snapshot_mode(mandl_ctx, view_ctx, snapshot_filename='snapshot.gif'):
    print("+ Running in snapshot mode ")

    mandl_ctx.snapshot = True
    view_ctx.vfilename = snapshot_filename

    mandl_ctx.img_width  = 3000
    mandl_ctx.img_height = 2000 

    mandl_ctx.max_iter   = 2000

    mandl_ctx.cmplx_width  = mandl_ctx.math_support.createFloat(3.0)
    mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(2.5)

    mandl_ctx.scaling_factor = .99 # set so we can zoom in more accurately

    #mandl_ctx.escape_rad     = 4.
    mandl_ctx.escape_rad     = 32768. 

    view_ctx.duration       = 0
    view_ctx.fps            = 0


def parse_options(mandl_ctx, view_ctx):
    argv = sys.argv[1:]
    
    opts, args = getopt.getopt(argv, "pd:m:s:f:z:w:h:c:",
                               ["preview",
                                "demo",
                                "demo-julia-walk",
                                "duration=",
                                "fps=",
                                "clip-start-frame=",
                                "clip-total-frames=",
                                "max-iter=",
                                "img-w=",
                                "img-h=",
                                "cmplx-w=",
                                "cmplx-h=",
                                "center=",
                                "scaling-factor=",
                                "snapshot=",
                                "zoom=",
                                "gif=",
                                "mpeg=",
                                "verbose=",
                                "julia-walk=",
                                "center=",
                                "palette-test=",
                                "color=",
                                "burn",
                                "flint",
                                "gmp",
                                "project-name=",
                                "shared-cache-path=",
                                "build-cache",
                                "invalidate-cache",
                                "banner",
                                "smooth"])

    # Math support as to be handled first, so other parameter 
    # instantiations are properly typed
    for opt, arg in opts:
        if opt in ['--gmp']:
            mandl_ctx.math_support = fm.DiveMathSupportGmp()
        elif opt in ['--flint']:
            mandl_ctx.math_support = fm.DiveMathSupportFlint()

    for opt,arg in opts:
        if opt in ['-p', '--preview']:
            set_preview_mode(mandl_ctx, view_ctx)
        elif opt in ['-s', '--snapshot']:
            set_snapshot_mode(mandl_ctx, view_ctx, arg)
        elif opt in ['--demo']:
            set_demo1_params(mandl_ctx, view_ctx)
        elif opt in ['--demo-julia-walk']:
            set_julia_walk_demo1_params(mandl_ctx, view_ctx)

    for opt, arg in opts:
        if opt in ['-d', '--duration']:
            view_ctx.duration  = float(arg) 
        elif opt in ['-f', '--fps']:
            view_ctx.fps  = float(arg)
        elif opt in ['--clip-start-frame']:
            mandl_ctx.clip_start_frame = int(arg)
        elif opt in ['--clip-total-frames']:
            mandl_ctx.clip_total_frames = int(arg)
        elif opt in ['-m', '--max-iter']:
            mandl_ctx.max_iter = int(arg)
        elif opt in ['-w', '--img-w']:
            mandl_ctx.img_width = int(arg)
        elif opt in ['-h', '--img-h']:
            mandl_ctx.img_height = int(arg)
        elif opt in ['--cmplx-w']:
            mandl_ctx.cmplx_width = mandl_ctx.math_support.createFloat(arg)
        elif opt in ['--cmplx-h']:
            mandl_ctx.cmplx_height = mandl_ctx.math_support.createFloat(arg)
        elif opt in ['-c', '--center']:
            mandl_ctx.cmplx_center= mandl_ctx.math_support.createComplex(arg)
        elif opt in ['--scaling-factor']:
            mandl_ctx.scaling_factor = float(arg)
        elif opt in ['-z', '--zoom']:
            mandl_ctx.set_zoom_level = int(arg)
        elif opt in ['--smooth']:
            mandl_ctx.smoothing = True 
        elif opt in ['--julia-list']:
            mandl_ctx.fractal = 'julia'
            raw_julia_list = eval(arg)  # expects a list of complex numbers
            if len(raw_julia_list) <= 1:
                print("Error: List of complex numbers for Julia walk must be at least two points")
                sys.exit(0)
            julia_list = []
            for currCenter in raw_julia_list:
                julia_list.append(mandl_ctx.math_support.create_complex(currCenter))
            mandl_ctx.julia_list = julia_list
        elif opt in ['--palette-test']:
            m = fp.MandlPalette()
            if str(arg) == "gauss":
                m.create_gauss_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp":    
                m.create_exp_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp2":    
                m.create_exp2_gradient((0,0,0),(128,128,128))
            elif str(arg) == "list":    
                m.create_gradient_from_list()
            else:
                print("Error: --palette-test arg must be one of gauss|exp|list")
                sys.exit(0)
            m.display()
            sys.exit(0)
        elif opt in ['--color']:
            m = fp.MandlPalette()
            if str(arg) == "gauss":
                m.create_gauss_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp":    
                m.create_exp_gradient((255,255,255),(0,0,0))
            elif str(arg) == "exp2":    
                m.create_exp2_gradient((0,0,0),(128,128,128))
            elif str(arg) == "list":    
                m.create_gradient_from_list()
            else:
                print("Error: --palette-test arg must be one of gauss|exp|list")
                sys.exit(0)
            mandl_ctx.palette = m
        elif opt in ['--burn']:
            mandl_ctx.burn_in = True
        elif opt in ['--project-name']:
            mandl_ctx.project_name = arg
        elif opt in ['--shared-cache-path']:
            mandl_ctx.shared_cache_path = arg 
        elif opt in ['--build-cache']:
            mandl_ctx.build_cache = True
        elif opt in ['--invalidate-cache']:
            mandl_ctx.invalidate_cache = True
        elif opt in ['--banner']:
            view_ctx.banner = True
        elif opt in ['--verbose']:
            verbosity = int(arg)
            if verbosity not in [0,1,2,3]:
                print("Invalid verbosity level (%d) use range 0-3"%(verbosity))
                sys.exit(0)
            mandl_ctx.verbose = verbosity
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


if __name__ == "__main__":

    print("++ mandlebort.py version %s" % (MANDL_VER))

    mandl_ctx = MandlContext()
    view_ctx  = MediaView(16, 16, mandl_ctx)

    parse_options(mandl_ctx, view_ctx)
   
    view_ctx.setup()
    view_ctx.run()


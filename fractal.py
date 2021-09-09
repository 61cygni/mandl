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
import json # For params file reading and writing

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
from mandelbrot_solo import MandelbrotSolo
from mandelbrot_smooth import MandelbrotSmooth
from mandeldistance import MandelDistance
from julia_solo import JuliaSolo
from julia_smooth import JuliaSmooth 

MANDL_VER = "0.1"

class DiveTimeline: 
    """
    Representation of an edit timeline. This maps parameters to specific 
    times, which can be used for generating all the frames of a sequence.
 
    The first span is a special 'main' span which defines the run time
    of the timeline, regardless of the properties of any other spans.

    Overview of the sequencing classes
    ----------------------------------
    DiveTimeline
    DiveTimelineSpan (Basis for setting keyframes)
    DiveSpanKeyframe
    - DiveSpanCustomKeyframe (not implemented yet)
   
    # TODO: Seems like algorithm should be a per-span property, instead of
    # per-timeline, doesn't it?
    """

    @staticmethod
    def algorithm_map():
        return {'mandelbrot_solo' : MandelbrotSolo,
                'mandelbrot_smooth' : MandelbrotSmooth,
                'mandeldistance' : MandelDistance,
                'julia_solo' : JuliaSolo, 
                'julia_smooth' : JuliaSmooth, 
                #'smooth': Smooth,
        }

    @staticmethod
    def build_from_json_and_params(json, params):
        newTimeline = DiveTimeline(params['project_name'], json['algorithmName'], float(json['framerate']), int(json['frameWidth']), int(json['frameHeight']), None)
        newTimeline.__setstate__(json)
        return newTimeline 

    def __init__(self, projectFolderName, algorithmName, framerate, frameWidth, frameHeight, mathSupport):
        
        self.projectFolderName = projectFolderName
        self.algorithmName = algorithmName

        self.framerate = float(framerate)
        self.frameWidth = int(frameWidth)
        self.frameHeight = int(frameHeight)

        self.mathSupport = mathSupport

        # No definition made yet for edit gaps, so let's just enforce adjacency of ranges for now.
        self.timelineSpans = []

    def getFramesInDuration(self, duration):
        # Use 6 decimal places for rounding down, but otherwise, round up.
        # 1 second *  24 frames / second = 24 frames
        # .041 seconds * 24 frames / second = .984 frames (should be 1)
        # .042 seconds * 24 frames / second = 1.008 frames (should be 2)
        # .041666 * 24 frames / second = .999984 frames (should be 1)
        # .04166666 * 24 frames / second = .99999984 frames (should be 1)
        # .04166667 * 24 frames / second = 1.00000008 frames (should be 1)
        # .0416667 * 24 frames / second = 1.0000008 frames (should be 2)
        rawCount = (duration / 1000) * self.framerate 
        framesCount = int(rawCount)
        remainderCount = rawCount - framesCount

        if remainderCount != 0.0 and round(remainderCount,6) != 0.0:
            framesCount += 1

        return framesCount

    def getTimeForFrameNumber(self, frameNumber):
        # Want to return max time when within 1 frame, right?  To be nice.
        # When 9 frames, the frames are 0-8
        # That's 9 positions, with 8 deltas.
        rawTime = frameNumber * 1000 * (1 / self.framerate)
        return int(rawTime)

    def getMainSpan(self):
        if len(self.timelineSpans) == 0:
            return None
        else:
            return self.timelineSpans[0]

    def getTotalSpanFrameCount(self):
        # Duration 
        # Span durations are exclusive, so a duration of 100 means it covers 0-99.
        mainSpan = self.getMainSpan()
        if mainSpan != None:
            return self.getFramesInDuration(mainSpan.duration)
        else:
            return 0

    def addNewSpan(self, time, duration):
        span = DiveTimelineSpan(self, time, duration)
        self.timelineSpans.append(span)
        return span         

    def getSpansForTime(self, targetTime):
        overlappingSpans = []
        for currSpan in self.timelineSpans:
            if currSpan.time <= targetTime and (currSpan.time + currSpan.duration) >= targetTime:
                overlappingSpans.append(currSpan)
        return overlappingSpans
 
    def getMeshForTime(self, targetTime):
        """
        Calculate a discretized 2d plane of complex points based on spans and keyframes in this timeline.

        Order of operations:
         1.) base window definitions 
         2.) distortions on generators
         3.) overall distortions on the calculated 2D mesh
        """
        targetSpans = self.getSpansForTime(targetTime)

        # Enforce that the 'main' span (the first span) at least, was returned 
        if len(targetSpans) == 0 or targetSpans[0] != self.timelineSpans[0]:
            raise IndexError("Time '%d' (milliseconds) is out of range for this timeline" % targetTime)

        # Build up the set of parameter values, as interpolated between nearest
        # surrounding keyframes.
        # Take one entire span at a time, before moving to the next span.
        # Deny a second span setting an already existing parameter directly.
        # To allow a following span to adjust the setting, its target must be 
        # named as 'parameterName_modifiyPlus' or 'parameterName_modifyTimes'.
        plusSuffix = "_modifyPlus"
        timesSuffix = "_modifyTimes"

        parameterValues = {}
        for currSpan in targetSpans:
            spanParamValues = currSpan.getParamValuesAtTime(targetTime, parameterValues)
            for newParamName in spanParamValues:
                #print(f"Looking at parameter name \"{newParamName}\"")
                if newParamName in parameterValues:
                    raise ValueError(f"Spans can't overwrite existing parameters, so can't set the value of \"{newParamName}\" from \"{parameterValues[newParamName]}\" to \"{spanParamValues[newParamName]}\".  You might be trying to modify an existing value, in which case it could be done with a parameter named \"{newParamName}_modifyPlus\".")
              
                elif newParamName.endswith(plusSuffix):
                    realParamName = newParamName[:-len(plusSuffix)]
                    #print(f"PLUS maps to \"{realParamName}\"")
                    if realParamName not in parameterValues:
                        print(f"WARNING - span's attempt to modify \"{realParamName}\" failed, because it wasn't previously defined")
                    else:
                        parameterValues[realParamName] += spanParamValues[newParamName]
                elif newParamName.endswith(timesSuffix):
                    realParamName = newParamName[:-len(timesSuffix)]
                    #print(f"TIMES maps to \"{realParamName}\"")
                    if realParamName not in parameterValues:
                        print(f"WARNING - span's attempt to modify \"{realParamName}\" failed, because it wasn't previously defined")
                    else:
                        parameterValues[realParamName] *= spanParamValues[newParamName]
                else:
                    #print(f"just assigning param \"{newParamName}\"")
                    parameterValues[newParamName] = spanParamValues[newParamName]

        # Use the parameter values needed for constructing the mesh window,
        # and REMOVE them from the parameter list, leaving only 'extra' parameters.
        #
        # I suppose just trying to access non-existent params here will be enough
        # to trigger a runtime warning if they weren't defined, so not adding
        # any extra guards for now.
        meshCenter = parameterValues['meshCenter']
        del(parameterValues['meshCenter'])

        meshRealWidth = parameterValues['meshRealWidth']
        del(parameterValues['meshRealWidth'])

        meshImagWidth = parameterValues['meshImagWidth']
        del(parameterValues['meshImagWidth'])

        # Not strictly required, and 0.0 tilt (uniform) is assumed if not specified.
        meshRealTilt = 0.0
        meshImagTilt = 0.0
        if 'meshRealTilt' in parameterValues:
            meshRealTilt = parameterValues['meshRealTilt']
            del(parameterValues['meshRealTilt'])
        if 'meshImagTilt' in parameterValues:
            meshImagTilt = parameterValues['meshImagTilt']
            del(parameterValues['meshImagTilt'])
        
        realMeshGenerator = None
        imagMeshGenerator = None

        #print(f"parameters near the end of getMeshForTime(): \"{parameterValues}\"")

        if meshRealTilt == 0.0:
            realMeshGenerator = mesh.MeshGeneratorUniform(mathSupport=self.mathSupport, varyingAxis='width', valuesCenter=meshCenter.real, baseWidth=meshRealWidth)
        else:
            realMeshGenerator = mesh.MeshGeneratorTilt(mathSupport=self.mathSupport, varyingAxis='width', valuesCenter=meshCenter.real, baseWidth=meshRealWidth, tiltAngle=float(meshRealTilt))

        if meshImagTilt == 0.0:
            imagMeshGenerator = mesh.MeshGeneratorUniform(mathSupport=self.mathSupport, varyingAxis='height', valuesCenter=meshCenter.imag, baseWidth=meshImagWidth)
        else:
            imagMeshGenerator = mesh.MeshGeneratorTilt(mathSupport=self.mathSupport, varyingAxis='height', valuesCenter=meshCenter.imag, baseWidth=meshImagWidth, tiltAngle=float(meshImagTilt))
        
        #print(f"Making mesh with realWidth {meshRealWidth} and imagWidth {meshImagWidth}")   
        diveMesh = mesh.DiveMesh(self.frameWidth, self.frameHeight, realMeshGenerator, imagMeshGenerator, self.mathSupport, parameterValues)

        #print(diveMesh)
        return diveMesh

    def __getstate__(self):
        """ Pickle encoding helper, generates simply encoded keyframes. """
        pickleInfo = self.__dict__.copy()

        # Not storing project folder in timeline file, because
        # that's parameters and param file's responsibility
        del(pickleInfo['projectFolderName'])

        # Going to encode both the class name of the MathSupport, and
        # the 'precision' it was apparently set at, making a string
        # like "DiveMathSupportFlint:2048".
        mathSupportString = type(self.mathSupport).__name__ + ":" + str(self.mathSupport.precision())
        pickleInfo['mathSupport'] = mathSupportString
       
        convertedSpans = []
        for currSpan in self.timelineSpans:
            convertedSpans.append(currSpan.__getstate__())
        pickleInfo['timelineSpans'] = convertedSpans

        return pickleInfo

    def __setstate__(self, state):
        """
        (Just like in divemesh.py...)
         A new MathSupport sublass is instantiated during un-pickling.
        This can be a problem if you're relying on specific precision settings, 
        because the only MathSupport configuration that's handled here
        is setting the precision from the encode classname:precision string.

        It *is* important to set the precision before loading any numbers, or
        else the numbers may be clipped to lower precision than what they 
        were saved at.
        """
        mathSupportClasses = {"DiveMathSupportFlintCustom":fm.DiveMathSupportFlintCustom,
                "DiveMathSupportFlint":fm.DiveMathSupportFlint,
                "DiveMathSupport":fm.DiveMathSupport}

        (mathSupportClassName, precisionString) = state['mathSupport'].split(':')
        #print("mathSupport reads as: %s" % mathSupportClassName)

        self.mathSupport = mathSupportClasses[mathSupportClassName]()
        self.mathSupport.setPrecision(int(precisionString))
        #print("mathSupport is: %s" % str(self.mathSupport))
        #print(f"timeline's mathSupport set to {self.mathSupport.digitsPrecision()} digits")

        #self.projectFolderName = state['projectFolderName']
        self.algorithmName = state['algorithmName']
        self.framerate = float(state['framerate'])
        self.frameWidth = int(state['frameWidth'])
        self.frameHeight = int(state['frameHeight'])

        for storedSpanInfo in state['timelineSpans']:
            newSpan = self.addNewSpan(storedSpanInfo['time'], storedSpanInfo['duration'])
            newSpan.__setstate__(storedSpanInfo)


class DiveTimelineSpan:
    """

    """
    def __init__(self, timeline, timePosition, duration):
        self.timeline = timeline
        self.time = int(timePosition)
        self.duration = int(duration)

        # Only a single 'track' for each keyframe type, so each named
        # parameter can only have one keyframe for one time stamp within
        # a single span.  Spans can layer though.
        self.parameterKeyframes = defaultdict(dict)
        # parameterKeyframes['meshRealWidth'][2300] = keyframeObject

    def getParamValuesAtTime(self, targetTime, currentParameters):
        """
        The returned set of parameters are only the parameters from this 
        span, and do not automatically include values from the
        currentParameters input.
        
        Current parameters are used only as additional information for calculations.
        """
        answerParameters = {}
    
        keyframesList = self.getKeyframesClosestToTime(targetTime)
    
        for (parameterName, previousKeyframe, nextKeyframe) in keyframesList:
            # TODO: Would like to enforce matching transitions, rather than
            # just using one keyframe's transition type.  But, that kinda requires
            # a more compassionate implementation of adding keyframes which validates
            # surrounding types and values.
            if isinstance(previousKeyframe, DiveSpanCustomKeyframe) and isinstance(nextKeyframe, DiveSpanCustomKeyframe):
                # Keyframes are 'call-a-function' type
    
                # I guess, only the first keyframe's function is used?
                # When keyframes were retrieved, their time position was stashed.
                # Calculate how far along the target time is between the keyframes
                targetPercentBetweenKeyframes = ((target - prev.time) / (next.time - prev.time))
                answerParameters[parameterName] = previousKeyframe.calculateValueForTime(targetTime, targetPercentBetweenKeyframes, currentParameters)
            else:
                #  Keyframes are 'values-to-interpolate' type
                answerParameters[parameterName] = self.interpolateBetweenKeyframes(targetTime, previousKeyframe, nextKeyframe)
   
        print(answerParameters)
 
        return answerParameters
    
    def getKeyframesClosestToTime(self, targetTime):
        answerKeyframes = []
        # answerKeyframes[(paramName, previousKeyframe, nextKeyframe),...]

        # parameterKeyframes['meshRealWidth'][2300] = keyframeObject
        for currParamName, keyframesByTime in self.parameterKeyframes.items(): 
            #print(f"Looking at {currParamName}")

            # Direct hit 
            if targetTime in keyframesByTime:
                targetKeyframe = keyframesByTime[targetTime]
                targetKeyframe.lastObservedTime = targetTime
                answerKeyframes.append((currParamName, targetKeyframe, targetKeyframe))
                continue

            previousKeyframe = None
            nextKeyframe = None
            for currTime in sorted(keyframesByTime.keys()):
                if currTime <= targetTime:
                    previousKeyframe = keyframesByTime[currTime]
                    previousKeyframe.lastObservedTime = currTime
                if currTime > targetTime:
                    nextKeyframe = keyframesByTime[currTime]
                    nextKeyframe.lastObservedTime = currTime
                    break # Past the sorted range, so done looking
  
            if previousKeyframe != None and nextKeyframe != None:
                answerKeyframes.append((currParamName, previousKeyframe, nextKeyframe))

        #print("found: %s" % str(answerKeyframes))
        return answerKeyframes

    def interpolateBetweenKeyframes(self, targetTime, leftKeyframe, rightKeyframe):
        """
        Relies heavily on the stashed/cached 'lastObservedTime' of a keyframe
        """
        print("interpolating %s -> %s at time %s" % (str(leftKeyframe), str(rightKeyframe), str(targetTime)))

        if targetTime < leftKeyframe.lastObservedTime or targetTime > rightKeyframe.lastObservedTime:
            raise IndexError("Time '%d' isn't between 2 keyframes at '%d' and '%d'" % (targetTime, leftKeyframe.lastObservedTime, rightKeyframe.lastObservedTime))

        # Let's just be lenient for now?
        ## Enforce that left keyframe's transitionOut should match right
        ##  keyframe's transitionIn
        #if leftKeyframe.transitionOut != rightKeyframe.transitionIn:
        #    raise ValueError("Keyframe transition types mismatched for frame number '%d'" % frameNumber)
        transitionType = leftKeyframe.transitionOut
        mathSupport = self.timeline.mathSupport

        if leftKeyframe.dataType == 'float':
            # Recognize when left and right are the same, and dont' calculate anything.
            if leftKeyframe == rightKeyframe:
                return leftKeyframe.value

            return mathSupport.interpolate(transitionType, leftKeyframe.lastObservedTime, leftKeyframe.value, rightKeyframe.lastObservedTime, rightKeyframe.value, targetTime)
        elif leftKeyframe.dataType == 'complex':
            # Recognize when left and right are the same, and dont' calculate anything.
            if leftKeyframe == rightKeyframe:
                return leftKeyframe.value
            interpolatedReal = mathSupport.interpolate(transitionType, leftKeyframe.lastObservedTime, leftKeyframe.value.real, rightKeyframe.lastObservedTime, rightKeyframe.value.real, targetTime)
            interpolatedImag = mathSupport.interpolate(transitionType, leftKeyframe.lastObservedTime, leftKeyframe.value.imag, rightKeyframe.lastObservedTime, rightKeyframe.value.imag, targetTime)
            return mathSupport.createComplex(interpolatedReal, interpolatedImag)
        elif leftKeyframe.dataType == 'int':
            # Recognize when left and right are the same, and dont' calculate anything.
            if leftKeyframe == rightKeyframe:
                return int(leftKeyframe.value)

            return int(float(mathSupport.interpolate(transitionType, leftKeyframe.lastObservedTime, leftKeyframe.value, rightKeyframe.lastObservedTime, rightKeyframe.value, targetTime)))
        else:
            raise ValueError(f"Can't perform interpolation for data type \"{leftKeyframe.dataType}\"")


    ####
    # TODO: All of these helper functions need to perform a modification 
    # of upstream and downstream keyframes when forcing addition, to make
    # sure the transition types are all in order.
    # When creating a new keyframe, default is to inherit the lead-in and 
    # lead-out interpolators of the existing span.
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

    def addNewParameterKeyframe(self, targetTime, dataType, name, value, transitionIn='linear', transitionOut='linear'):
        newKeyframe = DiveSpanKeyframe(self, dataType, value, transitionIn, transitionOut)
        self.parameterKeyframes[name][targetTime] = newKeyframe 
        return newKeyframe

    def renderKeyframesAsStringList(self):
        # First shovel all values into the array, then sort by time
        # and re-assign the time to also be a string.
        answerList = []
        for currParamName, keyframesByTime in self.parameterKeyframes.items():
            for currTime, currKeyframe in keyframesByTime.items():
                # TODO: should have at least 2 types of keyframes, one for
                # values, and one for functions, but starting with JUST values...
                answerList.append([currTime, currParamName, currKeyframe.dataType, str(currKeyframe.value), currKeyframe.transitionIn, currKeyframe.transitionOut])

        answerList = sorted(answerList)
        for currItem in answerList:
            currItem[0] = str(currItem[0])

        return answerList

    def setKeyframesFromStringList(self, keyframesList):
        # Completely overwrite any existing keyframes
        self.parameterKeyframes = defaultdict(dict)

        mathSupport = self.timeline.mathSupport
        # TODO: Again, should have at least 2 types of keyframes, one for
        # values, and one for functions, but starting with JUST values...
        for (timeString, paramName, dataType, valueString, transitionIn, transitionOut) in keyframesList:
            loadedValue = None
            if dataType == 'float':
                loadedValue = mathSupport.createFloat(valueString)
            elif dataType == 'complex':
                loadedValue = mathSupport.createComplex(valueString)
            elif dataType == 'int':
                loadedValue = int(valueString)

            self.parameterKeyframes[paramName][int(float(timeString))] = DiveSpanKeyframe(self, dataType, loadedValue, transitionIn, transitionOut) 

    def __getstate__(self):
        """ Pickle encoding helper, generates all string arrays and hashes. """
        spanState = {}
        spanState['time'] = str(self.time)
        spanState['duration'] = str(self.duration)
        spanState['keyframes'] = self.renderKeyframesAsStringList()
        return spanState

    def __setstate__(self, state):
        self.time = int(state['time'])
        self.duration = int(state['duration'])
        self.setKeyframesFromStringList(state['keyframes'])

class DiveSpanKeyframe:
    def __init__(self, span, dataType, value, transitionIn='quadratic-to', transitionOut='quadratic-from'):
        
        dataTypeOptions = ['float', 'complex', 'int', 'no_type'] 
        # Probably would like int and bool too?
        if dataType not in dataTypeOptions:
            raise ValueError("dataType must be one of (%s)" % ", ".join(dataTypeOptions))

# log-from doesn't work yet...
        transitionOptions = ['quadratic-to', 'quadratic-from', 'quadratic-to-from', 'log-to', 'root-to', 'root-from', 'root-to-ease-in', 'root-to-ease-out', 'root-from-ease-in', 'root-from-ease-out', 'linear', 'step']
        if transitionIn not in transitionOptions:
            raise ValueError("transitionIn must be one of (%s)" % ", ".join(transitionOptions))
        if transitionOut not in transitionOptions:
            raise ValueError("transitionOut must be one of (%s)" % ", ".join(transitionOptions))

        self.span = span 
        self.dataType = dataType
        self.value = value
        self.transitionIn = transitionIn
        self.transitionOut = transitionOut

        self.lastObservedTime = 0 # For stashing times in

    def __repr__(self):
        return(f"DiveSpanKeyframe {self.lastObservedTime} {self.dataType} {self.value}, in={self.transitionIn}, out={self.transitionOut}")

class DiveSpanCustomKeyframe(DiveSpanKeyframe):
    """
    TODO: Don't use this yet.  
    Function-based keyframes aren't yet handled by the persistence functions.
    """
    def __init__(self, span, targetObject, targetFunction, transitionIn='linear', transitionOut='linear'):
        super().__init__(span, 'no_type', None, transitionIn, transitionOut)
        self.targetObject = targetObject
        self.targetFunction = targetFunction

    def calculateValueForTime(self, targetTime, targetPercentBetweenKeyframes, currentParameters):
        print("CALCULATION CANCELLED - DEBUGGING DiveSpanCustomKeyframe")
        # Not even tested yet...
        #runnableFunction = getattr(self.targetObject, targetFunction)
        #return runnableFunction(targetTime, targetPercentBetweenKeyframes, currentParameters)

def make_project(params):
    """
    Sets up a project directory structure, as long as the provided
    name is legal enough and not already taken.

    It might be more pure to only create subdirs as they're needed
    and used, but it's actually helpful to have a pre-defined structure
    in place to rely on, even if the paths end up shifting around later
    because of changes to settings.
    """
    folder_name = params['make_project_name']
    print("Making project: \"%s\"" % folder_name)
    if os.path.exists(folder_name):
        print("NOT CREATING project:")
        print("  File already exists where fractal project would be created.")
        exit(0)

    project_params = {
        "math_support": params['math_support'].precisionType,
        "exploration_mesh_width": "160",
        "exploration_mesh_height": "120",
        "exploration_default_algo_name": "mandelbrot_solo",
        "exploration_default_zoom_factor": "0.8",
        "exploration_output_path": "exploration/output",
        "exploration_markers_path": "exploration/markers",
        "edit_markers_path": "edit/markers",
        "edit_timelines_path": "edit/timelines",
        "render_image_width": "1024",
        "render_image_height": "768",
        "render_fps": "23.976",
        "render_output_path": "output",
        "render_exports_path": "exports",
        }
 
    subfolder_names = [project_params['exploration_output_path'],
        project_params['exploration_markers_path'],
        project_params['edit_markers_path'],
        project_params['edit_timelines_path'],
        project_params['render_output_path'],
        project_params['render_exports_path'],
        ]

    for curr_subfolder_name in subfolder_names:
        os.makedirs(os.path.join(folder_name, curr_subfolder_name))

    param_file_name = os.path.join(folder_name, 'params.json')
    with open(param_file_name, 'wt') as param_handle:
        param_handle.write(json.dumps(project_params, indent=4))
        param_handle.close()

    print("Project folder created at \"%s\"" % folder_name)

def parse_options():
    """
    Handles parsing of command line parameters, including those specified
    in Algo implementations.

    For all modes except "--make-project", also loads the values 
    from params.json into params['project_params']
    """
    params = {} # Most everything here fills this dictionary

    options_list = ["math-support=",
                    "digits-precision=",
                    "project=",
                    # Mode params
                    "make-project=",
                    "exploration",
                    "timeline-name=",
                    # Exploration-only params
                    "expl-algo=",
                    "expl-real-width=",
                    "expl-imag-width=",
                    "expl-center=",
                    "expl-frame-number=",
                    # Timeline-only params
                    "batch-frame-file=",
                   ]

    # Add *all* the command line options that Algos recognize, even though
    # I'd rather only load up for the active Algo.  There's not a really
    # clean way to first load the Algo, and second, trim out the algo-specific
    # options, in a way that keeps getopt working simply.      
    algorithm_map = DiveTimeline.algorithm_map()
    algorithm_extra_params = {}
    for algorithm_name, algorithm_class in algorithm_map.items():
        options_list.extend(algorithm_class.options_list())
    # Make param list unique, just for readability
    options_list = list(set(options_list)) 
    opts, args = getopt.getopt(sys.argv[1:], "", options_list)

    # First-pass at params to set up MathSupport because lots of 
    # things depend on it for names and types.
    math_support_classes = {'native': fm.DiveMathSupport,
        'flint': fm.DiveMathSupportFlint,
        'flintcustom': fm.DiveMathSupportFlintCustom,
        # maybe not complete?# 'gmp': fm.DiveMathSupportGmp,
        # maybe not complete?# 'decimal': fm.DiveMathSupportDecimal,
        # definitely not built yet.# 'libbf': fm.DiveMathSupportLibbf,
    } 
    math_support = math_support_classes['native']() # Creates an instance
    for opt, arg in opts:
        if opt in ['--math-support']:
            if arg in math_support_classes:
                # Creates an instance
                math_support = math_support_classes[arg]() 
        elif opt in ['--digits-precision']:
            params['digits_precision'] = int(arg)
    # Important to also set expected precision before parsing param values
    support_precision = params.get('digits_precision', 16) # 16 == native
    math_support.setPrecision(round(support_precision * 3.32)) # ~3.32 bits per position
    params['math_support'] = math_support

    # Second-pass at params, figures out which mode we're operating
    # in, so enforcement of parameter consistency can be localized.
    #
    # 4 modes: make-project, exploration, timeline, batch-in-timeline
    mode_count = 0
    for opt, arg in opts:
        if opt in ['--make-project']:
            params['mode'] = 'make_project'
            params['make_project_name'] = arg
            mode_count += 1
        elif opt in ['--exploration']:
            params['mode'] = 'exploration'
            mode_count += 1
        elif opt in ['--timeline-name']:
            # 'timeline' mode may be overwritten later with more specificity
            params['mode'] = 'timeline'
            params['timeline_name'] = arg
            mode_count += 1
        elif opt in ['--project']:
            params['project_name'] = arg

    if mode_count != 1:
        raise ValueError("Exactly one of '--make-project=', '--timeline-name=', or '--exploration' is required to set the running mode.")

    # Special case where we can do all the work here and be done.
    if params['mode'] == 'make_project':
        make_project(params)
        exit(0)

    # We know we're in a project mode, if we made it this far, so
    # require that a project has been specified.
    if 'project_name' not in params:
        raise ValueError("Specifying --project=<name> is required")

    # Load project parameters out of the params file
    param_file_name = os.path.join(params['project_name'], 'params.json')
    with open(param_file_name, 'rt') as param_handle:
        params['project_params'] = json.load(param_handle)
 
    # Fourth pass at parameters, gathering those that are
    # mode-specific, with mode-specific enforcement 
    if params['mode'] == 'exploration':
        # Exploration also reads parameters out of the project's params.json:
        # exploration_mesh_width
        # exloration_mesh_height
        # exploration_output_path
        for opt, arg in opts:
            if opt in ['--expl-algo']:
                params['expl_algo'] = arg
            elif opt in ['--expl-real-width']:
                params['expl_real_width'] = math_support.createFloat(arg) 
            elif opt in ['--expl-imag-width']:
                params['expl_imag_width'] = math_support.createFloat(arg)
            elif opt in ['--expl-center']:
                params['expl_center'] = math_support.createComplex(arg)
            elif opt in ['--expl-frame-number']:
                params['expl_frame_number'] = int(arg)

        required_params = ["expl_algo",
                    "expl_real_width",
                    "expl_imag_width",
                    "expl_center",
                    "expl_frame_number",
        ]
        for param_name in required_params:
            if param_name not in params:
                raise ValueError("For exploration mode, \"%s\" is a required parameter (well, that name with hyphens, not underscores)." % param_name)

        # Kinda a crazy invocation.  Loads algorithm-specific parameters into
        # a dictionary, based on that algorithm's static class parse function.
        expl_algorithm_name = params.get('expl_algo', 'mandelbrot_solo')
        params['algorithm_extra_params'] = algorithm_map[expl_algorithm_name].load_options_with_math_support(opts, math_support)
        # Theoretically possible we'll eventually want to run this for all
        # possible algorithm types, but for now, just loading for the 
        # 'active' algorithm.

    elif params['mode'] == 'timeline':
        for opt, arg in opts:
            if opt in ['--batch-frame-file']:
                # We're in batch mode, instead of entire-timeline mode,
                # so get more specific.
                params['mode'] = 'batch_timeline'
                params['batch_frame_file'] = arg

    return params

def run_exploration(params):
    algorithm_name = params['expl_algo']
    project_params = params['project_params']
    project_folder_name = params['project_name']

    timeline = DiveTimeline(projectFolderName=project_folder_name, algorithmName=algorithm_name, framerate=23.976, frameWidth=project_params['exploration_mesh_width'], frameHeight=project_params['exploration_mesh_height'], mathSupport=params['math_support'])
    mesh_real_width = params['expl_real_width']
    mesh_imag_width = params['expl_imag_width']
    mesh_center = params['expl_center']

    # Note: --max-escape-iterations is passed along via algorithm_extra_params

    span_duration = 40
    main_span = timeline.addNewSpan(0,span_duration)

    main_span.addNewParameterKeyframe(0, 'complex', 'meshCenter', mesh_center, transitionIn='root-to', transitionOut='root-to')
    main_span.addNewParameterKeyframe(span_duration, 'complex', 'meshCenter', mesh_center, transitionIn='root-to', transitionOut='root-to')

    main_span.addNewParameterKeyframe(0, 'float', 'meshRealWidth', mesh_real_width, transitionIn='root-to', transitionOut='root-to')
    main_span.addNewParameterKeyframe(span_duration, 'float', 'meshRealWidth', mesh_real_width, transitionIn='root-to', transitionOut='root-to')

    main_span.addNewParameterKeyframe(0, 'float', 'meshImagWidth', mesh_imag_width, transitionIn='root-to', transitionOut='root-to')
    main_span.addNewParameterKeyframe(span_duration, 'float', 'meshImagWidth', mesh_imag_width, transitionIn='root-to', transitionOut='root-to')

    extra_params = params['algorithm_extra_params']
    frame_time = timeline.getTimeForFrameNumber(0)
    dive_mesh=timeline.getMeshForTime(frame_time)
    # Allow per-frame info to overwrite algorithm info.
    extra_params.update(dive_mesh.extraParams) 

    output_folder_name = os.path.join(project_folder_name, project_params['exploration_output_path'])

    # Following is a class instantiation, of a string-specified Algo class
    algorithm_map = DiveTimeline.algorithm_map()
    frame_algorithm = algorithm_map[algorithm_name](dive_mesh=dive_mesh, frame_number=params['expl_frame_number'], output_folder_name=output_folder_name, extra_params=extra_params)

    frame_algorithm.run()

def run_timeline(params):
    """
    Parameters come from command-line script 'params' hash, as well
    as from the 'project_params' sub-hash, which loads the params.json
    """
    project_params = params['project_params']
    project_folder_name = params['project_name']
    timeline_name = params['timeline_name']

    timeline_file_name = os.path.join(params['project_name'], params['project_params']['edit_timelines_path'], params['timeline_name'])
    timeline = load_timeline_from_file(timeline_file_name, params)

    main_span = timeline.getMainSpan()
    frame_count = timeline.getFramesInDuration(main_span.duration) 
    print(f"duration: {main_span.duration}")
    print(f"frame count: {frame_count}")

    output_folder_name = os.path.join(project_folder_name, project_params['render_output_path'], timeline_name)
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    frame_file_names = []
    for curr_frame_number in range(frame_count):
        frame_time = timeline.getTimeForFrameNumber(curr_frame_number)
        print(f"frame {curr_frame_number} at {frame_time}")
        dive_mesh = timeline.getMeshForTime(frame_time)    
        print(f"{dive_mesh.realMeshGenerator.baseWidth} x {dive_mesh.imagMeshGenerator.baseWidth} ({dive_mesh.extraParams['max_escape_iterations']} iter)")
        # Following is a class instantiation, of a string-specified Algo class
        algorithm_map = DiveTimeline.algorithm_map()
        frame_algorithm = algorithm_map[timeline.algorithmName](dive_mesh=dive_mesh, frame_number=curr_frame_number, output_folder_name=output_folder_name, extra_params=dive_mesh.extraParams)
        frame_algorithm.run()

        frame_file_names.append(frame_algorithm.output_image_file_name)

    export_base_name = f"{timeline_name}.mp4"
    export_file_name = os.path.join(project_folder_name, project_params['render_exports_path'], export_base_name)
    clip = mpy.ImageSequenceClip(frame_file_names, fps=timeline.framerate)
    clip.write_videofile(export_file_name, fps=timeline.framerate, audio=False, codec="mpeg4")

def load_timeline_from_file(timeline_file_name, params):
    print(f"loading timeline from:\n{timeline_file_name}")
    with open(timeline_file_name, 'rt') as timeline_handle:
        timelineJSON = json.load(timeline_handle)
        timeline = DiveTimeline.build_from_json_and_params(timelineJSON, params)
    return timeline

####
#### This block will be useful when timeline is a script loader, not just json
####
#    timeline_file_base = u"%s.py" % timeline_name
#    timeline_file = os.path.join(project_folder_name, project_params['edit_timelines_path'], timeline_file_base)
#
#    # Using SourceFileLoader to load a script file from a specified path.
#    from importlib.machinery import SourceFileLoader
#    timeline_module = SourceFileLoader('timeline_file', timeline_file).load_module()
#    timeline = timeline_module.getTimeline(params)
#
#    timeline = debugTimelineMaker(params)
####

def debugTimelineMaker(params):
    project_folder_name = params['project_name']
    math_support = params['math_support']

    timeline_name = params['timeline_name']

    project_params = params['project_params']

    framerate = float(project_params.get('render_fps', 23.976))
    frame_width = int(project_params.get('render_image_width', 160))
    frame_height = int(project_params.get('render_image_height', 120))

    max_iterations = 255

    timeline = DiveTimeline(project_folder_name, 'mandelbrot_smooth', framerate, frame_width, frame_height, math_support)

    span_duration = 2000 
    main_span = timeline.addNewSpan(0,span_duration)

    mesh_center = math_support.createComplex('-1.76938+0.00423j')
    mesh_real_width = math_support.createFloat('1.0')
    mesh_imag_width = math_support.createFloat(frame_height / frame_width * mesh_real_width) 
    
    end_mesh_real_width = mesh_real_width * .01
    end_mesh_imag_width = mesh_imag_width * .01

    main_span.addNewParameterKeyframe(0, 'complex', 'meshCenter', mesh_center, transitionIn='root-to', transitionOut='root-to')
    main_span.addNewParameterKeyframe(span_duration, 'complex', 'meshCenter', mesh_center, transitionIn='root-to', transitionOut='root-to')

    main_span.addNewParameterKeyframe(0, 'float', 'meshRealWidth', mesh_real_width, transitionIn='root-to', transitionOut='root-to')
    main_span.addNewParameterKeyframe(span_duration, 'float', 'meshRealWidth', end_mesh_real_width, transitionIn='root-to', transitionOut='root-to')

    main_span.addNewParameterKeyframe(0, 'float', 'meshImagWidth', mesh_imag_width, transitionIn='root-to', transitionOut='root-to')
    main_span.addNewParameterKeyframe(span_duration, 'float', 'meshImagWidth', end_mesh_imag_width, transitionIn='root-to', transitionOut='root-to')
  
    iterations_delta = 200 
    main_span.addNewParameterKeyframe(0, 'int', 'max_escape_iterations', max_iterations, transitionIn='linear', transitionOut='linear')
    main_span.addNewParameterKeyframe(span_duration * .25, 'int', 'max_escape_iterations', max_iterations-iterations_delta, transitionIn='linear', transitionOut='linear')
    main_span.addNewParameterKeyframe(span_duration * .5, 'int', 'max_escape_iterations', max_iterations, transitionIn='linear', transitionOut='linear')
    main_span.addNewParameterKeyframe(span_duration * .75, 'int', 'max_escape_iterations', max_iterations-iterations_delta, transitionIn='linear', transitionOut='linear')
    main_span.addNewParameterKeyframe(span_duration, 'int', 'max_escape_iterations', max_iterations, transitionIn='linear', transitionOut='linear')

    return timeline
   
def run_batch_timeline(params):
    """
    Parameters come from command-line script 'params' hash, as well
    as from the 'project_params' sub-hash, which loads the params.json
    """
    project_params = params['project_params']
    project_folder_name = params['project_name']
    timeline_name = params['timeline_name']

    timeline_file_name = os.path.join(params['project_name'], params['project_params']['edit_timelines_path'], params['timeline_name'])
    timeline = load_timeline_from_file(timeline_file_name, params)

    main_span = timeline.getMainSpan()
    frame_count = timeline.getFramesInDuration(main_span.duration) 
    print(f"duration: {main_span.duration}")
    print(f"frame count: {frame_count}")

    output_folder_name = os.path.join(project_folder_name, project_params['render_output_path'], timeline_name)
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    frame_numbers = []
    batch_frame_file
    with open(batch_frame_file, 'rt') as batch_handle:
        for currLine in batch_handle:
            frameNumbers.append(int(currLine.strip()))

    #frame_file_names = []
    for curr_frame_number in frame_numbers: 
        frame_time = timeline.getTimeForFrameNumber(curr_frame_number)
        print(f"batch frame {curr_frame_number} at {frame_time}")
        dive_mesh = timeline.getMeshForTime(frame_time)    
        print(f"{dive_mesh.realMeshGenerator.baseWidth} x {dive_mesh.imagMeshGenerator.baseWidth} ({dive_mesh.extraParams['max_escape_iterations']} iter)")
        # Following is a class instantiation, of a string-specified Algo class
        algorithm_map = DiveTimeline.algorithm_map()
        frame_algorithm = algorithm_map[timeline.algorithmName](dive_mesh=dive_mesh, frame_number=curr_frame_number, output_folder_name=output_folder_name, extra_params=dive_mesh.extraParams)
        frame_algorithm.run()

        #frame_file_names.append(frame_algorithm.output_image_file_name)

if __name__ == "__main__":

    print("++ fractal.py version %s" % (MANDL_VER))

    params = parse_options()
    #print(params)
    mode = params['mode']
    if mode == 'exploration':
        run_exploration(params)
    elif mode == 'timeline':
        run_timeline(params)
    elif mode == 'batch_timeline':
        run_batch_timeline(params)
    else:
        raise ValueError("Run mode is unrecognized - abandoning run")

 

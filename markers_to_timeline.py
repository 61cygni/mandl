# --
# File: markers_to_timeline.py
#
# For a specified project (path), and a set of marker IDs, construct a
# timeline that places the markers as keyframes at 'reasonable' places
# across the duration.
# 
# All markers need to be stored in the project's exploration markers 
# subdirectory, named just by their marker ID (integer) number.
#
# --

import getopt
import os, sys

import pickle
import json

from divemesh import *
from fractalmath import *
from fractal import *

from algo import Algo # Abstract base class import, because we rely on it.
from mandelbrot_solo import MandelbrotSolo
from mandelbrot_smooth import MandelbrotSmooth
from mandeldistance import MandelDistance
from julia_solo import JuliaSolo
from julia_smooth import JuliaSmooth 

def parse_options():
    params = {}
    opts, args = getopt.getopt(sys.argv[1:], "", 
                               ["project=",
                                "timeline-name=",
                                "duration=",
                                "marker-list=",
                                ])

    # First-pass at params pulls out the project file, because
    # it probably gives us a default math support
    for opt, arg in opts:
        if opt in ['--project']:
            params['project_name'] = arg
        elif opt in ['--timeline-name']:
            params['timeline_name'] = arg
        elif opt in ['--marker-list']:
            params['raw_marker_list'] = eval(arg)  # expects a list of ints
        elif opt in ['--duration']:
            params['duration'] = float(arg)

    # Require that a project has been specified.
    if 'project_name' not in params:
        raise ValueError("Specifying --project=<name> is required")

    # Load project parameters out of the params file
    param_file_name = os.path.join(params['project_name'], 'params.json')
    with open(param_file_name, 'rt') as param_handle:
        params['project_params'] = json.load(param_handle)

    # Require at least 2 markers
    if not 'raw_marker_list' in params or len(params['raw_marker_list']) <= 1:
        raise ValueError("--marker-list=... List of marker IDs (ints) be at least two markers long")

    # Require a non-existing timeline name for writing
    if not 'timeline_name' in params:
        raise ValueError("Specifying --timeline-name=<name> is required")
    timeline_file_name = os.path.join(params['project_name'], params['project_params']['edit_timelines_path'], params['timeline_name'])
    if os.path.exists(timeline_file_name):
        raise ValueError("NOT CREATING timeline:\n   File already exists where timeline would be created.")

    params['timeline_file_name'] = timeline_file_name

    # Require a duration (in seconds)
    if 'duration' not in params:
        raise ValueError("Specifying --duration=<seconds.float> is required")

    # Calculate the needed duration for the parameters, which is
    # basically rounding up the parameter duration to the next full frame's time.
    framerate = float(params['project_params']['render_fps'])
    coveringFrameCount = getFramesForSecondsAndFramerate(params['duration'], framerate) 
    params['millisecond_duration'] = getDurationForFramesAndFramerate(coveringFrameCount, framerate)

    return params

def loadMarkers(params):
    # Only one algo type allowed for the entire set of markers.
    observedAlgorithm = None

    markerList = []
    for currMarkerID in params['raw_marker_list']:
        markerTitle = "%d.marker.pik" % currMarkerID
        markerFileName = os.path.join(params['project_name'], params['project_params']['exploration_markers_path'], markerTitle)

        with open(markerFileName, 'rb') as markerHandle:
            currMarker = pickle.load(markerHandle)
      
        if observedAlgorithm == None:
            observedAlgorithm = currMarker.algorithmName
        if observedAlgorithm != currMarker.algorithmName:
            raise ValueError(f"Can't ingest markers with mixed algorithm types \"{observedAlgorithm}\" and \"{currMarker.algorithmName}\"")
        
        markerList.append(currMarker)
 
    return markerList

def getMathSupportFromMarkerList(markerList):
    # Also for now, (because no converters written yet), force all 
    # markers to be one math support type, though precision may vary.

    observedPrecisionType = None
    maxDigits = 1
    for currMarker in markerList:
        if observedPrecisionType == None:
            observedPrecisionType = currMarker.diveMesh.mathSupport.precisionType
        if observedPrecisionType != currMarker.diveMesh.mathSupport.precisionType:
            raise ValueError(f"Can't ingest markers with mixed MathSupport types \"{observedPrecisionType}\" and \"{currMarker.diveMesh.mathSupport.precisionType}\"")

        currDigits = currMarker.diveMesh.mathSupport.digitsPrecision()

        if currDigits > maxDigits:
            maxDigits = currDigits

    # Going to steal the last mathSupport, and force it to the max
    # observed precision
    hijackedSupport = markerList[-1].diveMesh.mathSupport
    hijackedSupport.setDigitsPrecision(maxDigits)

    return hijackedSupport

def assignTimingsForMarkerList(markerList, params):
    # Stupid for now, but just to get running, let's just divide the spans
    # evenly across the duration
    framerate = float(params['project_params']['render_fps']) 
    framesAcrossDuration = getFramesForSecondsAndFramerate(params['millisecond_duration'] / 1000, framerate) 
    # Assign across one fewer frame duration than this is, because then the
    # endpoint will be ON the start of the last frame?
    framesAcrossDuration -= 1 
    timeForLastFrameStart = getDurationForFramesAndFramerate(framesAcrossDuration, framerate)
    return params['math_support'].createLinspace(0, timeForLastFrameStart, len(markerList)).astype('int')

def getFramesForSecondsAndFramerate(secondsDuration, framerate):
    """
    Adapted from from fractal.py->DiveTimeline->getFramesInDuration
    """
    # Use 6 decimal places for rounding down, but otherwise, round up.
    # 1 second *  24 frames / second = 24 frames
    # .041 seconds * 24 frames / second = .984 frames (should be 1)
    # .042 seconds * 24 frames / second = 1.008 frames (should be 2)
    # .041666 * 24 frames / second = .999984 frames (should be 1)
    # .04166666 * 24 frames / second = .99999984 frames (should be 1)
    # .04166667 * 24 frames / second = 1.00000008 frames (should be 1)
    # .0416667 * 24 frames / second = 1.0000008 frames (should be 2)
    rawCount = secondsDuration * framerate 
    framesCount = int(rawCount)
    remainderCount = rawCount - framesCount

    if remainderCount != 0.0 and round(remainderCount,6) != 0.0:
        framesCount += 1

    return framesCount

def getDurationForFramesAndFramerate(frames, framerate):
    secondsDuration = frames / framerate
    return int(1000 * secondsDuration)

if __name__ == '__main__':

    params = parse_options()

    markerList = loadMarkers(params)
    mathSupport = getMathSupportFromMarkerList(markerList)

    # Extra stashing of math support for easier use elsewhere
    params['math_support'] = mathSupport

    markerTimings = assignTimingsForMarkerList(markerList, params)

    # Only one algo, so just grab the first
    algorithmName = markerList[0].algorithmName

    project_params = params['project_params']

    timeline = DiveTimeline(projectFolderName=params['project_name'], algorithmName=algorithmName, framerate=project_params['render_fps'], frameWidth=project_params['render_image_width'], frameHeight=project_params['render_image_height'], mathSupport=mathSupport)

    span = timeline.addNewSpan(0, params['millisecond_duration'])
    for currIndex in range(len(markerList)):
        currTime = markerTimings[currIndex]
        currMarker = markerList[currIndex]
        currMesh = currMarker.diveMesh

        span.addNewParameterKeyframe(currTime, 'complex', 'meshCenter', currMesh.getCenter(), 'linear', 'linear')  
        span.addNewParameterKeyframe(currTime, 'float', 'meshRealWidth', currMesh.realMeshGenerator.baseWidth, 'root-to', 'root-to')
        span.addNewParameterKeyframe(currTime, 'float', 'meshImagWidth', currMesh.imagMeshGenerator.baseWidth, 'root-to', 'root-to')

        span.addNewParameterKeyframe(currTime, 'int', 'max_escape_iterations', currMarker.maxEscapeIterations, 'linear', 'linear')
        
    with open(params['timeline_file_name'], 'w') as outfile:
        json.dump(timeline.__getstate__(), outfile, indent=4)




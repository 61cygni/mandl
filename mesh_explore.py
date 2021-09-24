# --
# File: mesh_explore.py
#
# Viewer to explore results generated by fractal.py.
# Command-line parameters set how the image is generated.
#
# In addition, to achieve higher depths, you need to adjust 2 values
# as you explore:
# 1.) --max-escape-iterations
# 2.) mathSupport.setPrecision(numberOfDigits)
#
# If the image goes all white, it's likely you need higher iteration count.
# If the image goes splotchy, it's likely you need higher precision.
# If the image goes pixellated, then you definitely need higher precision.
#
# --

import getopt
import os
import pickle

import subprocess

from divemesh import *
from fractal import *
from fractalmath import *

import matplotlib as mpl
from matplotlib import pyplot as plt

def parse_options():
    params = {}
    opts, args = getopt.getopt(sys.argv[1:], "", 
                               ["math-support=",
                                "digits-precision=",
                                "escape-iterations=",
                                "project=",
                                "center=",
                                "next-frame-number=",
                                "real-width=",
                                "imag-width=",
                                "zoom-factor=",
                                "algo=",
                                # Special read-from-file inovcation
                                "load-marker-number=",
                                # Explicit extra params, should probably
                                # get/set these like in fractal.py
                                "julia-center=",
                                ])

    # First-pass at params pulls out the project file, because
    # it probably gives us a default math support
    for opt, arg in opts:
        if opt in ['--project']:
            params['project_name'] = arg
        if opt in ['--load-marker-number']:
            params['load_marker_number'] = arg

    # Require that a project has been specified.
    if 'project_name' not in params:
        raise ValueError("Specifying --project=<name> is required")

    # Load project parameters out of the params file
    paramFileName = os.path.join(params['project_name'], 'params.json')
    print(f"opening \"{paramFileName}\"")
    with open(paramFileName, 'rt') as paramHandle:
        params['project_params'] = json.load(paramHandle)

    # Second-pass at params to set up MathSupport because lots of 
    # things depend on it for names and types.
    mathSupportClasses = {'native': fm.DiveMathSupport,
        'mpfr': fm.DiveMathSupportMPFR,
        'flint': fm.DiveMathSupportFlint,
        'flintcustom': fm.DiveMathSupportFlintCustom,
        # maybe not complete?# 'gmp': fm.DiveMathSupportGmp,
        # maybe not complete?# 'decimal': fm.DiveMathSupportDecimal,
        # definitely not built yet.# 'libbf': fm.DiveMathSupportLibbf,
    } 

    # Before getting into normal param load, shortcut when we're asked to
    # load a marker file
    if 'load_marker_number' in params:
        markerNumber = params['load_marker_number']
        markerFileBase = f"{markerNumber}.marker.pik"
        markerFileName = os.path.join(params['project_name'], params['project_params']['exploration_markers_path'], markerFileBase)

        marker = None
        with open(markerFileName, 'rb') as markerHandle:
            marker = pickle.load(markerHandle)

        if marker == None:
            raise ValueError("Failed to load marker number \"{markerNumber}\"")

        diveMesh = marker.diveMesh 
        mathSupport = diveMesh.mathSupport
        params['math_support'] = mathSupport
        params['digits_precision'] = mathSupport.digitsPrecision()

        realCenter = diveMesh.realMeshGenerator.valuesCenter
        imagCenter = diveMesh.imagMeshGenerator.valuesCenter
        params['center'] = mathSupport.createComplex(realCenter, imagCenter)
        params['real_width'] = diveMesh.realMeshGenerator.baseWidth
        params['imag_width'] = diveMesh.imagMeshGenerator.baseWidth
        
        params['next_frame_number'] = marker.markerNumber
        params['escape_iterations'] = marker.maxEscapeIterations
        params['algo'] = marker.algorithmName

        # Guess we ignore the marker's mesh size, because we're using
        # the project param's exploration size?
    else:
        # 'Normal' params, not marker load
        mathSupportName = params['project_params'].get('math_support', 'native')

        for opt, arg in opts:
            if opt in ['--math-support']:
                if arg in mathSupportClasses:
                    mathSupportName = arg
            elif opt in ['--digits-precision']:
                params['digits_precision'] = int(arg)
        # Creates an instance
        mathSupport = mathSupportClasses[mathSupportName]() 

        # Important to also set expected precision before parsing param values
        supportPrecision = params.get('digits_precision', 16) # 16 == native
        # May have been defaulted, so (re)set the param
        params['digits_precision'] = supportPrecision 

        mathSupport.setPrecision(round(supportPrecision * 3.32)) # ~3.32 bits per position
        params['math_support'] = mathSupport

        # Defaults, overwritable by cmd line args
        params['algo'] = 'mandelbrot_smooth'
        params['next_frame_number'] = 1

        # Third pass at params, now that math support is all set up
        for opt, arg in opts:
            if opt in ['--center']:
                params['center'] = mathSupport.createComplex(arg)
            elif opt in ['--next-frame-number']:
                params['next_frame_number'] = int(arg)
            elif opt in ['--real-width']:
                params['real_width'] = mathSupport.createFloat(arg)
            elif opt in ['--imag-width']:
                params['imag_width'] = mathSupport.createFloat(arg)
            elif opt in ['--escape-iterations']:
                params['escape_iterations'] = int(arg)
            elif opt in ['--zoom-factor']:
                # No extra precision needed, but multiplied vs Decimal, so 
                # using native float, but have to convert when used.
                params['zoom_factor'] = float(arg) 
            elif opt in ['--algo']:
                params['algo'] = arg
            elif opt in ['--julia-center']:
                params['julia_center'] = mathSupport.createComplex(arg)

        escapeIterations = params.get('escape_iterations', 255)
        # May have been defaulted, so (re)set the param
        params['escape_iterations'] = escapeIterations

        # Heck - defaults for window widths and heights too, matched to the aspect
        # ratio of the project image.
        if 'real_width' not in params or 'imag_width' not in params:
            explorationWidth = mathSupport.createFloat(params['project_params'].get('exploration_mesh_width', 160.0))
            explorationHeight = mathSupport.createFloat(params['project_params'].get('exploration_mesh_height', 120.0))
            explorationAspect = explorationWidth / explorationHeight
            if 'real_width' not in params and 'imag_width' not in params:
                params['real_width'] = mathSupport.createFloat('3.0')
                params['imag_width'] = mathSupport.createFloat(params['real_width'] / explorationAspect)
            elif 'real_width' not in params:
                params['real_width'] = mathSupport.createFloat(params['imag_width'] * explorationAspect)
            else: # 'imag_width' not in params:
                params['imag_width'] = mathSupport.createFloat(params['real_width'] / explorationAspect)
   

    # Finally, params writing which should happen whether or not we loaded from a marker
    params['wholeCacheFolder'] = os.path.join(params['project_name'], params['project_params']['exploration_output_path']) 
   
    projectDefaultZoom = float(params['project_params'].get('exploration_default_zoom_factor', 0.8))
    params['zoom_factor'] = params.get('zoom_factor', projectDefaultZoom)
 
    return params

def nextClicked(event):
    global params
    sameTypeZoomFactor = params['math_support'].createFloat(params['zoom_factor'])
    params['real_width'] = params['real_width'] * sameTypeZoomFactor
    params['imag_width'] = params['imag_width'] * sameTypeZoomFactor

    updateView()
 
def prevClicked(event):
    global params
    sameTypeZoomFactor = params['math_support'].createFloat(params['zoom_factor'])
    params['real_width'] = params['real_width'] / sameTypeZoomFactor
    params['imag_width'] = params['imag_width'] / sameTypeZoomFactor

    updateView()

def updateView():
    global params
    global uiElements

    clickedLocus = uiElements.get('lastClickedLocus', None)
    if clickedLocus == None:
        return

    runFractalCallForCenter(clickedLocus)

    frameNumber = params['next_frame_number']

    imageFileTitle = "%d.tiff" % frameNumber
    imageFileName = os.path.join(params['wholeCacheFolder'], imageFileTitle)
    imageData = mpl.image.imread(imageFileName)

    clickableImage = uiElements['clickableImage']
    clickableImage.set_data(imageData)

    meshFileName = getMeshFileNameForFrameNumber(frameNumber)
    with open(meshFileName, 'rb') as meshHandle:
        diveMesh = pickle.load(meshHandle)

    uiElements['diveMesh'] = diveMesh

    updateTitle()
    plt.draw()

def runFractalCallForCenter(center):
    global params

    # TODO: remove the existing cached outputs of this name, if present

    projectName = params['project_name']
    mathSupport = params['math_support']
    algoName = params['algo']
    realWidthString = str(params['real_width'])
    imagWidthString = str(params['imag_width'])
    nextFrameNumber = params['next_frame_number']

    projectDefaultZoom = float(params['project_params'].get('exploration_default_zoom_factor', 0.8))
    zoomFactor = params.get('zoom_factor', projectDefaultZoom)

    # Shift to the new center - bit of string deviousness to prevent
    # parser errors when no imaginary component exists (flint trims the j?!)
    params['center'] = center
    centerString = str(center)
    if center.imag == 0:
        trimmed = centerString
        if centerString.endswith(')'):
            trimmed = centerString[:-1]
        if not trimmed.endswith('j'):
            trimmed = trimmed + "+0j"
        if centerString.endswith(')'):
            centerString = trimmed + ")"
        else:
            centerString = trimmed     

    fractalCallString = f"python3.9 ./fractal.py --project='{projectName}' --exploration --expl-algo={algoName} --burn --math-support={mathSupport.precisionType} --digits-precision={params['digits_precision']} --max-escape-iterations={params['escape_iterations']}  --expl-frame-number={nextFrameNumber} --expl-real-width='{realWidthString}' --expl-imag-width='{imagWidthString}' --expl-center='{centerString}'" 

    # Just hacked on for now...
    if algoName in ['julia_solo', 'julia_smooth']:
        juliaCenter = params['julia_center']
        juliaCenterString = str(juliaCenter)
        if juliaCenter.imag == 0:
            trimmed = juliaCenterString
            if juliaCenterString.endswith(')'):
                trimmed = juliaCenterString[:-1]
            if not trimmed.endswith('j'):
                trimmed = trimmed + "+0j"
            if juliaCenterString.endswith(')'):
                juliaCenterString = trimmed + ")"
            else:
                juliaCenterString = trimmed     
        fractalCallString += f" --julia-center='{juliaCenterString}'"  

    print("Calling: %s" % fractalCallString)
    subprocess.call([fractalCallString], shell=True)
    print("(Call finished.)")
    print("Exploration invocation for this point:")
    explorationCallString = f"python3.9 ./mesh_explore.py --project='{projectName}' --algo={algoName} --math-support={mathSupport.precisionType} --digits-precision={params['digits_precision']} --escape-iterations={params['escape_iterations']} --next-frame-number={nextFrameNumber} --real-width='{realWidthString}' --imag-width='{imagWidthString}' --zoom-factor={zoomFactor} --center='{centerString}'" 

    # Just hacked on for now...
    if algoName in ['julia_solo', 'julia_smooth']:
        explorationCallString += f" --julia-center='{juliaCenterString}'"  

    print(explorationCallString)
    print("")

def lastMarkerClicked(event):
    pass


def refreshClicked(event):
    updateView()

def saveMarkerClicked(event):
    global uiElements

    diveMesh = uiElements['diveMesh']
    currFrameNumber = params['next_frame_number']

    marker = MeshMarker(diveMesh, currFrameNumber, params['algo'], params['escape_iterations'])
 
    markerFileName = getMarkerFileNameForFrameNumber(currFrameNumber)
    with open(markerFileName, 'wb') as markerHandle:
        pickle.dump(marker, markerHandle)

    params['next_frame_number'] += 1

    # Not really ideal to rebuild the image, but it *is* a good way
    # to get the frame increment burned in, and to show that an
    # increment/save happened.
    updateView()

def plusPrecisionClicked(event):
    global params
    params['digits_precision'] += 1
    updatePrecisionText() 
    plt.draw()

def minusPrecisionClicked(event):
    global params
    params['digits_precision'] -= 1
    if params['digits_precision'] < 1:
        params['digits_precision'] = 1
    updatePrecisionText() 
    plt.draw()

def updatePrecisionText():
    global uiElements
    global params
    screenText = uiElements['precisionText']
    screenText.set_text(str(params['digits_precision']))

def plusIterationsClicked(event):
    global params
    params['escape_iterations'] += 16
    updateIterationsText() 
    plt.draw()

def minusIterationsClicked(event):
    global params
    params['escape_iterations'] -= 16
    if params['escape_iterations'] < 16:
        params['escape_iterations'] = 16
    updateIterationsText() 
    plt.draw()

def updateIterationsText():
    global uiElements
    global params
    screenText = uiElements['iterationsText']
    screenText.set_text(str(params['escape_iterations']))

def plusClicked(event):
    global params
    params['zoom_factor'] = round(params['zoom_factor'] + .01, 2)
    updateAdvanceText() 
    plt.draw()

def minusClicked(event):
    global params
    params['zoom_factor'] = round(params['zoom_factor'] - .01, 2)
    if params['zoom_factor'] < .01:
        params['zoom_factor'] = .01
    updateAdvanceText() 
    plt.draw()

def updateAdvanceText():
    global uiElements
    global params
    screenText = uiElements['advanceText']
    screenText.set_text(str(params['zoom_factor']))

def updateTitle():
    global uiElements
    diveMesh = uiElements['diveMesh']

    widthString = diveMesh.mathSupport.shorterStringFromFloat(diveMesh.realMeshGenerator.baseWidth, 10)
    plt.suptitle("%s wide" % widthString)

def onclick(event):
    global uiElements

    # event.xdata and event.ydata are floats, but we want pixel ints
    #print(event)
    if event.xdata is None or event.ydata is None:
        return

    # Only clicks in the image can change the focus.
    clickableImage = uiElements['clickableImage']
    if event.inaxes != clickableImage.axes:
        return

    clickX = int(event.xdata)
    clickY = int(event.ydata)

    diveMesh = uiElements['diveMesh']
    meshData = diveMesh.generateMesh()
    #print("click (%s,%s)" % (str(clickX), str(clickY)))
    clickedLocus = meshData[clickY, clickX]
    # Extra steps to try to clear out the 'error' radius from arb.
    clickedRealString = diveMesh.mathSupport.stringFromFloat(clickedLocus.real)
    clickedImagString = diveMesh.mathSupport.stringFromFloat(clickedLocus.imag)
    rebuiltClickedLocus = diveMesh.mathSupport.createComplex(clickedRealString, clickedImagString)
    print(rebuiltClickedLocus)
    uiElements['lastClickedLocus'] = rebuiltClickedLocus

def getMeshFileNameForFrameNumber(frameNumber):
    global params

    meshFileTitle = "%d.mesh.pik" % frameNumber
    return os.path.join(params['wholeCacheFolder'], meshFileTitle)

def getMarkerFileNameForFrameNumber(frameNumber):
    global params

    meshFileTitle = "%d.marker.pik" % frameNumber
    return os.path.join(params['project_name'], params['project_params']['exploration_markers_path'], meshFileTitle)

if __name__ == '__main__':

    global params
    params = parse_options()

    frameNumber = params.get('next_frame_number', 1)
    center = params.get('center', params['math_support'].createComplex("(-1.0+0j)"))
    
    # Run the whole frame on first call, so we can read the
    # mesh dimensions from the result.  Otherwise, we have to
    # guess at a lot of things like image size.
    runFractalCallForCenter(center)

    global uiElements
    uiElements = {}

    # Start off with the parameterized locus as the center.
    uiElements['lastClickedLocus'] = center

    imageFileTitle = "%d.tiff" % frameNumber
    meshFileName = getMeshFileNameForFrameNumber(frameNumber)

    with open(meshFileName, 'rb') as meshHandle:
        diveMesh = pickle.load(meshHandle)

    uiElements['diveMesh'] = diveMesh
    
    mainFigure = plt.figure()
    uiElements['mainFigure'] = mainFigure

    imageFileName = os.path.join(params['wholeCacheFolder'], imageFileTitle)
    imageData = mpl.image.imread(imageFileName)
    uiElements['clickableImage'] = plt.imshow(imageData)

    mainFigure.canvas.mpl_connect('button_press_event', onclick)

    buttonHeight = 0.05
    largerButtonWidth = 0.08
    smallerButtonWidth = .04
    buttonWidth = smallerButtonWidth

    gutter = 0.01
    positionX = gutter
    positionY = gutter

    ####
    # NOTE: 
    # When adding elements (Buttons OR Text) , you CAN'T reuse the
    # capturing variable name, because the assignment apparently 
    # uses an internal reference.
    ####

    button1Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button1Axes, label="|<")
    button.on_clicked(lastMarkerClicked)
    uiElements['lastMarkerButton'] = button

    buttonRedAxes = plt.axes([positionX,positionY+ buttonHeight + gutter,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(buttonRedAxes, label="re")
    button.on_clicked(refreshClicked)
    uiElements['refreshButton'] = button
   
    positionX += buttonWidth + gutter

    buttonWidth = largerButtonWidth

    button2Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button2Axes, label="+Prec")
    button.on_clicked(plusPrecisionClicked)
    uiElements['plusPrecisionButton'] = button
   
    positionX += buttonWidth + gutter

    button3Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button3Axes, label="-Prec")
    button.on_clicked(minusPrecisionClicked)
    uiElements['minusPrecisionButton'] = button
   
    positionX += buttonWidth + gutter

    screenText1 = plt.text(positionX + .8, positionY, "(00)", horizontalalignment='left')
    uiElements['precisionText'] = screenText1

    positionX += 0.05

    button4Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button4Axes, label="+Iter")
    button.on_clicked(plusIterationsClicked)
    uiElements['plusIterationsButton'] = button

    positionX += buttonWidth + gutter

    button5Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button5Axes, label="-Iter")
    button.on_clicked(minusIterationsClicked)
    uiElements['minusIterationsButton'] = button
   
    positionX += buttonWidth + gutter

    screenText2 = plt.text(positionX + .6, positionY, "(00)", horizontalalignment='left')
    uiElements['iterationsText'] = screenText2
  
    positionX += 0.06

    button6Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button6Axes, label="Out")
    button.on_clicked(prevClicked)
    uiElements['previousButton'] = button
   
    positionX += buttonWidth + gutter

    button7Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button7Axes, label="In")
    button.on_clicked(nextClicked)
    uiElements['nextButton'] = button

    positionX += buttonWidth + gutter

    button8Axes = plt.axes([positionX,positionY,buttonWidth,buttonHeight])
    button = mpl.widgets.Button(button8Axes, label="Save")
    button.on_clicked(saveMarkerClicked)
    uiElements['saveMarkerButton'] = button

    positionX += buttonWidth + gutter

    buttonWidth = smallerButtonWidth

    button9Axes = plt.axes([positionX,positionY,smallerButtonWidth,buttonHeight])
    button = mpl.widgets.Button(button9Axes, label="+")
    button.on_clicked(plusClicked)
    uiElements['plusButton'] = button

    positionX += smallerButtonWidth + gutter

    button10Axes = plt.axes([positionX,positionY,smallerButtonWidth,buttonHeight])
    button = mpl.widgets.Button(button10Axes, label="-")
    button.on_clicked(minusClicked)
    uiElements['minusButton'] = button

    positionX += smallerButtonWidth + gutter

    screenText3 = plt.text(positionX + .1, positionY, "(scale)", horizontalalignment='left')
    uiElements['advanceText'] = screenText3

    updatePrecisionText()
    updateIterationsText()
    updateAdvanceText()

    updateTitle()

    plt.show()


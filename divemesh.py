# --
# File: divemesh.py
# 
# For a specific frame/epoch, the mesh is the plane across complex 
# space (real+imaginary) that we're computing fractals on.
#
# Overview of mesh classes
# ------------------------
#    DiveMesh
#    MeshGenerator
#    - MeshGeneratorUniform
#    - MeshGeneratorTilt
#
# --

import numpy as np

import fractalmath as fm # Technically not required, but definitely needed 

class DiveMesh:
    """
    A DiveMesh is a 2D array of complex numbers that serves as a basis for
    calculation, with the x-axis (across the width) based on real component 
    distribution, and y-axis (across the height) based on imaginary 
    component distribution.  Conceptually, this is the mapping of a portion 
    of complex space into a 2D "imaging"(?) plane.  
    
    Two separate 2D (real-valued) meshes are generated first, one for the real component, 
    and one for the imaginary component.  This means range and distribution types of the 
    component meshes are the highest priority parameters.

    # Not Implemented:
    # Next,  distortions are applied separately to the real mesh, and to the imaginary mesh.
    # Then, the separate 2D meshes are combined to become the overall mesh values.
    # Finally, overall distortions are applied to the overall mesh values.
    """
    def __init__(self, width, height, realMeshGenerator, imagMeshGenerator, mathSupport, extraParams={}):
        # Trying not to apply castings to these types, to keep them the same as
        # the original parameters, which could make swapping out different 
        # precision libraries simpler?
        self.meshWidth = width
        self.meshHeight = height

        self.realMeshGenerator = realMeshGenerator
        self.imagMeshGenerator = imagMeshGenerator

        #self.realMeshDistortions = []
        #self.imagMeshDistortions = []

        #self.meshDistortions = []

        # Technically, it seems redundant to have mathSupport specified separately for MeshGenerators
        # and for DiveMesh, but it's enough of a chicken-and-egg problem that I'll just
        # pass an extra parameter here and there.
        self.mathSupport = mathSupport

        # Currently, extraParams is NOT pickled
        # because we don't know which types the values are
        self.extraParams = extraParams 

    def __getstate__(self):
        pickleInfo = self.__dict__.copy() 
        # Currently, extraParams is NOT pickled
        # because we don't know which types the values are
        if 'extraParams' in pickleInfo:
            del(pickleInfo['extraParams'])

        # Going to encode both the class name of the MathSupport, and
        # the 'precision' it was apparently set at, making a string
        # like "DiveMathSupportFlint:2048".
        mathSupportString = type(self.mathSupport).__name__ + ":" + str(self.mathSupport.precision())
        pickleInfo['mathSupport'] = mathSupportString

        return pickleInfo

    def __setstate__(self, state):
        """ 
        NOTE: A new MathSupport sublass is instantiated during un-pickling.
        This can be a problem if you're relying on specific precision settings, 
        because the only MathSupport configuration that's handled here
        is setting the precision from the encoded 'classname:precision' string.

        It *is* important to set the precision before loading any numbers, or
        else the numbers may be clipped to lower precision than what they 
        were saved at.
        """
        mathSupportClasses = {"DiveMathSupportFlintCustom":fm.DiveMathSupportFlintCustom,
                "DiveMathSupportFlint":fm.DiveMathSupportFlint,
                "DiveMathSupport":fm.DiveMathSupport}

        (mathSupportClassName, precisionString) = state['mathSupport'].split(':')
        #print("mathSupport reads as: %s" % mathSupportClassName)
 
        state['mathSupport'] = mathSupportClasses[mathSupportClassName]()
        state['mathSupport'].setPrecision(int(precisionString))
        #print("mathSupport is: %s" % str(state['mathSupport']))

        self.__dict__.update(state)

    def getCenter(self):
        """ Pretty common to want the mesh center, so assemble it from the generators. """
        return self.mathSupport.createComplex(self.realMeshGenerator.valuesCenter, self.imagMeshGenerator.valuesCenter)

    def generateMesh(self):
        #print(f"Mesh math support is \"{self.mathSupport.precisionType}\" at {self.mathSupport.digitsPrecision()} digits and {self.mathSupport.precision()} bits")
        realMesh = self.realMeshGenerator.generateForDiveMesh(self)
        imagMesh = self.imagMeshGenerator.generateForDiveMesh(self)

        if realMesh.shape != imagMesh.shape:
            raise ValueError("Real sub-mesh (%s) and Imaginary sub-mesh (%s) shapes don't match." % (realMesh.shape, imagMesh.shape))

        meshShape = realMesh.shape
        # The native python 'complex' type assigns into "object" type arrays 
        # without problems, but not vice-versa, so use object type for everything.
        combinedMesh = np.zeros(meshShape, dtype=object) 
        # numpyArray.shape returns (rows, columns)
        for y in range(0, meshShape[0]):
            for x in range(0, meshShape[1]):
                combinedMesh[y,x] = self.mathSupport.createComplex(realMesh[y,x], imagMesh[y,x])

        #print("mesh:")
        #print(combinedMesh)

        return combinedMesh

    def isUniform(self):
        return self.realMeshGenerator and isinstance(self.realMeshGenerator, MeshGeneratorUniform) and self.imagMeshGenerator and isinstance(self.imagMeshGenerator, MeshGeneratorUniform)

    def __repr__(self):
        return """\
[DiveMesh {{{mwidth},{mheight}}} realGenerator:{rgen} imagGenerator:{igen} ]\
""".format(mwidth=self.meshWidth, mheight=self.meshHeight, rgen=str(self.realMeshGenerator), igen=str(self.imagMeshGenerator))

class MeshGenerator:
    def __init__(self, mathSupport, varyingAxis):
        self.mathSupport = mathSupport

        axisOptions = ['width', 'height']
        if varyingAxis not in axisOptions:
            raise ValueError("varyingAxis must be one of (%s)" % ", ".join(axisOptions))
        self.varyingAxis = varyingAxis

    def __getstate__(self):
        pickleInfo = self.__dict__.copy() 
        # Going to encode both the class name of the MathSupport, and
        # the 'precision' it was apparently set at, making a string
        # like "DiveMathSupportFlint:2048".
        mathSupportString = type(self.mathSupport).__name__ + ":" + str(self.mathSupport.precision())
        pickleInfo['mathSupport'] = mathSupportString
       
        return pickleInfo

    def __setstate__(self, state):
        """ 
        NOTE: A new MathSupport sublass is instantiated during un-pickling.
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
 
        state['mathSupport'] = mathSupportClasses[mathSupportClassName]()
        state['mathSupport'].setPrecision(int(precisionString))
        #print("mathSupport is: %s" % str(state['mathSupport']))
        self.__dict__.update(state)

    def generateForDiveMesh(self):
        raise NotImplementedError("generateForDiveMesh() must be overridden in a MeshGenerator subclass")

class MeshGeneratorUniform(MeshGenerator):
    """
    Generates a 2D mesh of values distributed across with width, along the axis specified.

    DiveMesh is used for calculations of ranges, so the DiveMesh is responsible for 
    instantiation of correct types.  In other words, we try to avoid doing instantiations here.

    Technically, should probably make the DiveMesh responsible for valuesCenter, so there's a single
    point of reference.  It just seemed less clear that way at the moment.
    """
    def __init__(self, mathSupport, varyingAxis, valuesCenter, baseWidth):
        super().__init__(mathSupport, varyingAxis)
        #print(f"making MeshGeneratorUniform with {mathSupport.digitsPrecision()} at center {valuesCenter}")

        self.valuesCenter = valuesCenter
        self.baseWidth = baseWidth

    def __getstate__(self):
        pickleInfo = MeshGenerator.__getstate__(self)

        pickleInfo['valuesCenter'] = str(pickleInfo['valuesCenter'])
        pickleInfo['baseWidth'] = str(pickleInfo['baseWidth'])

        return pickleInfo

    def __setstate__(self, state):
        super().__setstate__(state)
        #print("mathSupport exists as: %s" % str(self.mathSupport))
        #print(f"__setstate__ math support is \"{self.mathSupport.precisionType}\" at {self.mathSupport.digitsPrecision()} digits and {self.mathSupport.precision()} bits")
        # Should probably be updating __dict__ instead?
        self.valuesCenter = self.mathSupport.createFloat(self.valuesCenter)
        self.baseWidth = self.mathSupport.createFloat(self.baseWidth)

    def generateForDiveMesh(self, diveMesh):
        """
        e.g.
        diveMesh.meshWidth=3, diveMesh.meshHeight=2, self.varyingAxis='width', self.valuesCenter=1.0, baseWidth=.2 returns:
        [[0.9, 1.0, 1.1],
         [0.9, 1.0, 1.1]]
    
        diveMesh.meshWidth=3, diveMesh.meshHeight=2, self.varyingAxis='height', self.valuesCenter=1.0, baseWidth=.2 returns:
        [[0.9, 0.9, 0.9],
         [1.1, 1.1, 1.1]]
        """
        mesh = np.zeros((diveMesh.meshHeight, diveMesh.meshWidth), dtype=object)

        #print(f"generator math support is \"{self.mathSupport.precisionType}\" at {self.mathSupport.digitsPrecision()} digits and {self.mathSupport.precision()} bits")

        if self.varyingAxis == 'width':
            #calculate start/end...  Probably need to be subtype aware for this...
            discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, self.baseWidth, diveMesh.meshWidth)
            #print(f"W center: {self.valuesCenter} baseWidth: {str(self.baseWidth)} meshWidth: {str(diveMesh.meshWidth)}")
            #print("W: %s" % discretizedValues)

            mesh[0:] = discretizedValues # Assign the one-row discretization to every row of the mesh
        else: # self.varyingAxis == 'height'
            discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, self.baseWidth, diveMesh.meshHeight)
            #print(f"H center: {self.valuesCenter} baseWidth: {str(self.baseWidth)} meshHeight: {str(diveMesh.meshHeight)}")
            #print("H: %s" % discretizedValues)
            # Assign the one-row discretization (as a column) to every column of the mesh
            mesh[0:] = discretizedValues[:,np.newaxis] 

        return mesh

    def __repr__(self):
        return """\
[MeshGeneratorUniform valuesCenter:{vCenter} baseWidth:{vWidth} along axis:{vAxis}]\
""".format(vCenter=self.valuesCenter, vWidth=self.baseWidth, vAxis=self.varyingAxis)

# Not implemented yet: 
# MeshGeneratorLogTilt, which uses a log scaling anchored at the middle
# Maybe also:
# DiveMeshGeneratorSqueeze (base-width->different-middle-width->base-width)
# Mostly, looking for effects that give me control over something that feels
# like camera behavior, such as lens barrel distortion.

class MeshGeneratorTilt(MeshGenerator):
    def __init__(self, mathSupport, varyingAxis, valuesCenter, baseWidth, tiltAngle, tiltMaxRatio=10.0):
        """
        Tilt is symmetric about the baseWidth.

        Using an "angle" here isn't really correct, but does seem to be an
        intuitive and controllable analog, which allows an interpolation 
        across zero.  With most (simpler) direct multiplication approaches,
        the range from {-1.0,1.0} does strange things with continutity.

        Ranges from {-45,45} are acceptable (for now?).  Didn't go with more
        'ordinary' ranges like 90 or 180, partly as a reminder that this
        deviation isn't actually an angle, but instead maps a number across
        the range defined by tiltMaxRatio.
        
        The tiltMaxRatio is the most that the first/last line is allowed to 
        differ from the baseWidth. 
       
        Originally thought this would need 2 axis parameters, but for 
        now, requiring the tilt axis to be the same as varying axis seems 
        to make sense.

        Because the tilt factor is spread across the mesh rows, there might be
        some strong aliasing (across dive frames) if there's an even number 
        of rows or columns? I could imagine a back-and-forth wiggle developing 
        if the scales and factors line up the right way.

        Axis discretization happens for every frame, on 2 axes
        (across complex range, and across real range).
        Mesh generation is the combination of these 2 axes 
        (perhaps plus further post-processing modifications).

        A stretched axis (wider range), compared to the current frame's 
        baseline, is akin to calculating previous steps.  For example, 
        if you happen to stretch the axis as much as the previous frame 
        transition's zoom factor, then you're sorta recalculating the 
        previous frame's axis again.
        Similarly, a squished axis (narrower range), is akin to calculating 
        future steps.
        """
        super().__init__(mathSupport, varyingAxis)
        self.valuesCenter = valuesCenter
        self.baseWidth = baseWidth
        self.maxAbsTiltAngle = 45
        self.tiltAngle = tiltAngle

        self.tiltMaxRatio = tiltMaxRatio 
        self.safetyCheckTiltMaxRatio()

    def __getstate__(self):
        pickleInfo = MeshGenerator.__getstate__(self)

        pickleInfo['valuesCenter'] = str(pickleInfo['valuesCenter'])
        pickleInfo['baseWidth'] = str(pickleInfo['baseWidth'])

        return pickleInfo

    def __setstate__(self, state):
        super().__setstate__(state)
        #print("mathSupport exists as: %s" % str(self.mathSupport))
        #print(f"__setstate__ math support is \"{self.mathSupport.precisionType}\" at {self.mathSupport.digitsPrecision()} digits and {self.mathSupport.precision()} bits")
        # Should probably be updating __dict__ instead?
        self.valuesCenter = self.mathSupport.createFloat(self.valuesCenter)
        self.baseWidth = self.mathSupport.createFloat(self.baseWidth)
 
    def clampMaxAngle(self, paramAngle):
        answerAngle = paramAngle
        if answerAngle > self.maxAbsTiltAngle:
            answerAngle = maxAbsTiltAngle
        elif abs(answerAngle) > self.maxAbsTiltAngle:
            answerAngle = -1 * self.maxAbsTiltAngle
        return answerAngle

    def safetyCheckTiltMaxRatio(self):
        """
        Can't be between {-1.0,1.0}, and we use abs() later, so 
        causes an error when max ratio is less than 1.0.
        """
        if self.tiltMaxRatio < 1.0:
            raise ValueError("ERROR - MeshGeneratorTilt can't have tiltMaxRatio under 1.0.")
        
    def getStartAndEndWidths(self):
        # Calculating only one side from the tiltFactor, then using the delta from
        # the original as the other side's size.  This keeps our center in the linear
        # center of the ranges, instead of shifting it some amount dependent on the factor.

        # TODO: pretty sure this star/tend calculation needs to have its math done by 
        # the MathSupport too, to keep type requirements localized there?

        self.safetyCheckTiltMaxRatio()

        safeAngle = self.clampMaxAngle(self.tiltAngle) 
        print(f"tilt {self.tiltAngle} => safe {safeAngle}")

        if abs(safeAngle) < 0.01:
            # Effectively zero, or probably should be?
            return (self.baseWidth, self.baseWidth)

        anglePercent = safeAngle / self.maxAbsTiltAngle
        tiltRatioRange = self.tiltMaxRatio - 1.0

        print(f"anglePercent {anglePercent}\ntiltRatioRange {tiltRatioRange}")

        if anglePercent > 0.0:
            tiltRatio = 1.0 + (tiltRatioRange * anglePercent)
            print(f"positive tiltRatio {tiltRatio}")
            #startWidth = 1.0 / tiltRatio * self.baseWidth        
            startWidth = tiltRatio * self.baseWidth        
            startDelta = self.baseWidth - startWidth
            print(f"positive startDelta {startDelta}")
            endWidth = self.baseWidth + startDelta
            return (startWidth, endWidth) 
        else:
            tiltRatio = 1.0 + (tiltRatioRange * abs(anglePercent))
            print(f"negative tiltRatio {tiltRatio}")
            endWidth = tiltRatio * self.baseWidth
            endDelta = self.baseWidth - endWidth
            print(f"negative endDelta {endDelta}")
            startWidth = self.baseWidth + endDelta
            return (startWidth, endWidth) 

#        if anglePercent > 0.0:
#            tiltRatio = 1.0 + (tiltRatioRange * anglePercent)
#            print(f"positive tiltRatio {tiltRatio}")
#            #startWidth = 1.0 / tiltRatio * self.baseWidth        
#            startWidth = 1.0 / tiltRatio * self.baseWidth        
#            startDelta = self.baseWidth - startWidth
#            print(f"positive startDelta {startDelta}")
#            endWidth = self.baseWidth + startDelta
#            return (startWidth, endWidth) 
#        else:
#            tiltRatio = (tiltRatioRange * anglePercent)
#            print(f"negative tiltRatio {tiltRatio}")
#            endWidth = 1.0 / tiltRatio * -1.0 * self.baseWidth
#            endDelta = self.baseWidth - endWidth
#            print(f"negative endDelta {endDelta}")
#            startWidth = self.baseWidth + endDelta
#            return (startWidth, endWidth) 

# Old version, right enough logic         
#        startWidth = 1.0 / self.tiltFactor * self.baseWidth
#        startDelta = self.baseWidth - startWidth
#        endWidth = self.baseWidth + startDelta
#        if self.tiltFactor < 0.0:
#            endWidth = 1.0 / self.tiltFactor * -1 * self.baseWidth
#            endDelta = self.baseWidth - endWidth
#            startWidth = self.baseWidth + endDelta
        

    def generateForDiveMesh(self, diveMesh):
        mesh = np.zeros((diveMesh.meshHeight, diveMesh.meshWidth), dtype=object)

        (startWidth, endWidth) = self.getStartAndEndWidths()
        print(f"mesh startWidth: {startWidth} endWidth: {endWidth}")

        if self.varyingAxis == 'width':
            # Values vary along the width axis, and the tiltFactor is applied to the
            # range of each row, which effectively treats the width axis as the 
            # rotation point
            meshRowBaseWidths = self.mathSupport.createLinspace(startWidth, endWidth, diveMesh.meshHeight)
            for y in range(0, diveMesh.meshHeight):
                #calculate start/end...  Probably need to be subtype aware for this...
                discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, meshRowBaseWidths[y], diveMesh.meshWidth)
                mesh[y] = discretizedValues # Assign the dscretization to this row
        else: # self.varyingAxis == 'height'
            meshColBaseWidths = self.mathSupport.createLinspace(startWidth, endWidth, diveMesh.meshWidth)
            for x in range(0, diveMesh.meshWidth):
                discretizedValues = self.mathSupport.createLinspaceAroundValuesCenter(self.valuesCenter, meshColBaseWidths[x], diveMesh.meshHeight)
                mesh[:,x] = discretizedValues # Assign the discretization (as a column)

        return mesh

    def __repr__(self):
        return """\
[MeshGeneratorTilt center:{vCenter} baseWidth:{vWidth} tiltFactor:{tilt} along axis:'{vAxis}']\
""".format(vCenter=self.valuesCenter, vWidth=self.baseWidth, tilt=self.tiltFactor, vAxis=self.varyingAxis)

class MeshMarker:
    """
    Rather than overloading DiveMesh's extra_params, or stashing stuff
    in the mesh, this thin-ish wrapper is intended to hold the 'extra' 
    values and variables from exploration/layout so we can keep 
    everything persistent in one area.
    """
    def __init__(self, diveMesh, markerNumber, algorithmName, maxEscapeIterations):
        self.diveMesh = diveMesh
        self.markerNumber = markerNumber
        self.algorithmName = algorithmName
        self.maxEscapeIterations = maxEscapeIterations


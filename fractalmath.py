# --
# File: fractalmath.py
#
# Home for math operations that should be high-precision aware
#
# --

import math

import numpy as np

#FLINT_HIGH_PRECISION_SIZE = 16 # 53 is how many bits are in float64
#FLINT_HIGH_PRECISION_SIZE = 53 # 53 is how many bits are in float64
#3.32 bits per digit, on average
#2200 was therefore, ~662 digits, got 54 frames down at .5 scaling

#FLINT_HIGH_PRECISION_SIZE = int(2800 * 3.32) # 2200*3.32 = 7304, lol
#FLINT_HIGH_PRECISION_SIZE = int(2200 * 3.32) # 2200*3.32 = 7304, lol
#FLINT_HIGH_PRECISION_SIZE = int(1800 * 3.32) # 2200*3.32 = 7304, lol
FLINT_HIGH_PRECISION_SIZE = int(1125 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(500 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(400 * 3.32) 

# For debugging, looks like we're bottoming out somewhere around e-11
# So, only really need ~20 digits for this test
# Blowing out frame 30, which is ~28?
#FLINT_HIGH_PRECISION_SIZE = int(50 * 3.32) 
# Native blew out somewhere e-13 to e-15
#FLINT_HIGH_PRECISION_SIZE = int(200 * 3.32) 


GMP_HIGH_PRECISION_SIZE=53

class DiveMathSupport:
    """
    Toolbox for math functions that need to be type-aware, so we
    can support different back-end math libraries.

    This base class is for native python calculations.

    This exists because globally swapping out types doesn't do enough
    to keep numeric types all in line. Some calculations also need additional 
    steps to make the math behave right, such as if we were using 
    the Decimal library, and had to keep separate real and imaginary
    components of a complex number ourselves.

    Trying to preserve types where possible, without forcing casting, because sometimes
    all the math operations will already work for custom numeric types.
    """
    def __init__(self):
        self.precisionType = 'native'

    def createComplex(self, *args):
        """
        Compatible complex types will return values for .real() and .imag()

        Native complex type just passes on all params to complex()
        """
        # Can't make a native complex from 2 strings though, so when we
        # detect that, jam them into floats
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            return complex(float(args[0]), float(args[1]))
        else:
            return complex(*args) 

    def createFloat(self, floatValue):
        return float(floatValue)

    def floor(self, value):
        return math.floor(value)

    def scaleValueByFactorForIterations(self, startValue, overallZoomFactor, iterations):
        """
        Shortcut calculate the starting point for the last frame's properties, 
        which we'll use for instantiation with specific widths.  This is
        because if any keyframes are added, the best we can do is measure an
        effective zoom factor between any 2 frames.

        Note: Flint's __pow__ needs TWO arb types as args, or else it drops 
        to default python bit depths
        """
        return startValue * (overallZoomFactor ** iterations)

    def createLinspace(self, paramFirst, paramLast, quantity):
        """Attempt at a mostly type-agnostic linspace(), seems to work with flint types too"""
        dataRange = paramLast - paramFirst
        answers = np.zeros((quantity), dtype=object)

        for x in range(0, quantity):
            answers[x] = paramFirst + dataRange * (x / (quantity - 1))

        return answers

    def createLinspaceAroundValuesCenter(self, valuesCenter, spreadWidth, quantity):
        """
        Attempt at a mostly type-agnostic linspace-from-center-and-width

        Turns out, we didn't *really* need this for using flint, though it
        was an important for a Python/Decimal version to be able to have a
        type-specific implementation in what was a DiveMeshDecimal function
        """
        return self.createLinspace(valuesCenter - spreadWidth * 0.5, valuesCenter + spreadWidth * 0.5, quantity)

    def interpolate(self, transitionType, startX, startY, endX, endY, targetX, extraParams={}):
        """
        Resisted having a single call point, in case more parameters
        were needed for each type, but we might just be able to pass
        in an extra param hash?
        """
        if transitionType == 'log-to':
            return self.interpolateLogTo(startX, startY, endX, endY, targetX)
        elif transitionType == 'root-to':
            return self.interpolateRootTo(startX, startY, endX, endY, targetX)
        elif transitionType == 'linear':
            return self.interpolateLinear(startX, startY, endX, endY, targetX)
        elif transitionType == 'quadratic-to':
            return self.interpolateQuadraticEaseOut(startX, startY, endX, endY, targetX)
        elif transitionType == 'quadratic-from':
            return self.interpolateQuadraticEaseIn(startX, startY, endX, endY, targetX)
        else: # transitionType == 'quadratic-to-from'
            return self.interpolateQuadraticEaseInOut(startX, startY, endX, endY, targetX)

    def interpolateLogTo(self, startX, startY, endX, endY, targetX):
        """
        Probably want additional log-defining params, but for now, let's just bake in one equation
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            # y1 = a * ln(b * x1)
            # y2 = a * ln(b * x2)
            # a = (y1-y2) / ln(x1/x2)
            # b = exp(((y2 * ln(x1)) - (y1 * ln(x2))) / (y1-y2))
            #
            # y = a * ln(b * x)
            # answer = a * ln(b * targetX)
            # answer = ((y1-y2)/ln(x1/x2)) * ln(exp(((y2 * ln(x1)) - (y1 * ln(x2))) / (y1-y2)) * targetX)
            # This kinda worked, but keeps the ln function so you just see part of 
            # some curve through the query points.  Really, want to make sure the asymptote 
            # is at or near the start (or end).
            # aVal = ((startY-endY) / math.log(startX / endX))
            # bVal = math.exp(startY / aVal) / startX
            # return aVal * math.log(bVal * targetX)

            # This version seems like it worked fine, but ended up with a pretty wrong
            # transition
            #
            # But, if we set the crossover point to the first point (add 1 to xVal), then all we
            # need to do is solve for a, right?  Also, using the 0-crossover as the
            # first point means the asymptote can't be hit, because it's 1 less than 
            # the start.
            #
            # endY = a * ln(endX + 1 - startX) + startY
            # a = (endY - startY) / ln(endX - startX + 1)

            aVal = (endY - startY) / math.log(endX - startX + 1)
            return aVal * math.log(targetX - startX + 1) + startY 

    def interpolateRootTo(self, startX, startY, endX, endY, targetX):
        """ 
        Iterative multiplications of window sizes for zooming means 
        we want to be able to interpolate between two points using
        the frame count as the root.
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            root  = endX - startX
            scaleFactor = (endY / startY) ** (1 / root)
            return startY * (scaleFactor ** targetX)

    def interpolateQuadraticEaseIn(self, startX, startY, endX, endY, targetX):
        """
        QuadraticEaseIn puts the majority of changes in the start of the X range.
        
        I *think* this is just the backwards solution to QuadraticEaseOut?  
        So, just swap in and out points?

        Probably want additional quadratic-defining params, but for now, let's just bake in one equation
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            return self.interpolateQuadraticEaseOut(endX, endY, startX, startY, targetX)

    def interpolateQuadraticEaseOut(self, startX, startY, endX, endY, targetX):
        """
        QuadraticEaseOut leaves the majority of changes to the end of the X range.

        Probably want additional quadratic params, but for now, let's just bake in one equation
        which uses the first point as the vertex, and passes through the second point.
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            # Find a, given that the start point is the vertex, and the parabola passes 
            # through the other point
            # y = a * (x - h)**2 + k
            # y = a * (x - startX)**2 + startY
            # endY = a * (endX - startX)**2 + startY
            # a = ((endY-startY)/((endX-startX)**2)
            #
            # answer = a * (targetX - startX)**2 + startY
            # answer = ((endY-startY)/((endX-startX)**2) * ((targetX-startX)**2) + startY
            return (endY-startY)/((endX-startX)**2) * ((targetX-startX)**2) + startY

    def interpolateQuadraticEaseInOut(self, startX, startY, endX, endY, targetX):
        """
        QuadraticEaseInOut finds a (linear) midpoint of the range, then leaves the 
        majority of the change to the middle of the range, easing both out of 
        the start, and into the end along separate parabolas.

        CAUTION: This forces the 'middle' X value to floor().  The idea is that X 
        will be frame numbers, so it should gracefully handle the inflection 
        around an integer-valued frame, instead of doing a (maybe more 
        complicated?) rounding inference.  
        The problem is that this means it isn't a universal solver.
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            midpointFrame = self.floor(((endX-startX) * .5)) + startX
            midpointY = self.interpolateLinear(startX, startY, endX, endY, midpointFrame) 
            if targetX <= midpointFrame:
                return self.interpolateQuadraticEaseOut(startX, startY, midpointFrame, midpointY, targetX)
            else:
                return self.interpolateQuadraticEaseIn(midpointFrame, midpointY, endX, endY, targetX)

    def interpolateLinear(self, startX, startY, endX, endY, targetX):
        # x1=2, y1=1, x2=12, y2=2, targetX = 7 
        # valuesRange = endX - startX = 10
        # targetPercent = (targetX - startX) / valuesRange = .5
        # valuesDomain = endY - startY = 1
        # answer = targetPercent * valuesDomain + startY = 1.5
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            return (((targetX - startX)/ (endX - startX)) * (endY - startY)) + startY

    def mandelbrot(self, c, escapeRadius, maxIter):
        """ 
        Now that smoothing is handled separately, the native python implementation
        COULD work for flint as well, except it uses 'complex(0,0)' instead
        of self.createComplex(0,0).
        The reason for this is just as below - to reduce one extra function call in 
        the core calculation.

        Could just call julia with a zero start here, but seems wiser to
        not have one extra function call in the core of the calculation?
        """
        z = complex(0,0)
        n = 0
        while abs(z) <= escapeRadius and n < maxIter:
            z = z*z + c
            n += 1

        return (n, z)

    def julia(self, c, z0, escapeRadius, maxIter):
        """
        Some interesting c values
        c = complex(-0.8, 0.156)
        c = complex(-0.4, 0.6)
        c = complex(-0.7269, 0.1889)

        Looks like this implementation is able to handle flint types, 
        now that smoothing is handled separately.

        fabs(z) returns the modulus of a complex number, which is the
        distance to 0 (on a 2x2 cartesian plane)
        However, instead we just square both sides of the inequality to
        avoid the sqrt
        e.g.: while (float((z.real**2)+(z.imag**2))) <= escapeSquared  and n < maxIter:

        However, for python, it looks like it takes almost 1/2 of the time to do abs(Z)
        """
        z = z0
        n = 0
        while abs(z) <= escapeRadius and n < maxIter:
            z = z*z + c
            n += 1

        return (n, z)

    def smoothAfterCalculation(self, endingZ, endingIter, maxIter, escapeRadius):
        if endingIter == maxIter:
            return float(maxIter)
        else:
            # The following code smooths out the colors so there aren't bands
            # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
            # Note: Results in a float. We think.
            #if endingIter != 1:
            #    print("iter was %d" % endingIter)
            #print("z: \"%s\" max_iter: %d iter: %d" % (endingZ, maxIter, endingIter))
            #return endingIter + 1 - math.log(math.log2(abs(endingZ)))
            #return endingIter + 1 - math.log(math.log2(abs(endingZ)) / math.log(2.0))
            return endingIter + 1 - self.twoLogsHelper(endingZ, escapeRadius) / math.log(2.0)

    def justTwoLogs(self, value):
        return math.log(math.log(value))

    def twoLogsHelper(self, value, radius):
        """
        The UNoptimized smoothing calculation.

        sn = n - ( ln( ln(abs(value))/ln(radius) ) / ln(2.0))

        Of which, we're just running the upper part here, because the ln(2.0)
        and subtraction don't need extra precision support.
        
        So, this calculates:
        ln(ln(abs(value))/ln(radius))
        """
        return math.log(math.log(abs(value))/math.log(radius)) 

    def mandelbrotDistanceEstimate(self, c, escapeRadius, maxIter):
# TODO: profile to make sure the exclusion is worth the extra multiplications
#        c2 = c.real*c.real + c.imag*c.imag
#        # skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
#        if  256.0*c2*c2 - 96.0*c2 + 32.0*c.real - 3.0 < 0.0: 
#            return 0.0
#        # skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
#        if 16.0*(c2+2.0*c.real+1.0) - 1.0 < 0.0: 
#            return 0.0

        didEscape = False
        z = complex(0,0)
        dz = complex(0,0) # (0+0j) start for mandelbrot, (1+0j) for julia
        absOfZ = 0.0 # Will re-use the final result for smoothing
        n = 0
        while absOfZ < escapeRadius and n < maxIter:
            # Z' -> 2·Z·Z' + 1
            dz = 2.0 * (z*dz) + 1
            # Z -> Z² + c           
            z = z*z + c

            absOfZ = abs(z)
            n += 1

        if n == maxIter:
            return (n, 0.0)
        else:    
            return (n, absOfZ * math.log(absOfZ) / abs(dz))

#        lastIter = 0
#        for i in range(0, maxIter):
#            if abs(z) > escapeRadius:
#               didEscape = True
#               break
#
#            # Z' -> 2·Z·Z' + 1
#            dz = 2.0 * (z*dz) + 1
#            # Z -> Z² + c           
#            z = z*z + c
#
#            lastIter += 1
#
#        if didEscape == False:
#            return (lastIter, 0.0)
#        else:    
#            absZ = abs(z) # Save an extra multiply
#            return (lastIter, absZ * math.log(absZ) / abs(dz))


    def rescaleForRange(self, rawValue, endingIter, maxIter, scaleRange):
#        #print("rescale %s for range %s" % (str(rawValue), str(scaleRange)))
#        if endingIter == maxIter or rawValue <= 0.0:
#            return 0.0
#        else:
#            zoomValue = float(math.pow((1/scaleRange) * rawValue / 0.1, 0.1))
#            return self.clamp(zoomValue, 0.0, 1.0)

        val = float(rawValue)
        if val < 0.0:
            val = 0.0
        zoo = .1 
        zoom_level = 1. / scaleRange 
        d = self.clamp( pow(zoom_level * val/zoo,0.1), 0.0, 1.0 );
        return float(d)

    def clamp(self, num, min_value, max_value):
       return max(min(num, max_value), min_value)

    def orig_mandelbrotDistanceEstimate(self, c, escapeRadius, maxIter):
        c2 = c.real*c.real + c.imag*c.imag
        # skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
        if  256.0*c2*c2 - 96.0*c2 + 32.0*c.real - 3.0 < 0.0: 
            return 0.0
        # skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
        if 16.0*(c2+2.0*c.real+1.0) - 1.0 < 0.0: 
            return 0.0

        # iterate
        di =  1.0;
        z  = complex(0.0);
        m2 = 0.0;
        dz = complex(0.0);
        for i in range(0,maxIter): 
            if m2>escapeRadius : 
              di=0.0 
              break

            # Z' -> 2·Z·Z' + 1
            dz = 2.0*complex(z.real*dz.real-z.imag*dz.imag, z.real*dz.imag + z.imag*dz.real) + complex(1.0,0.0);
                
            # Z -> Z² + c           
            z = complex( z.real*z.real - z.imag*z.imag, 2.0*z.real*z.imag ) + c;
                
            m2 = self.squared_modulus(z) 

        # distance  
        # d(c) = |Z|·log|Z|/|Z'|
        d = 0.5*math.sqrt(self.squared_modulus(z)/self.squared_modulus(dz))*math.log(self.squared_modulus(z));
        if  di>0.5:
            d=0.0
        
        return d             

    def squared_modulus(self, z):
        return ((z.real*z.real)+(z.imag*z.imag))
    

class DiveMathSupportFlint(DiveMathSupport):
    """
    Overrides to instantiate flint-specific complex types

    Looks like flint types are safe to use in base's createLinspace()
    """
    def __init__(self):
        super().__init__()

        self.flint = __import__('flint') # Only imports if you instantiate this DiveMathSupport subclass.
        self.flint.ctx.prec = FLINT_HIGH_PRECISION_SIZE  # Sets flint's precision (in bits)
        self.precisionType = 'flint'

    def createComplex(self, *args):
        """ 
        Flint complex type doesn't accept a string for instantiation.

        So, we do a string conversion here, to be flexible
        """
        if len(args) == 2:
            realPart = self.flint.arb(args[0])
            imagPart = self.flint.arb(args[1])
            return self.flint.acb(realPart, imagPart)
        elif len(args) == 1:
            if isinstance(args[0], str):
                partsString = args[0]

                # Trim off the leading sign
                firstIsPositive = True
                if partsString.startswith('+'):
                    partsString = partsString[1:]
                elif partsString.startswith('-'):
                    firstIsPositive = False
                    partsString = partsString[1:]

                # Trim off the trailing letter 
                lastIsImag = False
                if partsString.endswith('j'):
                    lastIsImag = True
                    partsString = partsString[:-1]

                # Remaining string might have an internal sign.
                # If there's no internal sign, then the whole remaining
                # string is either the real or the complex
                positiveParts = partsString.split('+')
                negativeParts = partsString.split('-')

                realIsPositive = True
                imagIsPositive = True
                realPart = ""
                imagPart = ""
                if len(positiveParts) == 2:
                    realIsPositive = firstIsPositive
                    realPart = positiveParts[0]
                    imagIsPositive = True
                    imagPart = positiveParts[1]
                elif len(negativeParts) == 2:
                    realIsPositive = firstIsPositive
                    realPart = negativeParts[0]
                    imagIsPositive = False
                    imagPart = negativeParts[1]
                elif len(positiveParts) == 1 and len(negativeParts) == 1:
                    # No internal + or -, so it should just be a number
                    if lastIsImag == True:
                        imagPart = partsString
                        imagIsPositive = firstIsPositive
                    else:
                        realPart = partsString
                        realIsPositive = firstIsPositive
                else:
                    raise ValueError("String parameter not identifiably a complex number, in createComplex()")

                preparedReal = '0.0'
                preparedImag = '0.0'
                if realPart != "":
                    if realIsPositive == True:
                        preparedReal = realPart
                    else:
                        preparedReal = "-%s" % realPart

                if imagPart != "":
                    if imagIsPositive == True:
                        preparedImag = imagPart
                    else:
                        preparedImag = "-%s" % imagPart

                return self.flint.acb(preparedReal, preparedImag)
            else:
                if isinstance(args[0], complex):
                    return self.flint.acb(args[0].real, args[0].imag)
                else:
                    return self.flint.acb(args[0]) # Just a constant, so make a 0-imaginary value
        elif len(args) == 0:
            return self.flint.acb(0,0)
        else:
            raise ValueError("Max 2 parameters are valid for createComplex(), but it's best to use one string")

    def createFloat(self, floatValue):
        return self.flint.arb(floatValue)

    def floor(self, value):
        return self.flint.arb(value).floor()

    def scaleValueByFactorForIterations(self, startValue, overallZoomFactor, iterations):
        """
        Shortcut calculate the starting point for the last frame's properties, 
        which we'll use for instantiation with specific widths.  This is
        because if any keyframes are added, the best we can do is measure an
        effective zoom factor between any 2 frames.

        Note: Flint's __pow__ needs TWO arb types as args, or else it drops 
        to default python bit depths
        """
        zoomAsArb = self.flint.arb(overallZoomFactor)
        iterationsAsArb = self.flint.arb(iterations)
        return startValue * pow(zoomAsArb, iterationsAsArb)
       

    def interpolateLogTo(self, startX, startY, endX, endY, targetX):
        """
        Probably want additional log-defining params, but for now, let's just bake in one equation
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            aVal = (endY - startY) / (self.flint.arb(endX - startX + 1).log())
            return aVal * (self.flint.arb(targetX - startX + 1).log()) + startY 

    def interpolateRootTo(self, paramStartX, paramStartY, paramEndX, paramEndY, paramTargetX):
        """ 
        Iterative multiplications of window sizes for zooming means 
        we want to be able to interpolate between two points using
        the frame count as the root.
        """
        startX = self.flint.arb(paramStartX)
        startY = self.flint.arb(paramStartY)
        endX = self.flint.arb(paramEndX)
        endY = self.flint.arb(paramEndY)
        targetX = self.flint.arb(paramTargetX)

        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            root  = endX - startX
            scaleFactor = (endY / startY) ** (1 / root)
            return startY * (scaleFactor ** targetX)

    def interpolateQuadraticEaseOut(self, paramStartX, paramStartY, paramEndX, paramEndY, paramTargetX):
        """
        QuadraticEaseOut leaves the majority of changes to the end of the X range.

        Probably want additional quadratic params, but for now, let's just bake in one equation
        which uses the first point as the vertex, and passes through the second point.
        """
        startX = self.flint.arb(paramStartX)
        startY = self.flint.arb(paramStartY)
        endX = self.flint.arb(paramEndX)
        endY = self.flint.arb(paramEndY)
        targetX = self.flint.arb(paramTargetX)

        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            # Find a, given that the start point is the vertex, and the parabola passes 
            # through the other point
            # y = a * (x - h)**2 + k
            # y = a * (x - startX)**2 + startY
            # endY = a * (endX - startX)**2 + startY
            # a = ((endY-startY)/((endX-startX)**2)
            #
            # answer = a * (targetX - startX)**2 + startY
            # answer = ((endY-startY)/((endX-startX)**2) * ((targetX-startX)**2) + startY
            return (endY-startY)/((endX-startX)**self.flint.arb(2.0)) * ((targetX-startX)**self.flint.arb(2.0)) + startY

    def mandelbrot(self, c, escapeRadius, maxIter):
        """ 
        Now that smoothing is handled separately, the native python implementation
        COULD work for flint as well, except it uses 'complex(0,0)' instead
        of self.createComplex(0,0).
        The reason for this is just as below - to reduce one extra function call in 
        the core calculation.

        Could just call julia with a zero start here, but seems wiser to
        not have one extra function call in the core of the calculation?
        """
        z = self.flint.acb(0,0)
        n = 0
        while float(z.abs_lower()) <= escapeRadius and n < maxIter:
            z = z*z + c
            n += 1

        return (n, z)

    def smoothAfterCalculation(self, endingZ, endingIter, maxIter, escapeRadius):
        """
        This flint-specific implementation only really does flint-y logs in the smoothing.
        The Decimal-specific implementation needed some extra steps, but I've ditched that one.
        Heh, now that twoLogsHelper exists, it looks like this flint version
        is identical to the native version.  Might not need it anymore.
        """
        if endingIter == maxIter:
            return float(maxIter)
        else:
            # The following code smooths out the colors so there aren't bands
            # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
            # Note: Results in a float. We think.
            #return float(endingIter - self.twoLogsHelper(endingZ, escapeRadius) / math.log(2.0))
            return float(endingIter - self.justTwoLogs(self.squared_modulus(endingZ)) + 4.0)

    def justTwoLogs(self, value):
        return self.flint.arb(value).log().log()

    def twoLogsHelper(self, value, radius):
        #print("trying two logs...")
        # ln(ln(abs(value))/ln(radius))
        # Note: flint needs 'value.log()' for ln, but 'value.const_log2() for log2?!

        # Also, radius is expected to be in 'normal' ranges, so just using math.log
        return (value.abs_lower().log() / math.log(radius)).log()
        #return math.log(math.log(abs(value))/math.log(radius)) 

    def mandelbrotDistanceEstimate(self, c, escapeRadius, maxIter):
        didEscape = False
        z = self.flint.acb(0,0)
        dz = self.flint.acb(0,0) # (0+0j) start for mandelbrot, (1+0j) for julia
        absOfZ = 0.0 # Will re-use the final result for smoothing
        n = 0
        while float(absOfZ) < escapeRadius and n < maxIter:
            # Z' -> 2·Z·Z' + 1
            dz = 2.0 * (z*dz) + 1
            # Z -> Z² + c           
            z = z*z + c

            absOfZ = z.abs_lower()
            n += 1

        dzMag = dz.abs_lower()
        if n == maxIter or dzMag == 0.0:
            return (n, 0.0)
        else:    
            return (n, float(absOfZ * absOfZ.const_log2() / dzMag))

#        didEscape = False
#        z = self.flint.acb(0,0)
#        dz = self.flint.acb(0,0) # (0+0j) start for mandelbrot, (1+0j) for julia
#
#        for i in range(0, maxIter):
#            if float(z.abs_lower()) > escapeRadius:
#                didEscape = True
#                break
#
#            # Z' -> 2·Z·Z' + 1
#            dz = 2.0 * (z*dz) + 1
#            # Z -> Z² + c           
#            z = z*z + c
#
#        if didEscape == False:
#            return 0.0
#        else:    
#            absZ = z.abs_lower()
#            return float(absZ * absZ.const_log2() / dz.abs_lower())

    def rescaleForRange(self, rawValue, endingIter, maxIter, scaleRange):
#        #print("rescale %s for range %s" % (str(rawValue), str(scaleRange)))
        val = float(rawValue)
        if val < 0.0:
            val = 0.0
        zoo = .1 
        zoom_level = 1. / scaleRange 
        if math.isnan(zoom_level):
            print("NaN wtf 1")
            return 0.0
        if zoom_level == 0:
            print("zoom level zero")
            return 0.0

        pow_result = pow(zoom_level * val/zoo, 0.1)
        if math.isnan(pow_result):
            print("NaN wtf 2: %s * %s / %s" % (str(zoom_level), str(val), str(zoo)))
            return 0.0
        if pow_result == 0:
            #print("pow zero")
            return 0.0

        #d = self.clamp( pow(zoom_level * val/zoo,0.1), 0.0, 1.0 );
        d = self.clamp(pow_result, 0.0, 1.0 );
        return float(d)

    def clamp(self, num, min_value, max_value):
       return max(min(num, max_value), min_value)

class DiveMathSupportFlintCustom(DiveMathSupportFlint):
    def mandelbrot(self, c, escapeRadius, maxIter):
        """ Slightly more efficient for HIGH maxIter values """
        #print("mandelbrot center: %s radius: %s maxIter: %s" % (str(c), str(escapeRadius), str(maxIter)))
        #print("mandelbrot maxIter: %s" % (str(maxIter)))
        (answer, lastZ, remainingPrecision) = c.our_steps_mandelbrot(escapeRadius, maxIter)
        #print("answer: %s, lastZ: %s remainingPrecision: %s" % (str(answer), str(lastZ), str(remainingPrecision)))
        #print("answer: %s, remainingPrecision: %s" % (str(answer), str(remainingPrecision)))
        return(answer, lastZ)

    def mandelbrot_check_precision(self, c, escapeRadius, maxIter):
        """ Slightly more efficient for HIGH maxIter values """
        #print("mandelbrot center: %s radius: %s maxIter: %s" % (str(c), str(escapeRadius), str(maxIter)))
        (answer, lastZ, remainingPrecision) = c.our_steps_mandelbrot(escapeRadius, maxIter)
        #print("answer: %s, lastZ: %s remainingPrecision: %s" % (str(answer), str(lastZ), str(remainingPrecision)))
        return(answer, lastZ, remainingPrecision)

    def mandelbrot_beginning(self, c, escapeRadius, maxIter):
        """ Slightly more efficient for LOW maxIter values """
        #print("mandelbrot_beginning center: %s radius: %s maxIter: %s" % (str(c), str(escapeRadius), str(maxIter)))
        #print("mandelbrot_beginning radius: %s maxIter: %s" % (str(escapeRadius), str(maxIter)))
        (answer, lastZ, remainingPrecision) = c.our_mandelbrot(escapeRadius, maxIter)
        #print("beginning answer: %s, lastZ: %s remainingPrecision: %s" % (str(answer), str(lastZ), str(remainingPrecision)))
        #print("beginning answer: %s, remainingPrecision: %s" % (str(answer), str(remainingPrecision)))
        return(answer, lastZ)

    def mandelbrot_beginning_check_precision(self, c, escapeRadius, maxIter):
        """ Slightly more efficient for LOW maxIter values """
        #print("mandelbrot_beginning center: %s radius: %s maxIter: %s" % (str(c), str(escapeRadius), str(maxIter)))
        (answer, lastZ, remainingPrecision) = c.our_mandelbrot(escapeRadius, maxIter)
        #print("beginning answer: %s, lastZ: %s remainingPrecision: %s" % (str(answer), str(lastZ), str(remainingPrecision)))
        return(answer, lastZ, remainingPrecision)


class DiveMathSupportGmp(DiveMathSupport):
    """
    Overrides to instantiate gmpy2-specific complex types

    Note from the GMP docs - contexts and context managers are not thread-safe!  
    Modifying the context in one thread will impact all other threads.
    """
    def __init__(self):
        super().__init__()

        self.gmp = __import__('gmpy2') # Only imports if you instantiate this DiveMathSupport subclass.
        self.gmp.get_context().precision = GMP_HIGH_PRECISION_SIZE
        self.precisionType = 'gmp'

    def createComplex(self, *args):
        """
        gmpy2.mpc() accepts one of:
        1 string (a native python complex string, with real and/or imag components)
        1 complex object (python native)
        1 float (no imaginary component)
        2 floats
        2 mpfr (gmp's float type)
        """
        if len(args) == 2:
            realPart = self.gmp.mpfr(args[0])
            imagPart = self.gmp.mpfr(args[1])
            return self.gmp.mpc(realPart, imagPart)
        elif len(args) == 1:
             return self.gmp.mpc(args[0])
        elif len(args) == 0:
            return self.gmp.mpc(0.0)
        else:
            raise ValueError("Max 2 parameters are valid for createComplex(), but it's best to use one string")

    def createFloat(self, floatValue):
        return self.gmp.mpfr(floatValue)

    def floor(self, value):
        return self.gmp.floor(value)

    def interpolateLogTo(self, startX, startY, endX, endY, targetX):
        """
        Probably want additional log-defining params, but for now, let's just bake in one equation
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            aVal = (endY - startY) / (self.gmp.log(endX - startX + 1))
            return aVal * (self.gmp.log(targetX - startX + 1)) + startY 

    def mandelbrot(self, c, escapeRadius, maxIter):
        """ 
        Now that smoothing is handled separately, the native python implementation
        COULD work for flint as well, except it uses 'complex(0,0)' instead
        of self.createComplex(0,0).
        The reason for this is just as below - to reduce one extra function call in 
        the core calculation.

        Could just call julia with a zero start here, but seems wiser to
        not have one extra function call in the core of the calculation?

        Really super didn't help to try locally caching function names
        and using paren syntax - it doubled the runtime.  Trying that was
        probably just old advice for pre-python3?
        """
        z = self.gmp.mpc(0.0)
        n = 0

        # Because norm() doesn't work for python3-gmp2 v 2.1.0a4 
        #while float(self.gmp.sqrt(self.gmp.square(z.real) + self.gmp.square(z.imag))) <= escapeRadius and n < maxIter:
        # looks like abs() gets to the right place, even though there's no explicit abs_lower() in libgmp?
        while abs(z) <= escapeRadius and n < maxIter:
            z = z*z + c
            n += 1
        return (n, z)

    def smoothAfterCalculation(self, endingZ, endingIter, maxIter, escapeRadius):
        """
        This flint-specific implementation only really does flint-y logs in the smoothing.
        The Decimal-specific implementation needed some extra steps, but I've ditched that one.

        NOTE: this wasn't updated with the new smoothing calcs like
        the 'native' and 'flint' implementations were
        """
        if endingIter == maxIter:
            return float(maxIter)
        else:
            # The following code smooths out the colors so there aren't bands
            # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
            # Note: Results in a float. We think.
            return float(endingIter + 1 - self.gmp.log10(self.gmp.log2(self.gmp.norm(endingZ))) / math.log(2.0))
            #return float(endingIter + 1 - self.gmp.log10(self.gmp.log10(self.gmp.sqrt(self.gmp.square(endingZ.real) + self.gmp.square(endingZ.imag)))))



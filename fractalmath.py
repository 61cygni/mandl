# --
# File: fractalmath.py
#
# Home for math operations that should be high-precision aware
#
# --

import math

import numpy as np

import arb_fractalmath
import mpfr_fractalmath

#FLINT_HIGH_PRECISION_SIZE = 53 # 53 is how many bits are in float64
#3.32 bits per digit, on average
#2200 was therefore, ~662 digits, got 54 frames down at .5 scaling

#FLINT_HIGH_PRECISION_SIZE = int(2800 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(2200 * 3.32) # 2200*3.32 = 7304, lol
#FLINT_HIGH_PRECISION_SIZE = int(1800 * 3.32)
#FLINT_HIGH_PRECISION_SIZE = int(1125 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(500 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(400 * 3.32) 

# For debugging, looks like we're bottoming out somewhere around e-11
# So, only really need ~20 digits for this test
# Blowing out frame 30, which is ~28?
#FLINT_HIGH_PRECISION_SIZE = int(50 * 3.32) 
# Native blew out somewhere e-13 to e-15
#FLINT_HIGH_PRECISION_SIZE = int(200 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(500 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(60 * 3.32) 
#FLINT_HIGH_PRECISION_SIZE = int(30 * 3.32) 
FLINT_HIGH_PRECISION_SIZE = int(16 * 3.32) 

class DecimalComplex:
    def __init__(self, realValue, imagValue):
        super().__init__()
        self.real = realValue
        self.imag = imagValue

    def __repr__(self):
        if self.real == None:
            realString = "0"
        else: # self.real != None:
            realString = str(self.real)
        if self.imag == None:
            imagString = "+0" 
        else: # self.imag != None:
            if self.imag >= 0.0:
                imagString = f"+{str(self.imag)}"
            else:
                imagString = str(self.imag)
        return f"({realString}{imagString}j)"

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

    Trying to preserve types where possible, without forcing casting, 
    because sometimes all the math operations will already work for 
    custom numeric types.
    """
    def __init__(self):
        self.precisionType = 'native'

        self.decimal = __import__('decimal') # Only imports if you instantiate this DiveMathSupport subclass.
        self.defaultPrecisionSize = 53 
        self.decimal.getcontext().prec = self.bitsToDigits(self.defaultPrecisionSize)

    def digitsToBits(self, digits):
        return round(digits * 3.3219)
        
    def bitsToDigits(self, bits):
        return round(bits / 3.3219)

    # Decimal precision is in digits natively, so we can handle
    # the in and out conversions to make bits work the right way too.
    def setPrecision(self, newPrecision):
        newDigits = self.bitsToDigits(newPrecision)
        oldPrecision = self.decimal.getcontext().prec
        self.decimal.getcontext().prec = newDigits
        return oldPrecision
        
    def precision(self):
        return self.digitsToBits(self.decimal.getcontext().prec)

    def setDigitsPrecision(self, newDigits):
        oldPrecision = self.decimal.getcontext().prec
        self.decimal.getcontext().prec = newDigits
        return oldPrecision

    def digitsPrecision(self):
        return self.decimal.getcontext().prec

    def createComplex(self, *args):
        """
        Most of the time, keeping separate real and imaginary numbers is
        going to be more efficient than creating complex representations.
 
        Compatible complex types will return values for .real() and .imag()
        """
        if len(args) == 2:
            return DecimalComplex(self.decimal.Decimal(args[0]), self.decimal.Decimal(args[1]))
        elif len(args) == 1:
            if isinstance(args[0], str):
                partsString = args[0]

                # Trim off surrounding parens, if present
                if partsString.startswith('('):
                    partsString = partsString[1:]
                if partsString.endswith(')'):
                    partsString = partsString[:-1]

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
                    raise ValueError("String parameter \"%s\" not identifiably a complex number, in createComplex()" % args[0])

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

                print(f"preparedReal \"{preparedReal}\"")
                print(f"preparedImag \"{preparedImag}\"")
                return DecimalComplex(self.decimal.Decimal(preparedReal), self.decimal.Decimal(preparedImag))
            else:
                if isinstance(args[0], complex):
                    return DecimalComplex(self.decimal.Decimal(args[0].real), self.decimal.Decimal(args[0].imag))
                else:
                    return DecimalComplex(self.decimal.Decimal(args[0]),self.decimal.Decimal(0))# Just a constant, so make a 0-imaginary value
        elif len(args) == 0:
            return DecimalComplex(self.decimal.Decimal(0),self.decimal.Decimal(0))
        else:
            raise ValueError("Max 2 parameters are valid for createComplex(), but it's best to use one string")

        # Maybe don't need real native 'complex' anywhere now?
        ## Can't make a native complex from 2 strings though, so when we
        ## detect that, jam them into floats
        #if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
        #    return complex(float(args[0]), float(args[1]))
        ##elif len(args) == 1 and isinstance(args[0], str):
        ##    # Strip parens out of the definition string?
        #else:
        #    return complex(*args) 

    def createFloat(self, floatValue):
        return self.decimal.Decimal(floatValue)

    def stringFromFloat(self, paramFloat):
        return str(paramFloat)

    def shorterStringFromFloat(self, paramFloat, places):
        return round(paramFloat, places)

    def floor(self, value):
        return math.floor(value)

    def arrayToStringArray(self, paramArray):
        #stringifier = np.vectorize(str)
        stringifier = np.vectorize(self.stringFromFloat) # So sublasses work better?
        return stringifier(paramArray)

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
        """
        Attempt at a mostly type-agnostic linspace(), seems to work with 
        flint types too.
        """
        #print(f"createLinspace {paramFirst}, {paramLast}, {quantity}")
        dataRange = paramLast - paramFirst
        answers = np.zeros((quantity), dtype=object)

        # Subtly, also likely forcing quantity to Decimal
        oneLessQuantity = self.createFloat(quantity - 1)

        #print(f"first {paramFirst.__class__}, last {paramLast.__class__}, quantity {quantity.__class__}") 
        for x in range(0, quantity):
            answers[x] = paramFirst + dataRange * (x / oneLessQuantity)

        return answers

    def createLinspaceAroundValuesCenter(self, valuesCenter, spreadWidth, quantity):
        """
        Attempt at a mostly type-agnostic linspace-from-center-and-width

        Turns out, we didn't *really* need this for using flint, though it
        was an important for a Python/Decimal version to be able to have a
        type-specific implementation in what was a DiveMeshDecimal function
        """
        #print(f"createLinspaceAroundValuesCenter {valuesCenter}, {spreadWidth}, {quantity}")

        return self.createLinspace(valuesCenter - spreadWidth * self.createFloat(0.5), valuesCenter + spreadWidth * self.createFloat(0.5), quantity)

    def interpolate(self, transitionType, startX, startY, endX, endY, targetX, extraParams={}):
        """
        Resisted having a single call point, in case more parameters
        were needed for each type, but we might just be able to pass
        in an extra param hash?
        """
        print(f"DiveMathSuppport interpolation type {transitionType}")

        startX = self.decimal.Decimal(startX)
        startY = self.decimal.Decimal(startY)
        endX = self.decimal.Decimal(endX)
        endY = self.decimal.Decimal(endY)
        targetX = self.decimal.Decimal(targetX)

        if transitionType == 'log-to':
            return self.interpolateLogTo(startX, startY, endX, endY, targetX)
        elif transitionType == 'root-to':
            return self.interpolateRootTo(startX, startY, endX, endY, targetX)
        elif transitionType == 'root-from':
            return self.interpolateRootFrom(startX, startY, endX, endY, targetX)
        elif transitionType == 'root-to-ease-in':
            return self.interpolateRootToEaseIn(startX, startY, endX, endY, targetX)
        elif transitionType == 'root-to-ease-out':
            return self.interpolateRootToEaseOut(startX, startY, endX, endY, targetX)
        elif transitionType == 'root-from-ease-in':
            return self.interpolateRootFromEaseIn(startX, startY, endX, endY, targetX)
        elif transitionType == 'root-from-ease-out':
            return self.interpolateRootFromEaseOut(startX, startY, endX, endY, targetX)
        elif transitionType == 'linear':
            return self.interpolateLinear(startX, startY, endX, endY, targetX)
        elif transitionType == 'step':
            return startY # Kinda a special not-actual-interpolation
        elif transitionType == 'quadratic-to':
            return self.interpolateQuadraticEaseOut(startX, startY, endX, endY, targetX)
        elif transitionType == 'quadratic-from':
            return self.interpolateQuadraticEaseIn(startX, startY, endX, endY, targetX)
        elif transitionType == 'quadratic-to-from':
            return self.interpolateQuadraticEaseInOut(startX, startY, endX, endY, targetX)
        else: # transitionType == 'quadratic-to-from'
            raise ValueError(f"ERROR - Transition type \"{transitionType}\" isn't recognized!")

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
        print(f"root-to attempt")
        if targetX == endX:
            print("  end")
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            print("  start or identical")
            return startY
        else:
            root = abs(endX - startX)
            scaleFactor = (endY / startY) ** (1 / root)
            debugAnswer = startY * (scaleFactor ** abs(targetX - startX))
            print(f"  root-to answer {debugAnswer}")
            return startY * (scaleFactor ** abs(targetX - startX))
            #root = endX - startX
            #scaleFactor = (endY / startY) ** (1 / root)
            #debugAnswer = startY * (scaleFactor ** (targetX - startX))
            #print(f"  root-to answer {debugAnswer}")
            #return startY * (scaleFactor ** (targetX - startX))

    def interpolateRootFrom(self, startX, startY, endX, endY, targetX):
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            print(f"root-froming into (({startX},{startY}),({endX},{endY}))")
            #debugAnswer = self.interpolateRootTo(startX, endY, endX, startY, targetX)
            #print(f"root-froming answer {debugAnswer}")
            #return self.interpolateRootTo(startX, endY, endX, startY, targetX)
            return self.interpolateRootTo(endX, endY, startX, startY, targetX)

    ########
    # For the combination interpolations...
    #
    # First, map the target position to a quadratic-adjusted
    # position, then use that adjusted target position as the
    # input to the 'primary' interpolation
    ########

    def interpolateRootToEaseIn(self, startX, startY, endX, endY, targetX):
        """
        For example, zoom factors are root-to, so use this to settle 
        into the target zoom (rather than hitting it suddenly). 
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            targetXRatio = self.interpolateQuadraticEaseIn(startX, 0.0, endX, 1.0, targetX)
            adjustedTargetX = ((endX - startX) * targetXRatio) + startX

            print(f"root-to-ease-in adjusted target from {targetX} to {adjustedTargetX} mapping into (({startX},{startY}),({endX},{endY}))")
            return self.interpolateRootTo(startX, startY, endX, endY, adjustedTargetX)

    def interpolateRootToEaseOut(self, startX, startY, endX, endY, targetX):
        """
        For example, zoom factors are root-to, so use this to slowly
        build up speed when zooming in, rather than suddenly starting.
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            targetXRatio = self.interpolateQuadraticEaseOut(startX, 0.0, endX, 1.0, targetX)
            adjustedTargetX = ((endX - startX) * targetXRatio) + startX
            print(f"root-to-ease-out adjusted target from {targetX} at ratio {targetXRatio} to {adjustedTargetX} mapping into (({startX},{startY}),({endX},{endY}))")

            return self.interpolateRootTo(startX, startY, endX, endY, adjustedTargetX)

    def interpolateRootFromEaseIn(self, startX, startY, endX, endY, targetX):
        """
        When zooming OUT at a constant speed, the behavior is root-from,
        so this interpolation settles into the end point. 
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            targetXRatio = self.interpolateQuadraticEaseIn(startX, 0.0, endX, 1.0, targetX)
            adjustedTargetX = ((endX - startX) * targetXRatio) + startX
            print(f"root-from-ease-in adjusted target from {targetX} to {adjustedTargetX} mapping into (({startX},{startY}),({endX},{endY}))")

            return self.interpolateRootFrom(startX, startY, endX, endY, adjustedTargetX)
    def interpolateRootFromEaseOut(self, startX, startY, endX, endY, targetX):
        """
        When zooming OUT at a constant speed, the behavior is root-from,
        so this interpolation slowly builds up speed from rest (rather
        than starting suddenly).
        """
        if targetX == endX:
            return endY
        elif targetX == startX or startX == endX or startY == endY:
            return startY
        else:
            targetXRatio = self.interpolateQuadraticEaseOut(startX, 0.0, endX, 1.0, targetX)
            adjustedTargetX = ((endX - startX) * targetXRatio) + startX
            print(f"root-from-ease-out adjusted target from {targetX} to {adjustedTargetX} mapping into (({startX},{startY}),({endX},{endY}))")

            return self.interpolateRootFrom(startX, startY, endX, endY, adjustedTargetX)

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

    def julia(self, realValues, imagValues, realJuliaValue, imagJuliaValue, escapeRadius, maxIter):
        currShape = realValues.shape
        results = np.empty(currShape, dtype=object)
        lastRealValues = np.empty(currShape, dtype=object)
        lastImagValues = np.empty(currShape, dtype=object)
  
        # numpyArray.shape returns (rows, columns) 
        for y in range(currShape[0]):
            for x in range(currShape[1]):
                (results[y,x], lastRealValues[y,x], lastImagValues[y,x]) = self.juliaSingle(realValues[y,x], imagValues[y,x], realJuliaValue, imagJuliaValue, escapeRadius, maxIter)

        return (results, lastRealValues, lastImagValues)

    def juliaSingle(self, realValue, imagValue, realJuliaValue, imagJuliaValue, escapeRadius, maxIter):
        """
        Default behavior is 2D decimal to 2D decimal.

        Some interesting c values
        c = complex(-0.8, 0.156)
        c = complex(-0.4, 0.6)
        c = complex(-0.7269, 0.1889)
        """
        radSquared = escapeRadius * escapeRadius
        result = 0

        # Perhaps no longer relevant, but observations about python math:
        # fabs(z) returns the modulus of a complex number, which is the
        # distance to 0 (on a 2x2 cartesian plane), and NORMALLY, we'd
        # check zRealSquared + zImagSquared against the escape-value-squared,
        # but in python it looks like it takes almost 1/2 the time to do
        # abs(Z) (when Z is a complex number), so checking against the
        # un-squared radius is even faster somehow. 

        realDecimal = self.decimal.Decimal(realValue)
        imagDecimal = self.decimal.Decimal(imagValue)

        zReal = self.decimal.Decimal(realJuliaValue)
        zImag = self.decimal.Decimal(imagJuliaValue)
        zrSquared = zReal * zReal
        ziSquared = zImag * zImag

        for currIter in range(maxIter + 1):
            result = currIter

            if (zrSquared + ziSquared) > radSquared:
                break

            # Below is 2(z.real*z.imag) + c.imag, but without an extra multiply
            partSum = zReal * zImag
            zImag = partSum + partSum + imagDecimal 

            zReal = zrSquared - ziSquared + realDecimal

            zrSquared = zReal * zReal
            ziSquared = zImag * zImag

        return (result, zReal, zImag)        

    def mandelbrot(self, realValues, imagValues, escapeRadius, maxIter):
        currShape = realValues.shape

        results = np.empty(currShape, dtype=object)
        lastRealValues = np.empty(currShape, dtype=object)
        lastImagValues = np.empty(currShape, dtype=object)
  
        # numpyArray.shape returns (rows, columns) 
        for y in range(currShape[0]):
            for x in range(currShape[1]):
                (results[y,x], lastRealValues[y,x], lastImagValues[y,x]) = self.mandelbrotSingle(realValues[y,x], imagValues[y,x], escapeRadius, maxIter)

        return (results, lastRealValues, lastImagValues)
 
    def mandelbrotSingle(self, realValue, imagValue, escapeRadius, maxIter):
        """
        Normally a sub-library would set up appropriate mandelbrot and
        julia calls, but since native implementation is right here, we
        have a special case where we know which implementation to use,
        and can call it directly
        """
        return self.juliaSingle(realValue, imagValue, 0, 0, escapeRadius, maxIter)

    def smoothAfterCalculation(self, lastRealValues, lastImagValues, endingIters, maxIter, escapeRadius):
        """
        Not at all ideal to iterate over the numpy array, but I can't see a 
        trivial way to iterate all the arrays in parallel at the moment.
        Probably a 'pass-the-index' technique of some kind.
        """
        currShape = lastRealValues.shape

        results = np.empty(currShape, dtype=object)
  
        # numpyArray.shape returns (rows, columns) 
        for y in range(currShape[0]):
            for x in range(currShape[1]):
                results[y,x] = self.smoothAfterCalculationSingle(lastRealValues[y,x], lastImagValues[y,x], endingIters[y,x], maxIter, escapeRadius)

        return results
 
    def smoothAfterCalculationSingle(self, lastReal, lastImag, endingIter, maxIter, escapeRadius):
        # Below from: 
        # https://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
        #
        # // smooth iteration count
        # //float sn = n - log(log(length(z))/log(B))/log(2.0); 
        #
        # // equivalent optimized smooth iteration count
        # float sn = n - log2(log2(dot(z,z))) + 4.0;  
        #
        # But, that seems to be for B = 2.0
        # I'm trying to use escape radius of 10.0
        # So, log10(10.0) = 1.0, which kills the denominator inside the logs.
        # Which brings the expression to 
        # log(log(length(z))) / log(2.0) 
        if endingIter == maxIter:
            return self.createFloat(maxIter)
        else:
            sqMagnitude = self.decimal.Decimal(lastReal) ** 2 + self.decimal.Decimal(lastImag) ** 2 
            return self.decimal.Decimal(endingIter) - ((sqMagnitude.sqrt().ln().ln()) / (self.decimal.Decimal('2').ln()))  

    def mandelbrotDistanceEstimate(self, realValues, imagValues, escapeRadius, maxIter):
        currShape = realValues.shape

        results = np.empty(currShape, dtype=object)
        lastRealValues = np.empty(currShape, dtype=object)
        lastImagValues = np.empty(currShape, dtype=object)
  
        # numpyArray.shape returns (rows, columns) 
        for y in range(currShape[0]):
            for x in range(currShape[1]):
                (results[y,x], lastRealValues[y,x], lastImagValues[y,x]) = self.mandelbrotDistanceEstimateSingle(realValues[y,x], imagValues[y,x], escapeRadius, maxIter)

        return (results, lastRealValues, lastImagValues)
 
    def mandelbrotDistanceEstimateSingle(self, realValue, imagValue, escapeRadius, maxIter):
        """
        Wonderfully slow implementation of distance estimate, but we need
        a fallback/baseline, so here we go.
        """
        radSquared = escapeRadius * escapeRadius
        result = 0

        realDecimal = self.decimal.Decimal(realValue)
        imagDecimal = self.decimal.Decimal(imagValue)

        zReal = self.decimal.Decimal(0)
        zImag = self.decimal.Decimal(0)
        zrSquared = zReal * zReal
        ziSquared = zImag * zImag

        # Mandelbrot derivative initialize to (0+0j), but julia is (1+0j), FYI
        dzReal = self.decimal.Decimal(0)
        dzImag = self.decimal.Decimal(0)
 
        for currIter in range(maxIter + 1):
            result = currIter
    
            if (zrSquared + ziSquared) > radSquared:
                break

            # Z' -> 2·Z·Z' + 1  
            # FYI: No "+1" for julia, dz is just 2·Z·Z'
            # (a + bi) * (c + di)
            # = ac + adi + bci - bd = ac - bd + (ad+bc)i
            partSum = zReal * dzReal - zImag * dzImag 
            newDzReal = partSum + partSum + 1 # Don't stomp dz yet
            dzImag = zReal * dzImag + zImag * dzReal 
            dzReal = newDzReal # Now safe to stomp

            # Z -> Z² + c           
            # Below is 2(z.real*z.imag) + c.imag, but without an extra multiply
            partSum = zReal * zImag
            zImag = partSum + partSum + imagDecimal 
            zReal = zrSquared - ziSquared + realDecimal

            zrSquared = zReal * zReal
            ziSquared = zImag * zImag

        # Tried to extract the distance smoothing from this a few times, 
        # but it seems a lot messier to pass around the (needed) derivative
        # as well, so we just go ahead and return an extra term here, knowing
        # this theoretically could be better handled in the 'normal' two steps.
        if result == maxIter:
            return (result, zReal, zImag)
        else:
            realPart = zReal * zReal 
            imagPart = zImag * zImag 
            zMagnitude = (realPart + imagPart).sqrt()
  
            realPart = dzReal * dzReal
            imagPart = dzImag * dzImag
            dzMagnitude = (realPart + imagPart).sqrt()

            ## Can't take logs of zMagnitude when <= 0, 
            ## and can't divide by zero-valued dzMagnitude.
            ## (though these shouldn't ever be, because of sqrt?)
            #if zMagnitude <= 0 or dzMagnitude <= 0:
            #    return (result, zReal, zImag)
            #
            # Current most likely: 
            #return ((zMagnitude * (zMagnitude.ln()) / dzMagnitude), zReal, zImag)
            # But need to include the actual result, right?
            #return (result - (zMagnitude * (zMagnitude.ln()) / dzMagnitude), zReal, zImag)
            tmpAnswer = result - (zMagnitude * (zMagnitude.ln()) / dzMagnitude)
            if result > 0:
                return (tmpAnswer, zReal, zImag)
            else:
                return (self.decimal.Decimal(0), zReal, zImag)

            # More math, slightly less blown out?
            #return ((zMagnitude * zMagnitude).ln() * zMagnitude / dzMagnitude, zReal, zImag)
            # People seem to like this for 2D:
            #return (result - self.decimal.Decimal('0.5') * zMagnitude.ln() * zMagnitude / dzMagnitude, zReal, zImag)

            # somewhere on: https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/demm
            #result:=2*log2(sqrt(xy2))*sqrt(xy2)/sqrt(sqr(eDx)+sqr(eDy));
            #return ((2 * zMagnitude.ln() * zMagnitude / dzMagnitude), zReal, zImag)
           
            # Perhaps identical to current most likely, but looks very close anyway 
            # From https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/demm
            #R = -K*Math.log(Math.log(R2+I2)*Math.sqrt((R2+I2)/(Dr*Dr+Di*Di))); // compute distance
            #return (result - (2 * (zReal * zReal + zImag * zImag).ln() * zMagnitude / dzMagnitude), zReal, zImag)

    def juliaDistanceEstimate(self, realValues, imagValues, realJuliaValue, imagJuliaValue, escapeRadius, maxIter):
        currShape = realValues.shape

        results = np.empty(currShape, dtype=object)
        lastRealValues = np.empty(currShape, dtype=object)
        lastImagValues = np.empty(currShape, dtype=object)
  
        # numpyArray.shape returns (rows, columns) 
        for y in range(currShape[0]):
            for x in range(currShape[1]):
                (results[y,x], lastRealValues[y,x], lastImagValues[y,x]) = self.juliaDistanceEstimateSingle(realValues[y,x], imagValues[y,x], realJuliaValue, imagJuliaValue, escapeRadius, maxIter)

        return (results, lastRealValues, lastImagValues)
 
    def juliaDistanceEstimateSingle(self, realValue, imagValue, realJuliaValue, imagJuliaValue, escapeRadius, maxIter):
        """
        Wonderfully slow implementation of distance estimate, but we need
        a fallback/baseline, so here we go.
        """
        radSquared = escapeRadius * escapeRadius
        result = 0

        realDecimal = self.decimal.Decimal(realValue)
        imagDecimal = self.decimal.Decimal(imagValue)

        zReal = self.decimal.Decimal(realJuliaValue)
        zImag = self.decimal.Decimal(imagJuliaValue)
        zrSquared = zReal * zReal
        ziSquared = zImag * zImag

        # Julia derivative inittialize to (1+0j), but mandelbrot is (0+0j), FYI
        dzReal = self.decimal.Decimal(1)
        dzImag = self.decimal.Decimal(0)
 
        for currIter in range(maxIter + 1):
            result = currIter
    
            if (zrSquared + ziSquared) > radSquared:
                break

            # Z' -> 2·Z·Z' + 1  
            # FYI: There's an extra "+1" for mandelbrot dzReal, but
            # julia dz is just 2·Z·Z'
            # (a + bi) * (c + di)
            # = ac + adi + bci - bd = ac - bd + (ad+bc)i
            partSum = zReal * dzReal - zImag * dzImag 
            newDzReal = partSum + partSum # Don't stomp dz yet
            dzImag = zReal * dzImag + zImag * dzReal 
            dzReal = newDzReal # Now safe to stomp

            # Z -> Z² + c           
            # Below is 2(z.real*z.imag) + c.imag, but without an extra multiply
            partSum = zReal * zImag
            zImag = partSum + partSum + imagDecimal 
            zReal = zrSquared - ziSquared + realDecimal

            zrSquared = zReal * zReal
            ziSquared = zImag * zImag

        # Tried to extract the distance smoothing from this a few times, 
        # but it seems a lot messier to pass around the (needed) derivative
        # as well, so we just go ahead and return an extra term here, knowing
        # this theoretically could be better handled in the 'normal' two steps.
        if result == maxIter:
            return (result, zReal, zImag)
        else:
            realPart = zReal * zReal 
            imagPart = zImag * zImag 
            zMagnitude = (realPart + imagPart).sqrt()
  
            realPart = dzReal * dzReal
            imagPart = dzImag * dzImag
            dzMagnitude = (realPart + imagPart).sqrt()

            ## Can't take logs of zMagnitude when <= 0, 
            ## and can't divide by zero-valued dzMagnitude.
            ## (though these shouldn't ever be, because of sqrt?)
            #if zMagnitude <= 0 or dzMagnitude <= 0:
            #    return (result, zReal, zImag)
            #return ((zMagnitude * (zMagnitude.ln()) / dzMagnitude), zReal, zImag)

            # Lots of commentary/options above in mandelbrot distance estimate,
            # this is just pasted in to keep in line with whatever was chosen
            # up there.
            tmpAnswer = result - (zMagnitude * (zMagnitude.ln()) / dzMagnitude)
            if result > 0:
                return (tmpAnswer, zReal, zImag)
            else:
                return (self.decimal.Decimal(0), zReal, zImag)

#    def rescaleForRange(self, rawValue, endingIter, maxIter, scaleRange):
#        ##print("rescale %s for range %s" % (str(rawValue), str(scaleRange)))
#        #if endingIter == maxIter or rawValue <= 0.0:
#        #    return 0.0
#        #else:
#        #    zoomValue = float(math.pow((1/scaleRange) * rawValue / 0.1, 0.1))
#        #    return self.clamp(zoomValue, 0.0, 1.0)
#
#        val = float(rawValue)
#        if val < 0.0:
#            val = 0.0
#        zoo = .1 
#        zoom_level = 1. / scaleRange 
#        d = self.clamp( pow(zoom_level * val/zoo,0.1), 0.0, 1.0 );
#        return float(d)
#
#    def clamp(self, num, min_value, max_value):
#       return max(min(num, max_value), min_value)
#
#    def squared_modulus(self, z):
#        return ((z.real*z.real)+(z.imag*z.imag))
    
class DiveMathSupportMPFR(DiveMathSupport):
    def __init__(self):
        super().__init__()

        # Only imports if you instantiate this DiveMathSupport subclass.
        self.mpfrlib = __import__('mpfr_fractalmath') 

        # Unlike super()'s Decimal which keeps digits of precision, 
        # mpfr natively keeps bits, so we'll keep bits too, since 
        # this number is used a lot
        self.defaultPrecisionSize = 53 
        self.currPrecision = self.defaultPrecisionSize
        self.precisionType = 'mpfr'

    # Now that Decimal is native DiveMathSupport, there are actually 
    # 2 libraries now to keep at the same precision, so also send 
    # precision changes to super()

    def setPrecision(self, newPrecision):
        super().setPrecision(newPrecision) # Also set Decimal's precision
        oldPrecision = self.currPrecision
        self.currPrecision = newPrecision
        return oldPrecision

    def precision(self):
        return self.currPrecision

    def setDigitsPrecision(self, newDigits):
        super().setDigitsPrecision(newDigits)
        newPrecision = self.digitsToBits(newDigits)
        oldPrecision = self.currPrecision
        self.currPrecision = newPrecision
        return oldPrecision
        
    def digitsPrecision(self):
        return self.bitsToDigits(self.currPrecision) 

    def julia(self, realValues, imagValues, realJuliaValue, imagJuliaValue, escapeRadius, maxIterations):
        """
        Default behavior is 2D decimal to 2D decimal.
        """
        return self.mpfrlib.julia_2d_pydecimal_to_decimal(realValues, imagValues, realJuliaValue, imagJuliaValue, escapeRadius, maxIterations, self.currPrecision)

    def juliaSingle(self, realValue, imagValue, realJuliaValue, imagJuliaValue, escapeRadius, maxIterations):
        realValues = np.array((realValue), dtype=object)
        imagValues = np.array((imagValue), dtype=object)
        return self.mpfrlib.julia_2d_pydecimal_to_decimal(realValues, imagValues, realJuliaValue, imagJuliaValue, escapeRadius, maxIterations, self.currPrecision)

    def juliaToString(self, realValues, imagValues, realJuliaValue, imagJuliaValue, escapeRadius, maxIterations):
        """ 
        Returns string types to dodge an extra string-to-decimal conversion.
        Useful when you know further processing (e.g. smoothing) will happen.
        """
        return self.mpfrlib.julia_2d_pydecimal_to_string(realValues, imagValues, realJuliaValue, imagJuliaValue, escapeRadius, maxIterations, self.currPrecision)

    def mandelbrot(self, realValues, imagValues, escapeRadius, maxIterations):
        """
        Default behavior is 2D decimal to 2D decimal.
        """
        return self.mpfrlib.mandelbrot_2d_pydecimal_to_decimal(realValues, imagValues, escapeRadius, maxIterations, self.currPrecision)

    def mandelbrotSingle(self, realValue, imagValue, escapeRadius, maxIterations):
        realValues = np.array((realValue), dtype=object)
        imagValues = np.array((imagValue), dtype=object)
        return self.mpfrlib.mandelbrot_2d_pydecimal_to_decimal(realValues, imagValues, escapeRadius, maxIterations, self.currPrecision)

    def mandelbrotToString(self, realValues, imagValues, escapeRadius, maxIterations):
        """ 
        Returns string types to dodge an extra string-to-decimal conversion.
        Useful when you know further processing (e.g. smoothing) will happen.
        """
        return self.mpfrlib.mandelbrot_2d_pydecimal_to_string(realValues, imagValues, escapeRadius, maxIterations, self.currPrecision)


class DiveMathSupportFlint(DiveMathSupport):
    """
    Overrides to instantiate flint-specific complex types

    Looks like flint types are safe to use in base's createLinspace()
    """
    def __init__(self):
        super().__init__()

        self.flint = __import__('flint') # Only imports if you instantiate this DiveMathSupport subclass.

        self.defaultPrecisionSize = FLINT_HIGH_PRECISION_SIZE
        self.flint.ctx.prec = self.defaultPrecisionSize  # Sets flint's precision (in bits)
        self.precisionType = 'flint'

    def setPrecision(self, newPrecision):
        #print(f"Setting precision: {newPrecision}")
        oldPrecision = self.flint.ctx.prec
        self.flint.ctx.prec = newPrecision
        return oldPrecision

    def precision(self):
        return self.flint.ctx.prec

    def createComplex(self, *args):
        """ 
        Flint complex only accepts a flint-style square-bracket complex
        string for instantiation.  So unless we detect that, we do some
        extra string conversions here, to be flexible and consistent.

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

                preparedReal = '0.0'
                preparedImag = '0.0'

                #print("Looking at \"%s\"" % partsString)
                if partsString.startswith('['):
                    # Flint-style 'acb' complex string detected
                    (preparedReal, preparedImag) = self.separateFlintComplexString(partsString)
                else:
                    (preparedReal, preparedImag) = self.separateOrdinaryComplexString(partsString)
    
                #print("real: %s" % preparedReal)
                #print("imag: %s" % preparedImag)
    
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


    def separateFlintComplexString(self, paramPartsString):
        partsString = paramPartsString

        lastIsImag = False
        if partsString.endswith('j'):
            lastIsImag = True
            partsString = partsString[:-1]

        # Remaining string might have an internal sign.
        # If there's no internal sign, then the whole remaining
        # string is either the real or the complex
        positiveParts = partsString.split(' + ')
        negativeParts = partsString.split(' - ')

        imagIsPositive = True
        realPart = ""
        imagPart = ""
        if len(positiveParts) == 2:
            realPart = positiveParts[0]
            imagIsPositive = True
            imagPart = positiveParts[1]
        elif len(negativeParts) == 2:
            realPart = negativeParts[0]
            imagIsPositive = False
            imagPart = negativeParts[1]
        elif len(positiveParts) == 1 and len(negativeParts) == 1:
            # No internal + or -, so it should just be a number
            if lastIsImag == True:
                realPart = '0.0'
                imagPart = partsString
            else:
                realPart = partsString
                imagPart = '0.0'
        else:
            raise ValueError("String parameter \"%s\" not identifiably a complex number, in createComplex()->separateFlintComplexString()" % paramPartsString)

        realPart = realPart.strip()
        imagPart = imagPart.strip()

        # When imag is negative, need to insert the negative
        # inside of the brackets.
        if imagPart != '0.0' and imagIsPositive == False:
            imagPart = imagPart[:1] + "-" + imagPart[1:]

        return (realPart, imagPart)

    def separateOrdinaryComplexString(self, partsString):
        # 'Normal' complex with or without parens
        # Trim off surrounding parens, if present
        if partsString.startswith('('):
            partsString = partsString[1:]
        if partsString.endswith(')'):
            partsString = partsString[:-1]
   
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
            raise ValueError("String parameter \"%s\" not identifiably a complex number, in createComplex()" % args[0])

        preparedReal = ""
        preparedImag = ""

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

        return (preparedReal, preparedImag)

    def createFloat(self, floatValue):
        return self.flint.arb(floatValue)

    def stringFromFloat(self, paramFloat):
        # Trying to clear out the error ball?!
        return self.flint.arb(paramFloat.mid(), 0).str() 

    def shorterStringFromFloat(self, paramFloat, places):
        # Trying to clear out the error ball?!
        return self.flint.arb(paramFloat.mid(), 0).str(places) 

    def floor(self, value):
        return self.flint.arb(value).floor()

    def stringFromARB(self, paramARB):
        return paramARB.str(radius=False, more=True)

    def arrayToStringArray(self, paramArray):
        stringifier = np.vectorize(self.stringFromARB)
        return stringifier(paramArray)

#    def createLinspace(self, paramFirst, paramLast, quantity):
#        """
#        """
#        print(f"Flint createLinspace {paramFirst}, {paramLast}, {quantity}")
#        dataRange = paramLast - paramFirst
#        answers = np.zeros((quantity), dtype=object)
#
#        for x in range(0, quantity):
#            answers[x] = paramFirst + dataRange * (x / (quantity - 1))
#            answers[x] = self.flint.arb(answers[x].mid(), 0)
#        return answers
#
#    def createLinspaceAroundValuesCenter(self, valuesCenter, spreadWidth, quantity):
#        """
#        """
#        firstValue = valuesCenter - spreadWidth * 0.5
#        firstValue = self.flint.arb(firstValue.mid(), 0)
#
#        lastValue = valuesCenter + spreadWidth * 0.5
#        lastValue = self.flint.arb(lastValue.mid(), 0)
#
#        print(f"createLinspaceAroundValuesCenter {valuesCenter}, {spreadWidth}, {quantity}")
#
#        return self.createLinspace(firstValue, lastValue, quantity)
#        #return self.createLinspace(valuesCenter - spreadWidth * 0.5, valuesCenter + spreadWidth * 0.5, quantity)

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
        #calculatedValue = startValue * pow(zoomAsArb, iterationsAsArb)
        #return self.flint.arb(calculatedValue.mid(), 0) # Trying to clear out the error ball?!
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
            root = endX - startX
            scaleFactor = (endY / startY) ** (1 / root)
            print(f"root: {root}\nscaleFactor: {scaleFactor}")
            debugAnswer = startY * (scaleFactor ** (targetX - startX))
            print(f"  flint root-to answer {debugAnswer}")
            return startY * (scaleFactor ** (targetX - startX))

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
        (In retrospect, the extra call isn't worth tabulating at higher
        iteration counts)
        """
        z = self.flint.acb(0,0)
        n = 0

        for currIter in range(maxIter + 1):
            n = currIter

            if float(z.abs_lower()) > escapeRadius:
                break

            z = z*z + c
      
            # Forcibly clearing the error component of the arb.
            # Custom flint implementation could be more graceful with this.
            zRealNoErr = self.flint.arb(z.real.mid(), 0)
            zImagNoErr = self.flint.arb(z.imag.mid(), 0)
            z = self.flint.acb(zRealNoErr, zImagNoErr) 

        #print("input: %s" % (str(c)))
        #print("answer: %s, lastZ: %s" % (str(n), str(z)))
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

        for currIter in range(maxIter + 1):
            n = currIter

            if float(absOfZ) > escapeRadius:
                break

            # Z' -> 2·Z·Z' + 1
            dz = 2.0 * (z*dz) + 1
            # Z -> Z² + c           
            z = z*z + c

            absOfZ = z.abs_lower()

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
        #print("rescale %s for range %s" % (str(rawValue), str(scaleRange)))
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

class DiveMathSupportFlintCustom(DiveMathSupportFlint):
    def __init__(self):
        super().__init__()
        self.precisionType = 'flintcustom'


    def mandelbrot(self, c, escapeRadius, maxIter):
        """ Slightly more efficient for HIGH maxIter values """
        #print("mandelbrot center: %s radius: %s maxIter: %s" % (str(c), str(escapeRadius), str(maxIter)))
        #print("mandelbrot maxIter: %s" % (str(maxIter)))
        (answer, lastZ, remainingPrecision) = c.our_steps_mandelbrot(escapeRadius, maxIter)
        #print("answer: %s, lastZ: %s remainingPrecision: %s" % (str(answer), str(lastZ), str(remainingPrecision)))
        #print("answer: %s, remainingPrecision: %s" % (str(answer), str(remainingPrecision)))
        return(answer, lastZ)

    def mandelbrot_arb(self, c, escapeRadius, maxIter):
        (answer, lastZ, remainingPrecision) = arb_fractalmath.mandelbrot_steps(c, escapeRadius, maxIter)
        return(answer, lastZ)

    def mandelbrot_mpfr(self, c, escapeRadius, maxIter):
        realString = c.real.str(radius=False, more=True)
        imagString = c.imag.str(radius=False, more=True)

        (answer, lastZReal, lastZImag) = mpfr_fractalmath.mandelbrot_steps(realString, imagString, escapeRadius, maxIter, self.precision())

        #print(f"precision: {self.precision()}")

        #return(answer, self.flint.acb(lastZReal, lastZImag))
        return(answer, lastZReal) # TODO: Pretty useless, but just testing
        #return(answer, lastZReals, lastZImags)

    def mandelbrot_mpfr_2d(self, numpyReals, numpyImags, escapeRadius, maxIter):

        #print(f"precision: {self.precision()}")

        (answer, lastZReals, lastZImags) = mpfr_fractalmath.mandelbrot_2d_string_to_string(numpyReals, numpyImags, escapeRadius, maxIter, self.precision())
 
        #(answer, lastZ) = mpfr_fractalmath.mandelbrot_steps(c, escapeRadius, maxIter)
        return(answer, lastZReals, lastZImags)

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


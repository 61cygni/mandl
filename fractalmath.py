# --
# File: fractalmath.py
#
# Home for math operations that should be high-precision aware
#
# --

import math

import numpy as np

#FLINT_HIGH_PRECISION_SIZE = 16 # 53 is how many bits are in float64
FLINT_HIGH_PRECISION_SIZE = 53 # 53 is how many bits are in float64
#FLINT_HIGH_PRECISION_SIZE = 200 

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

        Somehow, this seems to be ok for flint too?!
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
        Iterative multiplications of window sizes for zooming means we want to be able to
        interpolate between two points using the root if the frame count between them as
        the scale factor
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

        Looks like this implementation is able to handle flint types, now that smoothing
        is handled separately.

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

    def smoothAfterCalculation(self, endingZ, endingIter, maxIter):
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
            return endingIter + 1 - math.log(math.log2(abs(endingZ)))

class DiveMathSupportFlint(DiveMathSupport):
    """
    Overrides to instantiate flint-specific complex types

    Looks like flint types are safe to use in base's createLinspace()
    """
    def __init__(self):
        super().__init__()

        self.flint = __import__('flint') # Only imports if you instantiate this DiveMathSupport subclass.
        self.flint.prec = FLINT_HIGH_PRECISION_SIZE  # Sets flint's precision (in bits)
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

    def smoothAfterCalculation(self, endingZ, endingIter, maxIter):
        """
        This flint-specific implementation only really does flint-y logs in the smoothing.
        The Decimal-specific implementation needed some extra steps, but I've ditched that one.
        """
        if endingIter == maxIter:
            return float(maxIter)
        else:
            # The following code smooths out the colors so there aren't bands
            # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
            # Note: Results in a float. We think.
            return float(endingIter + 1 - endingZ.abs_lower().const_log2().const_log10())

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

    def smoothAfterCalculation(self, endingZ, endingIter, maxIter):
        """
        This flint-specific implementation only really does flint-y logs in the smoothing.
        The Decimal-specific implementation needed some extra steps, but I've ditched that one.
        """
        if endingIter == maxIter:
            return float(maxIter)
        else:
            # The following code smooths out the colors so there aren't bands
            # Algorithm taken from http://linas.org/art-gallery/escape/escape.html
            # Note: Results in a float. We think.
            return float(endingIter + 1 - self.gmp.log10(self.gmp.log2(self.gmp.norm(endingZ))))
            #return float(endingIter + 1 - self.gmp.log10(self.gmp.log10(self.gmp.sqrt(self.gmp.square(endingZ.real) + self.gmp.square(endingZ.imag)))))



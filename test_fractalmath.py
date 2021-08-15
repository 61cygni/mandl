import unittest

import fractalmath as fm


class TestMathSupport(unittest.TestCase):
    mathSupport = None

    @classmethod
    def setUpClass(cls):
        cls.mathSupport = fm.DiveMathSupport()

    @classmethod
    def tearDownClass(cls):
        cls.mathSupport = None

    def test_instantiations(self):
        """ 
        String-based number instantiations.
        Short enough for all implementations.
        """
        floatString = '-1.7693831791'
        complexString = '-1.7693831791+0.0042368479j'
        complexParenString = '(-1.7693831791+0.0042368479j)'
        
        complexRealString = '-1.7693831791'
        complexImagString = '0.0042368479'

        firstFloat = self.mathSupport.createFloat(floatString)
        self.assertIsNotNone(firstFloat)

        firstComplex = self.mathSupport.createComplex(complexString)
        self.assertIsNotNone(firstComplex)

        parenComplex = self.mathSupport.createComplex(complexParenString)
        self.assertIsNotNone(parenComplex)

        partwiseComplex = self.mathSupport.createComplex(complexRealString, complexImagString)
        self.assertIsNotNone(partwiseComplex)

    def test_scaleValue(self):
        """ Extra casts of returns to float() so assertAlmostEqual works """
        startValue = 5.0
        zoomFactor = 0.8

        # Zero iteration zoom, should be same as input
        self.assertEqual(startValue, float(self.mathSupport.scaleValueByFactorForIterations(startValue, zoomFactor, 0)))

        # 1 iteration zoom, should be same as single multiplication 
        self.assertAlmostEqual(startValue * zoomFactor, float(self.mathSupport.scaleValueByFactorForIterations(startValue, zoomFactor, 1)))

        # 2 iterations, should be doubly-multiplied zoom
        self.assertAlmostEqual(startValue * zoomFactor * zoomFactor, float(self.mathSupport.scaleValueByFactorForIterations(startValue, zoomFactor, 2)))


    def test_interpolateLinear(self):
        startX = 1.0
        startY = 2.0
        endX = 2.0
        endY = 4.0
        # Endpoints check
        self.assertAlmostEqual(startY, self.mathSupport.interpolate('linear', startX, startY, endX, endY, startX))
        self.assertAlmostEqual(endY, self.mathSupport.interpolate('linear', startX, startY, endX, endY, endX))

        # Not endpoint check
        self.assertAlmostEqual(2.5, self.mathSupport.interpolate('linear', startX, startY, endX, endY, 1.25))

    def test_mandelbrot(self):
        # Native python isn't accurate past ~120 iterations. 
        # So, we either reduce iterations, and require fewer decimal places to match.
        radius = 2.0

        maxIterations = 100 
        placesToMatch = 8 

        # The answer for this particular point, is supposed to be 133, but we're
        # capping iterations to 100 here.
        center = self.mathSupport.createComplex('-1.7693831791+0.0042368479j')
        (iterations, lastZee) = self.mathSupport.mandelbrot(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(100, iterations)
        self.assertAlmostEqual(0.07974379578605828, float(lastZee.real), placesToMatch) # when maxIterations == 100
        self.assertAlmostEqual(-0.03921502776351098, float(lastZee.imag), placesToMatch) # when maxIterations == 100

        # Slightly adjusted center, just to get a different answer
        center = self.mathSupport.createComplex('-1.7693831791-0.5042368479j')
        (iterations, lastZee) = self.mathSupport.mandelbrot(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(3, iterations)
        self.assertAlmostEqual(-2.182516841632256, float(lastZee.real), placesToMatch)
        self.assertAlmostEqual(2.3301940018826137, float(lastZee.imag), placesToMatch)


        # More iterations, fewer decimal places required to match for comparison
        # 3 digits seems to be near the low-end for correctness of the iteration count?
        maxIterations = 110
        placesToMatch = 3 

        # The answer for this particular point, is supposed to be 133, but we're
        # capping iterations to 110 here.
        center = self.mathSupport.createComplex('-1.7693831791+0.0042368479j')
        (iterations, lastZee) = self.mathSupport.mandelbrot(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(110, iterations)
        self.assertAlmostEqual(-1.7702357414195125, float(lastZee.real), placesToMatch) # when maxIterations == 110
        self.assertAlmostEqual(0.00893290183116224, float(lastZee.imag), placesToMatch) # when maxIterations == 110

class TestMathSupportDecimal(TestMathSupport):
    @classmethod
    def setUpClass(cls):
        cls.mathSupport = fm.DiveMathSupportDecimal()
        # Set prec to what gets the closest value for comparing to
        # native python.  It's a little arbitrary, even knowing the 
        # prec value *should* be 53 to match precision.
        cls.mathSupport.setPrecision(16)

class TestMathSupportFlint(TestMathSupport):
    @classmethod
    def setUpClass(cls):
        cls.mathSupport = fm.DiveMathSupportFlint()
        # Set prec to what gets the closest value for comparing to
        # native python.  It's a little arbitrary, even knowing the 
        # prec value *should* be 53 to match precision.
        cls.mathSupport.flint.ctx.prec = 53 

    def test_instantiations(self):
        """ 
        String-based number instantiations.
        Short enough for most all precisions.
        """
        floatString = '-1.7693831791'
        firstFloat = self.mathSupport.createFloat(floatString)
        self.assertIsNotNone(firstFloat)

        complexString = '-1.7693831791+0.0042368479j'
        firstComplex = self.mathSupport.createComplex(complexString)
        self.assertIsNotNone(firstComplex)

        complexParenString = '(-1.7693831791+0.0042368479j)'
        parenComplex = self.mathSupport.createComplex(complexParenString)
        self.assertIsNotNone(parenComplex)

        complexRealString = '-1.7693831791'
        complexImagString = '0.0042368479'
        partwiseComplex = self.mathSupport.createComplex(complexRealString, complexImagString)
        self.assertIsNotNone(partwiseComplex)

        complexFlintBracketString = '[-1.76938317910000 +/- 3.53e-16] + [0.00423684790000000 +/- 1.68e-18]j' 
        bracketComplex = self.mathSupport.createComplex(complexFlintBracketString)
        self.assertIsNotNone(bracketComplex)

        complexFlintBracketStringNeg = '[-1.76938317910000 +/- 3.53e-16] - [0.00423684790000000 +/- 1.68e-18]j' 
        bracketNegativeComplex = self.mathSupport.createComplex(complexFlintBracketStringNeg)
        self.assertIsNotNone(bracketNegativeComplex)

        complexFlintBracketRealOnly = '[-1.76938317910000 +/- 3.53e-16]' 
        bracketRealOnlyComplex = self.mathSupport.createComplex(complexFlintBracketRealOnly)
        self.assertIsNotNone(bracketRealOnlyComplex)
        #print("realOnly: %s" % complexFlintBracketRealOnly)

        complexFlintBracketImagOnly = '[0.00423684790000000 +/- 1.68e-18]j' 
        bracketImagOnlyComplex = self.mathSupport.createComplex(complexFlintBracketImagOnly)
        self.assertIsNotNone(bracketImagOnlyComplex)
        #print("imagOnly: %s" % complexFlintBracketImagOnly)

class TestMathSupportFlintCustom(TestMathSupportFlint):
    @classmethod
    def setUpClass(cls):
        cls.mathSupport = fm.DiveMathSupportFlintCustom()
        # Set prec to what gets the closest value for comparing to
        # native python.  It's a little arbitrary, even knowing the 
        # prec value *should* be 53 to match precision.
        cls.mathSupport.flint.ctx.prec = 53 

    def test_differentMandelbrots(self):
        """ 
        This checks results between different custom mandelbrot implementations.

        Currently there are 4 implementations of varying speeds,
        1.) A pure python version (in DiveMathSupport.mandelbrot())
        2.) A python-flint version (in DiveMathSupportFlint.mandelbrot())
        3.) A 'normal' cython version (in DiveMathSupportFlintCustom.mandelbrot_beginning())
        4.) A 'step-wise' cython version (in DiveMathSupportFlintCustom.mandelbrot())

        Since 2 of these (#3 and #4) belong to the custom math support, we make sure their
        values match each other here.  Could be more flexible and remember the answer, but oh well.
        """
        placesToMatch = 8 
        radius = 2.0

        maxIterations = 100 


        center = self.mathSupport.createComplex('-1.7693831791+0.0042368479j')
        # Native python basically good for ~120 iterations, using 53 bits of depth in a float 64
        (iterations, lastZee) = self.mathSupport.mandelbrot(center, radius, maxIterations)
        #print(str(iterations))
        #print("lastZee as string: %s" % str(lastZee))
        self.assertEqual(100, iterations)
        self.assertAlmostEqual(0.0797437953537099, float(lastZee.real), placesToMatch) # when maxIterations == 100
        self.assertAlmostEqual(-0.03921502719242046, float(lastZee.imag), placesToMatch) # when maxIterations == 100

        # Slightly adjusted center, just to get a different answer
        center = self.mathSupport.createComplex('-1.7693831791-0.5042368479j')
        (iterations, lastZee) = self.mathSupport.mandelbrot(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(3, iterations)
        self.assertAlmostEqual(-2.182516841632256, float(lastZee.real), placesToMatch)
        self.assertAlmostEqual(2.3301940018826137, float(lastZee.imag), placesToMatch)

        center = self.mathSupport.createComplex('-1.7693831791+0.0042368479j')
        # Native python basically good for ~120 iterations, using 53 bits of depth in a float 64
        (iterations, lastZee) = self.mathSupport.mandelbrot_beginning(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(100, iterations)
        self.assertAlmostEqual(0.0797437953537099, float(lastZee.real), placesToMatch) # when maxIterations == 100
        self.assertAlmostEqual(-0.03921502719242046, float(lastZee.imag), placesToMatch) # when maxIterations == 100

        # Slightly adjusted center, just to get a different answer
        center = self.mathSupport.createComplex('-1.7693831791-0.5042368479j')
        (iterations, lastZee) = self.mathSupport.mandelbrot_beginning(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(3, iterations)
        self.assertAlmostEqual(-2.182516841632256, float(lastZee.real), placesToMatch)
        self.assertAlmostEqual(2.3301940018826137, float(lastZee.imag), placesToMatch)

        #print("Done comparing")

if __name__ == '__main__':
    unittest.main()


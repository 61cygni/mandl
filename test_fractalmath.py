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
        maxIterations = 100 
        placesToMatch = 9 

        radius = 2.0

        # The answer for this particular point, is supposed to be 133, but we're
        # capping iterations to 128 here.
        center = self.mathSupport.createComplex('-1.7693831791+0.0042368479j')
        (iterations, lastZee) = self.mathSupport.mandelbrot(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(100, iterations)
        self.assertAlmostEqual(1.3599199256223002, float(lastZee.real), placesToMatch)
        self.assertAlmostEqual(-0.0159758949202937, float(lastZee.imag), placesToMatch)

        # Slightly adjusted center, just to get a different answer
        center = self.mathSupport.createComplex('-1.7693831791-0.5042368479j')
        (iterations, lastZee) = self.mathSupport.mandelbrot(center, radius, maxIterations)
        #print(str(iterations))
        #print(str(lastZee))
        self.assertEqual(3, iterations)
        self.assertAlmostEqual(-2.182516841632256, float(lastZee.real), placesToMatch)
        self.assertAlmostEqual(2.3301940018826137, float(lastZee.imag), placesToMatch)

class TestMathSupportFlint(TestMathSupport):
    @classmethod
    def setUpClass(cls):
        cls.mathSupport = fm.DiveMathSupportFlint()
        # Set prec to what gets the closest value for comparing to
        # native python.  It's a little arbitrary, even knowing the 
        # prec value *should* be 53 to match precision.
        cls.mathSupport.flint.ctx.prec = 53 

class TestMathSupportFlintCustom(TestMathSupport):
    @classmethod
    def setUpClass(cls):
        cls.mathSupport = fm.DiveMathSupportFlintCustom()
        # Set prec to what gets the closest value for comparing to
        # native python.  It's a little arbitrary, even knowing the 
        # prec value *should* be 53 to match precision.
        cls.mathSupport.flint.ctx.prec = 53 


if __name__ == '__main__':
    unittest.main()


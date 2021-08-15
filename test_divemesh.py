
import unittest

import pickle

import fractalmath as fm
from divemesh import *

class TestDiveMesh(unittest.TestCase):
    pyMathSupport = None
    flintMathSupport = None

    @classmethod
    def setUpClass(cls):
        cls.pyMathSupport = fm.DiveMathSupport()
        cls.flintMathSupport = fm.DiveMathSupportFlint()

    @classmethod
    def tearDownClass(cls):
        cls.pyMathSupport = None
        cls.flintMathSupport = None

    def test_pythonGeneratorPickle(self):
        centerFloatString = '-1.769383179195515018213'
        baseWidthString = '2.0'

        pyCenter = self.pyMathSupport.createFloat(centerFloatString)
        pyWidth = self.pyMathSupport.createFloat(baseWidthString)

        uniformGen = MeshGeneratorUniform(self.pyMathSupport, 'width', pyCenter, pyWidth)

        pickleValue = pickle.dumps(uniformGen)
        otherGen = pickle.loads(pickleValue)

        self.assertEqual(uniformGen.valuesCenter, otherGen.valuesCenter)
        self.assertEqual(uniformGen.baseWidth, otherGen.baseWidth)

    def test_flintGeneratorPickle(self):
        # Maybe better to use a bracket-arb string here?  But doesn't matter?
        centerFloatString = '-1.769383179195515018213'
        baseWidthString = '2.0'

        flintCenter = self.flintMathSupport.createFloat(centerFloatString)
        flintWidth = self.flintMathSupport.createFloat(baseWidthString)

        uniformGen = MeshGeneratorUniform(self.flintMathSupport, 'width', flintCenter, flintWidth)

        pickleValue = pickle.dumps(uniformGen)
        otherGen = pickle.loads(pickleValue)

        self.assertEqual(float(uniformGen.valuesCenter), float(otherGen.valuesCenter))
        self.assertEqual(float(uniformGen.baseWidth), float(otherGen.baseWidth))

    def test_pythonDiveMeshPickle(self):
        centerWidthString = '-1.769383179195515018213'
        centerHeightString = '0.00423684791873677221'

        baseRealWidthString = '5.0'
        baseImagWidthString = '3.0'

        mathSupport = self.pyMathSupport

        realCenter = mathSupport.createFloat(centerWidthString)
        realWidth = mathSupport.createFloat(baseRealWidthString)
        realGen = MeshGeneratorUniform(mathSupport, 'width', realCenter, realWidth)

        imagCenter = mathSupport.createFloat(centerHeightString)
        imagWidth = mathSupport.createFloat(baseImagWidthString)
        imagGen = MeshGeneratorUniform(mathSupport, 'height', imagCenter, imagWidth)
        
        centerComplexString = '-1.769383179195515018213+0.00423684791873677221j'
        pyComplex = mathSupport.createComplex(centerComplexString)

        meshWidth = 320
        meshHeight = 240

        diveMesh = DiveMesh(meshWidth, meshHeight, pyComplex, realGen, imagGen, mathSupport)

        pickleValue = pickle.dumps(diveMesh)
        loadedMesh = pickle.loads(pickleValue)

        self.assertEqual(int(diveMesh.meshWidth), int(loadedMesh.meshWidth))
        self.assertEqual(int(diveMesh.meshHeight), int(loadedMesh.meshHeight))

        self.assertEqual(float(diveMesh.center.real), float(loadedMesh.center.real))
        self.assertEqual(float(diveMesh.center.imag), float(loadedMesh.center.imag))
        self.assertEqual(float(diveMesh.center.imag), float(loadedMesh.center.imag))

        #print("realGen: \"%s\"" % str(diveMesh.realMeshGenerator))
        #print("imagGen: \"%s\"" % str(diveMesh.imagMeshGenerator))
        #print("after realGen: \"%s\"" % str(loadedMesh.realMeshGenerator))
        #print("after imagGen: \"%s\"" % str(loadedMesh.imagMeshGenerator))

    def test_flintDiveMeshPickle(self):
        # Maybe better to use a bracket-arb string here?  But doesn't matter?
        centerWidthString = '-1.769383179195515018213'
        centerHeightString = '0.00423684791873677221'

        baseRealWidthString = '5.0'
        baseImagWidthString = '3.0'

        mathSupport = self.flintMathSupport

        realCenter = mathSupport.createFloat(centerWidthString)
        realWidth = mathSupport.createFloat(baseRealWidthString)
        realGen = MeshGeneratorUniform(mathSupport, 'width', realCenter, realWidth)

        imagCenter = mathSupport.createFloat(centerHeightString)
        imagWidth = mathSupport.createFloat(baseImagWidthString)
        imagGen = MeshGeneratorUniform(mathSupport, 'height', imagCenter, imagWidth)
        
        centerComplexString = '-1.769383179195515018213+0.00423684791873677221j'
        pyComplex = mathSupport.createComplex(centerComplexString)

        meshWidth = 320
        meshHeight = 240

        diveMesh = DiveMesh(meshWidth, meshHeight, pyComplex, realGen, imagGen, mathSupport)

        pickleValue = pickle.dumps(diveMesh)
        loadedMesh = pickle.loads(pickleValue)

        self.assertEqual(int(diveMesh.meshWidth), int(loadedMesh.meshWidth))
        self.assertEqual(int(diveMesh.meshHeight), int(loadedMesh.meshHeight))

        self.assertEqual(float(diveMesh.center.real), float(loadedMesh.center.real))
        self.assertEqual(float(diveMesh.center.imag), float(loadedMesh.center.imag))
        self.assertEqual(float(diveMesh.center.imag), float(loadedMesh.center.imag))

        #print("realGen: \"%s\"" % str(diveMesh.realMeshGenerator))
        #print("imagGen: \"%s\"" % str(diveMesh.imagMeshGenerator))
        #print("after realGen: \"%s\"" % str(loadedMesh.realMeshGenerator))
        #print("after imagGen: \"%s\"" % str(loadedMesh.imagMeshGenerator))

if __name__ == '__main__':
    unittest.main()

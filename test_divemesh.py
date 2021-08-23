
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

    def test_meshShape(self):
        meshWidth = 5
        meshHeight = 3
        widthGen = MeshGeneratorUniform(self.pyMathSupport, 'width', 0.5, 0.5)
        heightGen = MeshGeneratorUniform(self.pyMathSupport, 'height', 2.0, 4.0)
        comparisonArrayString = """[[(0.25+0j) (0.375+0j) (0.5+0j) (0.625+0j) (0.75+0j)]
 [(0.25+2j) (0.375+2j) (0.5+2j) (0.625+2j) (0.75+2j)]
 [(0.25+4j) (0.375+4j) (0.5+4j) (0.625+4j) (0.75+4j)]]"""

        diveMesh = DiveMesh(meshWidth, meshHeight, widthGen, heightGen, self.pyMathSupport)

        meshArray = diveMesh.generateMesh()
        meshShape = meshArray.shape
        #print(meshArray)

        # numpy array.shape returns (rows, columns), which is (height, width)
        self.assertEqual(meshShape[1], meshWidth)
        self.assertEqual(meshShape[0], meshHeight)

        self.assertEqual(str(meshArray), comparisonArrayString)

    def test_pythonGeneratorPickle(self):
        centerFloatString = '-1.7693831791955'
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
        centerFloatString = '-1.7693831791955'
        baseWidthString = '2.0'

        flintCenter = self.flintMathSupport.createFloat(centerFloatString)
        flintWidth = self.flintMathSupport.createFloat(baseWidthString)

        uniformGen = MeshGeneratorUniform(self.flintMathSupport, 'width', flintCenter, flintWidth)

        pickleValue = pickle.dumps(uniformGen)
        otherGen = pickle.loads(pickleValue)

        self.assertEqual(float(uniformGen.valuesCenter), float(otherGen.valuesCenter))
        self.assertEqual(float(uniformGen.baseWidth), float(otherGen.baseWidth))

    def test_pythonDiveMeshPickle(self):
        centerWidthString = '-1.7693831791955'
        centerHeightString = '0.0042368479187'

        baseRealWidthString = '5.0'
        baseImagWidthString = '3.0'

        mathSupport = self.pyMathSupport

        realCenter = mathSupport.createFloat(centerWidthString)
        realWidth = mathSupport.createFloat(baseRealWidthString)
        realGen = MeshGeneratorUniform(mathSupport, 'width', realCenter, realWidth)

        imagCenter = mathSupport.createFloat(centerHeightString)
        imagWidth = mathSupport.createFloat(baseImagWidthString)
        imagGen = MeshGeneratorUniform(mathSupport, 'height', imagCenter, imagWidth)
        
        meshWidth = 320
        meshHeight = 240

        diveMesh = DiveMesh(meshWidth, meshHeight, realGen, imagGen, mathSupport)

        pickleValue = pickle.dumps(diveMesh)
        loadedMesh = pickle.loads(pickleValue)

        self.assertEqual(int(diveMesh.meshWidth), int(loadedMesh.meshWidth))
        self.assertEqual(int(diveMesh.meshHeight), int(loadedMesh.meshHeight))

        #print("realGen: \"%s\"" % str(diveMesh.realMeshGenerator))
        #print("imagGen: \"%s\"" % str(diveMesh.imagMeshGenerator))
        #print("after realGen: \"%s\"" % str(loadedMesh.realMeshGenerator))
        #print("after imagGen: \"%s\"" % str(loadedMesh.imagMeshGenerator))

    def test_flintDiveMeshPickle(self):
        # Maybe better to use a bracket-arb string here?  But doesn't matter?
        centerWidthString = '-1.7693831791955'
        centerHeightString = '0.0042368479187'

        baseRealWidthString = '5.0'
        baseImagWidthString = '3.0'

        mathSupport = self.flintMathSupport

        realCenter = mathSupport.createFloat(centerWidthString)
        realWidth = mathSupport.createFloat(baseRealWidthString)
        realGen = MeshGeneratorUniform(mathSupport, 'width', realCenter, realWidth)

        imagCenter = mathSupport.createFloat(centerHeightString)
        imagWidth = mathSupport.createFloat(baseImagWidthString)
        imagGen = MeshGeneratorUniform(mathSupport, 'height', imagCenter, imagWidth)
        
        meshWidth = 320
        meshHeight = 240

        diveMesh = DiveMesh(meshWidth, meshHeight, realGen, imagGen, mathSupport)

        pickleValue = pickle.dumps(diveMesh)
        loadedMesh = pickle.loads(pickleValue)

        self.assertEqual(int(diveMesh.meshWidth), int(loadedMesh.meshWidth))
        self.assertEqual(int(diveMesh.meshHeight), int(loadedMesh.meshHeight))

        #print("realGen: \"%s\"" % str(diveMesh.realMeshGenerator))
        #print("imagGen: \"%s\"" % str(diveMesh.imagMeshGenerator))
        #print("after realGen: \"%s\"" % str(loadedMesh.realMeshGenerator))
        #print("after imagGen: \"%s\"" % str(loadedMesh.imagMeshGenerator))

if __name__ == '__main__':
    unittest.main()

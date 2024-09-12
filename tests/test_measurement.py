import unittest
import numpy as np

from ggpeps.measurement import Measurement

class TestMeasurements(unittest.TestCase):
    def test_add(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for i in range(10):
            meas1.append(1)
            meas2.append(2)
        self.assertEqual(meas1.mean(),1)
        self.assertEqual(meas1.var(),0)
        self.assertEqual(meas1.std(),0)
        self.assertEqual(len(meas1),10)
        self.assertEqual(meas2.mean(),2)
        self.assertEqual(meas2.var(),0)
        self.assertEqual(meas2.std(),0)
        self.assertEqual(len(meas2),5)

    def test_random_scalar(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for _ in range(1000):
            rnd=np.random.rand()
            meas1.append(rnd)
            meas2.append(rnd)
        self.assertAlmostEqual(meas1.mean(),meas2.mean())

    def test_random_array(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for _ in range(1000):
            rnd=np.random.rand(2,2)
            meas1.append(rnd)
            meas2.append(rnd)
        self.assertTrue(np.allclose(meas1.mean(),meas2.mean()))

    def test_mul_meas(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",1)
        for i in range(10):
            meas1.append(1.5)
            meas2.append(2)
        meas3=meas1*meas2
        for i in range(10):
            self.assertAlmostEqual(3,meas3.datavec[i])

    def test_sub_meas(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",1)
        for i in range(10):
            meas1.append(1)
            meas2.append(2)
        meas3 = meas1 - meas2
        for i in range(10):
            self.assertAlmostEqual(-1,meas3.datavec[i])

    def test_add_meas(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",1)
        for i in range(10):
            meas1.append(1)
            meas2.append(2)
        meas3 = meas1 + meas2
        for i in range(10):
            self.assertAlmostEqual(3,meas3.datavec[i])

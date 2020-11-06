import numpy         as np
import numpy.random  as npr
import numpy.testing as nptest
import unittest      as ut

from gray import (
  right_shift,
  binary2gray,
  gray2binary,
)

class TestRightShift(ut.TestCase):

  def test_basic_1d(self):
    x = np.ones(4)
    y = right_shift(x)
    t = np.array([0, 1, 1, 1])

    nptest.assert_array_equal(y, t)

  def test_basic_2d_ax1(self):
    x = np.ones((4,5))
    y = right_shift(x, axis=1)
    t = np.hstack([ np.zeros((4,1)), np.ones((4,4)) ])

    nptest.assert_array_equal(y, t)

  def test_basic_2d_ax0(self):
    x = np.ones((4,5))
    y = right_shift(x, axis=0)
    t = np.vstack([ np.zeros((1,5)), np.ones((3,5)) ])

    nptest.assert_array_equal(y, t)

  def test_k2_1d(self):
    x = np.ones(4)
    y = right_shift(x, k=2)
    t = np.array([0, 0, 1, 1])

    nptest.assert_array_equal(y, t)

  def test_k2_2d_ax1(self):
    x = np.ones((4,5))
    y = right_shift(x, k=2, axis=1)
    t = np.hstack([ np.zeros((4,2)), np.ones((4,3)) ])

    nptest.assert_array_equal(y, t)

  def test_k2_2d_ax0(self):
    x = np.ones((4,5))
    y = right_shift(x, k=2, axis=0)
    t = np.vstack([ np.zeros((2,5)), np.ones((2,5)) ])

    nptest.assert_array_equal(y, t)

  def test_extra_1d(self):
    x = np.ones(4)
    y = right_shift(x, k=5)
    t = np.zeros(4)

    nptest.assert_array_equal(y, t)

  def test_extra_2d_ax1(self):
    x = np.ones((4,5))
    y = right_shift(x, k=6, axis=1)
    t = np.zeros((4,5))

    nptest.assert_array_equal(y, t)

  def test_extra_2d_ax0(self):
    x = np.ones((4,5))
    y = right_shift(x, k=6, axis=0)
    t = np.zeros((4,5))

    nptest.assert_array_equal(y, t)

class TestBinary2Gray(ut.TestCase):

  def test_basic_0000(self):
    x = np.zeros(4)
    y = binary2gray(x)
    t = np.zeros(4)

    nptest.assert_array_equal(y, t)

  def test_basic_11111(self):
    x = np.ones(5)
    y = binary2gray(x)
    t = np.array([1, 0, 0, 0, 0])

    nptest.assert_array_equal(y, t)

  def test_basic_1010(self):
    x = np.array([1, 0, 1, 0])
    y = binary2gray(x)
    t = np.array([1, 1, 1, 1])

    nptest.assert_array_equal(y, t)

  def test_basic_101001(self):
    x = np.array([1, 0, 1, 0, 0, 1])
    y = binary2gray(x)
    t = np.array([1, 1, 1, 1, 0, 1])

    nptest.assert_array_equal(y, t)

  def test_basic_0101010101010101(self):
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 ])
    y = binary2gray(x)
    t = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])

    nptest.assert_array_equal(y, t)

class TestGray2Binary(ut.TestCase):

  def test_basic_0000(self):
    x = np.zeros(4)
    y = gray2binary(x)
    t = np.zeros(4)

    nptest.assert_array_equal(y, t)

  def test_basic_11111(self):
    x = np.array([1, 0, 0, 0, 0])
    y = gray2binary(x)
    t = np.ones(5)

    nptest.assert_array_equal(y, t)


  def test_basic_1010(self):
    x = np.array([1, 1, 1, 1])
    y = gray2binary(x)
    t = np.array([1, 0, 1, 0])

    nptest.assert_array_equal(y, t)

  def test_basic_101001(self):
    x = np.array([1, 1, 1, 1, 0, 1])
    y = gray2binary(x)
    t = np.array([1, 0, 1, 0, 0, 1])

    nptest.assert_array_equal(y, t)

  def test_basic_0101010101010101(self):
    x = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])
    y = gray2binary(x)
    t = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 ])

    nptest.assert_array_equal(y, t)

  def test_random(self):
    npr.seed(1)
    mean = 30
    for ii in range(1000):
      length = npr.geometric(1/(mean-2)) + 2
      binary1 = npr.rand(length) < 0.5
      gray    = binary2gray(binary1)
      binary2 = gray2binary(gray)
      nptest.assert_equal(binary1, binary2)

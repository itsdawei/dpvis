import numpy as np
import pytest

from dp import DPArray


@pytest.mark.parametrize("shape", [(2, 2), (2, 3, 4)])
def test_constructor_shape(shape):
    dp = DPArray(shape)
    assert dp.arr.shape == shape


@pytest.mark.parametrize("dtype", [("f", np.float32), ("d", np.float64)],
                         ids=["f", "d"])
def test_str_dtype(dtype):
    str_dtype, np_dtype = dtype
    dp = DPArray((2, 2), dtype=str_dtype)
    assert dp.dtype == np_dtype


def test_read_write():
    dp = DPArray(10)
    truth = np.arange(10)**2

    # Write
    for i in range(10):
        dp[i] = i**2

    assert np.all(dp.arr == truth)

    # Read
    for i in range(10):
        assert dp[i] == i**2


def test_equal_to():
    dp_a = DPArray(10)
    dp_b = DPArray(10)
    truth = np.arange(10)**2

    for i in range(10):
        dp_a[i] = i**2
        dp_b[i] = i**2

    # Direct comparision between DPArray and numpy array.
    assert np.all(dp_a == truth)

    # Direct comparision between DPArray and DPArray.
    assert np.all(dp_a == dp_b)

    dp_a[0] = 100

    # Test for not equal.
    assert not np.all(dp_a == truth)
    assert not np.all(dp_a == dp_b)


def test_not_equal_to():
    dp_a = DPArray(10)
    dp_b = DPArray(10)
    truth = np.arange(10)**2

    for i in range(10):
        dp_a[i] = i**2
        dp_b[i] = i**2

    # Direct comparision between DPArray and numpy array.
    assert not np.any(dp_a != truth)

    # Direct comparision between DPArray and DPArray.
    assert not np.any(dp_a != dp_b)

    dp_a[0] = 100

    # Test for not equal.
    assert np.any(dp_a != truth)
    assert np.any(dp_a != dp_b)


def test_numpy_slicing_1d():
    dp = DPArray(10)
    truth = np.arange(10)**2

    for i in range(10):
        dp[i] = i**2

    assert dp[-1] == truth[-1]
    assert np.all(dp[::2] == truth[::2])
    assert np.all(dp[4:6] == truth[4:6])
    assert np.all(dp[4:] == truth[4:])
    assert np.all(dp[:6] == truth[:6])


def test_numpy_slicing_2d():
    dp = DPArray((100, 2))
    xy = np.mgrid[0:10:1, 0:10:1].reshape(2, -1).T

    for x in range(10):
        for y in range(10):
            dp[10 * x + y, 0] = x
            dp[10 * x + y, 1] = y

    assert np.all(dp == xy)

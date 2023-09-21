"""Tests the methods in DPArray."""
import numpy as np
import pytest

from dp import DPArray, Logger


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


def test_read_write():
    dp = DPArray(10)
    truth = np.arange(10)**2

    # Write
    for i in range(10):
        dp[i] = i**2

    assert np.all(dp == truth)

    # Read
    for i in range(10):
        assert dp[i] == i**2


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


def test_numpy_indexing_2d():
    dp = DPArray((100, 2))
    truth = np.mgrid[0:10:1, 0:10:1].reshape(2, -1).T

    for x in range(10):
        for y in range(10):
            dp[10 * x + y, 0] = x
            dp[10 * x + y, 1] = y

    assert np.all(dp == truth)
    assert dp[0, 1] == truth[0, 1]
    assert dp[10, 1] == truth[10, 1]


# @pytest.mark.parametrize("slice_1",
#                          [np.s_[::2], np.s_[4:6], np.s_[4:], np.s_[:6], 5],
#                          ids=["a", "b", "c", "d", "e"])
# @pytest.mark.parametrize("slice_2",
#                          [np.s_[::2], np.s_[4:6], np.s_[4:], np.s_[:6], 1],
#                          ids=["a", "b", "c", "d", "e"])
# def test_numpy_slicing_2d(slice_1, slice_2):
#     dp = DPArray((100, 2))
#     truth = np.mgrid[0:10:1, 0:10:1].reshape(2, -1).T

#     for x in range(10):
#         for y in range(10):
#             dp[10 * x + y, 0] = x
#             dp[10 * x + y, 1] = y

#     nd_slice = (slice_1, slice_2)
#     assert np.all(dp[nd_slice] == truth[nd_slice])


def test_arr_return_copy():
    dp = DPArray(10)
    truth = np.arange(10)**2

    for i in range(10):
        dp[i] = i**2

    copy = dp.arr
    copy[0] = 100

    assert np.any(dp == truth)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f", "d"])
def test_dtype_assignment(dtype):
    dp = DPArray(10, dtype=dtype)
    for i in range(10):
        dp[i] = i**2

    with pytest.raises(ValueError):
        dp[0] = "test"

    assert isinstance(dp[0], dp.dtype)
    assert dp.dtype == dp.arr.dtype


@pytest.mark.parametrize("len, funcs", [(10, [lambda x: x + 2, lambda x: x]), (5, [lambda x: 2 * x, lambda x: x])], ids=["a", "b"])
def test_max(len, funcs):
    dp = DPArray(len)
    dp[0] = 0
    dp[1] = 1

    with pytest.raises(ValueError, match="Expecting reference to at least one index"):
        dp.max(idx=2, refs=[])
    
    with pytest.raises(ValueError, match="Expected refs and preprocessing of same length or single preprocessing callable."):
        dp.max(idx=2, refs=[0, 1], preprocessing=[lambda x: x])

    for i in range(2, len):
        dp.max(idx=i, refs=[i - 2, i - 1], preprocessing=funcs)
        assert dp.arr[i] == max(funcs[0](dp.arr[i - 2]), funcs[1](dp.arr[i - 1]))
    


# Logger related tests #


def test_constructor_custom_logger():
    logger = Logger()
    dp1 = DPArray(10, "dp1", logger=logger)
    dp2 = DPArray(10, "dp2", logger=logger)
    assert dp1.logger == dp2.logger == logger
    assert logger.array_shapes == {"dp1": 10, "dp2": 10}


def test_constructor_default_logger():
    dp1 = DPArray(10, "dp1")
    dp2 = DPArray(10, "dp2", logger=dp1.logger)
    assert dp1.logger == dp2.logger
    assert dp1.logger.array_shapes == {"dp1": 10, "dp2": 10}

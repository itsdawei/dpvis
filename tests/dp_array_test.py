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


@pytest.mark.parametrize("slice_1",
                         [np.s_[::2], np.s_[4:6], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
@pytest.mark.parametrize("slice_2",
                         [np.s_[::2], np.s_[4:6], np.s_[4:], np.s_[:6], 1],
                         ids=["a", "b", "c", "d", "e"])
def test_numpy_slicing_2d(slice_1, slice_2):
    dp = DPArray((100, 2))
    truth = np.mgrid[0:10:1, 0:10:1].reshape(2, -1).T

    for x in range(10):
        for y in range(10):
            dp[10 * x + y, 0] = x
            dp[10 * x + y, 1] = y

    nd_slice = (slice_1, slice_2)
    assert np.all(dp[nd_slice] == truth[nd_slice])


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


def test_max_empty_arrays_error():
    dp = DPArray(10)
    dp[0] = 0
    dp[1] = 1
    with pytest.raises(ValueError):
        dp[2] = dp.max([], [])


def test_max_arrays_size_error():
    dp = DPArray(10)
    dp[0] = 0
    dp[1] = 1
    with pytest.raises(ValueError):
        dp[2] = dp.max([0, 1], [0])


def test_max_cmp():
    dp = DPArray(10)
    dp[0] = 0
    dp[1] = 1

    for i in range(2, 10):
        indices = [i - 2, i - 1]
        elements = [dp[indices[0]] + 2, 2 * dp[indices[1]]]
        dp[i] = dp.max(indices, elements)
        assert dp.arr[i] == max(*elements)


def add_const(x, const):
    """
    Add a constant to x. Used for partial functions.
    """
    return x + const


@pytest.mark.parametrize("max_min", ["max", "min"], ids=["max", "min"])
@pytest.mark.parametrize(
    "r, max_val, min_val",
    [(np.array([[4, 3, 1], [5, 2, 1], [1, 2, 1]]), 14, 10),
     (np.array([[3, 4, 0, 0, 5], [4, 1, 2, 4, 4], [5, 1, 5, 5, 4],
                [2, 1, 1, 1, 4], [0, 0, 4, 3, 5]]), 36, 20)])
def test_max_min_2d(max_min, r, max_val, min_val):
    """
    Given a reward matrix r, start at index (0, 0) (top left).
    Find the strategy yielding the largest reward when
    constrained to moving down and right. The largest
    reward acheivable is given by truth
    """
    h, w = r.shape[0] + 1, r.shape[1] + 1
    dp = DPArray((h, w))

    if max_min == "max":
        dp_cmp = dp.max
        truth = max_val
    else:
        dp_cmp = dp.min
        truth = min_val

    # Base cases.
    base_case = 0 if max_min == "max" else 1000
    for i in range(w):
        dp[0, i] = base_case
    for i in range(h):
        dp[i, 0] = base_case
    dp[1, 1] = r[0, 0]

    # Note that dp and r indicies are off by one.
    for i in range(1, h):
        for j in range(1, w):
            if i == 1 and j == 1:
                continue
            indices = [(i - 1, j), (i, j - 1)]
            elements = [
                dp[indices[0]] + r[i - 1, j - 1],
                dp[indices[1]] + r[i - 1, j - 1],
            ]
            dp[i, j] = dp_cmp(indices, elements)

    assert dp.arr[h - 1, w - 1] == truth

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


def test_reference_undefined_element():
    dp = DPArray(10, array_name="arr1")
    dp[0] = 0
    dp[1] = 1

    dp[2] = dp[1] + dp[0]

    with pytest.warns(RuntimeWarning):
        assert np.isnan(dp[4])

    dp[4] = dp[1] + dp[2]
    assert dp[4] == 2

    with pytest.warns(RuntimeWarning):
        temp = dp[5:7]
        assert np.all(np.isnan(temp))

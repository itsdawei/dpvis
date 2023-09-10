"""Tests the methods in DPArray."""
import numpy as np
import pytest

from dp import DPArray, Logger, Op


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


def text_occupied_arr():
    truth = np.zeros_like(10, dtype=bool)
    dp = DPArray(shape=10)

    for i in range(0, 10, 2):
        truth[i] = True
        dp[i] = i

        assert np.all(dp.occupied_arr == truth)

    for i in range(-1, 9, -2):
        truth[i] = True
        dp[i] = i

        assert np.all(dp.occupied_arr == truth)


@pytest.mark.parametrize(
        "shape, dtype, indices",
        [(5, np.float32, {2, 4}),
         ((5, 3), np.float64, {(0, 0), (3, 2), (4, 2)}),
         ((3, 4, 5), np.float32, {(0, 1, 0), (2, 2, 0)})],
        ids=["a", "b", "c"])
def test_to_csv(tmp_path, shape, dtype, indices):
    file = tmp_path / "test.csv"
    
    truth = DPArray(shape=shape, dtype=dtype)
    for index in indices:
        truth[index] = 1
    truth.save_csv(file, fmt='%.0f')

    with open(file) as f:
        header = f.readline()
        assert header == "# shape=" + str(truth.arr.shape) + ", dtype=" + str(dtype) + "\n"

    arr = np.loadtxt(file, delimiter=",", dtype=dtype, skiprows=1)
    arr = arr.reshape(shape)

    assert np.all(truth.arr[truth.occupied_arr] == arr[truth.occupied_arr])
    assert np.all(np.isnan(arr[~truth.occupied_arr]))


# Logger related tests #


def test_constructor_custom_logger():
    logger = Logger()
    dp1 = DPArray(10, "dp1", logger=logger)
    dp2 = DPArray(10, "dp2", logger=logger)
    assert dp1.logger == dp2.logger == logger
    assert logger.array_names == {"dp1", "dp2"}


def test_constructor_default_logger():
    dp1 = DPArray(10, "dp1")
    dp2 = DPArray(10, "dp2", logger=dp1.logger)
    assert dp1.logger == dp2.logger
    assert dp1.logger.array_names == {"dp1", "dp2"}


@pytest.mark.parametrize("s", [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
def test_slice_logging(s):
    dp = DPArray(10)

    dp[s] = 1
    if isinstance(s, int):
        s = np.s_[s:s + 1]
    truth = set(i for i in range(*s.indices(10)))
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp_array": truth}}
    assert len(dp.logger.logs) == 1


@pytest.mark.parametrize("slice_1",
                         [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
@pytest.mark.parametrize("slice_2",
                         [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 1],
                         ids=["a", "b", "c", "d", "e"])
def test_2d_slice_logging(slice_1, slice_2):
    dp = DPArray((10, 10))

    dp[slice_1, slice_2] = 1
    if isinstance(slice_1, int):
        slice_1 = np.s_[slice_1:slice_1 + 1]
    if isinstance(slice_2, int):
        slice_2 = np.s_[slice_2:slice_2 + 1]
    truth = {(i, j)
             for i in range(*slice_1.indices(10))
             for j in range(*slice_2.indices(10))}
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp_array": truth}}
    assert len(dp.logger.logs) == 1





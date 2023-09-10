import numpy as np
import pytest

from dp import DPArray, Op


def test_overwrite_index_log():
    dp = DPArray(10, "dp")

    dp[0] = 1
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0}}}

    # Overwriting same index with same op does not create additional log.
    dp[0] = 2
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0}}}
    assert len(dp.logger.logs) == 1


def test_array_read_write_log():
    dp = DPArray(10, "dp")

    dp[0] = 1
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0}}}
    assert len(dp.logger.logs) == 1

    dp[1] = 2
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0, 1}}}

    temp = dp[1]
    assert dp.logger.logs[0]["op"] == Op.WRITE
    assert dp.logger.logs[0]["idx"] == {"dp": {0, 1}}
    assert dp.logger.logs[1]["op"] == Op.READ
    assert dp.logger.logs[1]["idx"] == {"dp": {1}}
    assert len(dp.logger.logs) == 2

    dp[2] = temp
    assert dp.logger.logs[2]["op"] == Op.WRITE
    assert dp.logger.logs[2]["idx"] == {"dp": {2}}
    assert len(dp.logger.logs) == 3


def test_2d_array_read_write_log():
    dp = DPArray((10, 10), "name")
    dp[0, 0] = 1
    assert dp.logger.logs[0]["op"] == Op.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": {(0, 0)}}
    assert len(dp.logger.logs) == 1

    temp = dp[0, 0]
    assert dp.logger.logs[0]["op"] == Op.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": {(0, 0)}}
    assert dp.logger.logs[1]["op"] == Op.READ
    assert dp.logger.logs[1]["idx"] == {"name": {(0, 0)}}
    assert len(dp.logger.logs) == 2

    dp[3, 6] = temp
    assert dp.logger.logs[2]["op"] == Op.WRITE
    assert dp.logger.logs[2]["idx"] == {"name": {(3, 6)}}
    assert len(dp.logger.logs) == 3


def test_array_read_write_log_multiple_arrays():
    dp1 = DPArray(10, "dp_1")
    dp2 = DPArray(10, "dp_2", logger=dp1.logger)

    dp1[0] = 1
    dp2[0] = 2
    assert dp1.logger.logs[0]["op"] == Op.WRITE
    assert dp1.logger.logs[0]["idx"] == {"dp_1": {0}, "dp_2": {0}}

    dp1[1] = 3
    dp2[1] = dp1[1]
    assert dp1.logger.logs[0]["op"] == Op.WRITE
    assert dp1.logger.logs[0]["idx"] == {"dp_1": {0, 1}, "dp_2": {0}}
    assert dp1.logger.logs[1]["op"] == Op.READ
    assert dp1.logger.logs[1]["idx"]["dp_1"] == {1}
    assert dp1.logger.logs[2]["op"] == Op.WRITE
    assert dp1.logger.logs[2]["idx"]["dp_2"] == {1}
    assert len(dp1.logger.logs) == 3


def test_array_slice_read_write_log():
    dp = DPArray(10)
    dp[0:5] = 1
    assert dp.logger.logs[0]["op"] == Op.WRITE
    assert dp.logger.logs[0]["idx"] == {"dp_array": set([0, 1, 2, 3, 4])}
    assert len(dp.logger.logs) == 1

    temp = dp[1:4]
    assert dp.logger.logs[0]["op"] == Op.WRITE
    assert dp.logger.logs[0]["idx"] == {"dp_array": set([0, 1, 2, 3, 4])}
    assert dp.logger.logs[1]["op"] == Op.READ
    assert dp.logger.logs[1]["idx"] == {"dp_array": set([1, 2, 3])}
    assert len(dp.logger.logs) == 2


@pytest.mark.parametrize("slice_1",
                         [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
@pytest.mark.parametrize("slice_2",
                         [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 1],
                         ids=["a", "b", "c", "d", "e"])
def test_2d_array_slice_read_write_log(slice_1, slice_2):
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

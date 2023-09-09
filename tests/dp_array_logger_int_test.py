import pytest

from dp import DPArray, Operation

def test_array_constructor_inits_logger():
    dp = DPArray(10)
    assert dp.logger.array_names == set(["dp_array"])

def test_array_constructor_takes_logger():
    dp = DPArray(10)
    dp2 = DPArray(10, "name", logger=dp.logger)
    assert dp.logger == dp2.logger

def test_array_constructor_duplicate_name_error():
    dp = DPArray(10, "duplicate_name")
    with pytest.raises(ValueError):
        DPArray(10, "duplicate_name", logger=dp.logger)

def test_array_read_write_log():
    dp = DPArray(10, "name")
    dp[0] = 1
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": set([0])}
    assert len(dp.logger.logs) == 1

    dp[0] = 2
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": set([0])}

    dp[1] = 3
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": set([0, 1])}

    temp = dp[1]
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": set([0, 1])}
    assert dp.logger.logs[1]["op"] == Operation.READ
    assert dp.logger.logs[1]["idx"] == {"name": set([1])}
    assert len(dp.logger.logs) == 2

    dp[2] = temp
    assert dp.logger.logs[2]["op"] == Operation.WRITE
    assert dp.logger.logs[2]["idx"] == {"name": set([2])}
    assert len(dp.logger.logs) == 3

def test_2d_array_read_write_log():
    dp = DPArray((10, 10), "name")
    dp[0, 0] = 1
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": set([(0, 0)])}
    assert len(dp.logger.logs) == 1

    temp = dp[0, 0]
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"name": set([(0, 0)])}
    assert dp.logger.logs[1]["op"] == Operation.READ
    assert dp.logger.logs[1]["idx"] == {"name": set([(0, 0)])}
    assert len(dp.logger.logs) == 2

    dp[3, 6] = temp
    assert dp.logger.logs[2]["op"] == Operation.WRITE
    assert dp.logger.logs[2]["idx"] == {"name": set([(3, 6)])}
    assert len(dp.logger.logs) == 3

def test_array_read_write_log_multiple_arrays():
    dp = DPArray(10)
    dp2 = DPArray(10, "dp_array2", logger=dp.logger)

    dp[0] = 1
    dp2[0] = 2
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"dp_array": set([0]), "dp_array2": set([0])}

    dp[1] = 3
    dp2[1] = dp[1]
    assert dp.logger.logs[0]["op"] == Operation.WRITE
    assert dp.logger.logs[0]["idx"] == {"dp_array": set([0, 1]), "dp_array2": set([0])}
    assert dp.logger.logs[1]["op"] == Operation.READ
    assert dp.logger.logs[1]["idx"] == {"dp_array": set([1])}
    assert dp.logger.logs[2]["op"] == Operation.WRITE
    assert dp.logger.logs[2]["idx"] == {"dp_array2": set([1])}
    assert len(dp.logger.logs) == 3

# def test_array_slice_read_write_log():
#     dp = DPArray(10)
#     dp[0:5] = 1
#     assert dp.logger.logs[0]["op"] == Operation.WRITE
#     assert dp.logger.logs[0]["idx"] == {"dp_array": set([0, 1, 2, 3, 4])}
#     assert len(dp.logger.logs) == 1

#     temp = dp[1:4]
#     assert dp.logger.logs[0]["op"] == Operation.WRITE
#     assert dp.logger.logs[0]["idx"] == {"dp_array": set([0, 1, 2, 3, 4])}
#     assert dp.logger.logs[1]["op"] == Operation.READ
#     assert dp.logger.logs[1]["idx"] == {"dp_array": set([1, 2, 3])}
#     assert len(dp.logger.logs) == 2

# def test_2d_array_slice_read_write_log():
#     dp = DPArray((10, 10))
#     dp[0:5, 0:5] = 1
#     assert dp.logger.logs[0]["op"] == Operation.WRITE
#     assert dp.logger.logs[0]["idx"] == {"dp_array": set([(i, j) for i in range(5) for j in range(5)])}
#     assert len(dp.logger.logs) == 1

#     dp[::3, ::3] = 2
#     assert dp.logger.logs[0]["op"] == Operation.WRITE
#     assert dp.logger.logs[0]["idx"] == {"dp_array": set([(i, j) for i in range(5) for j in range(3)])}

#     temp = dp[3::, 4:8]
#     assert dp.logger.logs[0]["op"] == Operation.WRITE
#     assert dp.logger.logs[0]["idx"] == {"dp_array": set([(i, j) for i in range(5) for j in range(5)])}
#     assert dp.logger.logs[1]["op"] == Operation.READ
#     assert dp.logger.logs[1]["idx"] == {"dp_array": set([(i, j) for i in range(3, 10) for j in range(4, 8)])}
#     assert len(dp.logger.logs) == 2

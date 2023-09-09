import pytest

from dp import DPArray, Logger, Operation

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

    dp[0] = dp[1]

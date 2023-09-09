import pytest

from dp import DPArray
from dp import Logger

def test_array_constructor_inits_logger():
    dp = DPArray(10, "name")
    assert dp.logger().array_count == 1
    assert dp.logger().array_names == ["name"]

def test_array_constructor_takes_logger():
    logger = Logger("test")
    dp = DPArray(10, "name", logger=logger)
    assert dp.logger() == logger

def test_array_constructor_duplicate_name_error():
    dp = DPArray(10, "name")
    with pytest.raises(ValueError):
        dp2 = DPArray(10, "name", logger=dp.logger())

def test_array_read_write_log():
    dp = DPArray(10, "name")
    dp[0] = 1
    assert dp.logger().logs[0].operation == Logger.Operation.WRITE
    assert dp.logger().logs[0].indices == {"name": set([0])}
    assert len(dp.logger().logs) == 1

    dp[0] = 2
    assert dp.logger().logs[0].operation == Logger.Operation.WRITE
    assert dp.logger().logs[0].indices == {"name": set([0])}

    dp[1] = 3
    assert dp.logger().logs[0].operation == Logger.Operation.WRITE
    assert dp.logger().logs[0].indices == {"name": set([0,1])}

    dp[0] = dp[1]
    
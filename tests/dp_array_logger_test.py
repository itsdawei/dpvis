"""Tests the interaction between DPArray and Logger."""
import numpy as np
import pytest

from dp import DPArray, Op


def test_read_write():
    dp = DPArray(10, "dp")

    dp[0] = 1
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0}}}
    assert len(dp.logger.logs) == 1

    dp[1] = 2
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0, 1}}}

    temp = dp[1]
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0, 1}}}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"dp": {1}}}
    assert len(dp.logger.logs) == 2

    dp[2] = temp
    assert dp.logger.logs[2] == {"op": Op.WRITE, "idx": {"dp": {2}}}
    assert len(dp.logger.logs) == 3


def test_2d_read_write():
    dp = DPArray((10, 10), "name")

    dp[0, 0] = 1
    assert len(dp.logger.logs) == 1

    temp = dp[0, 0]
    assert len(dp.logger.logs) == 2

    dp[3, 6] = temp
    assert len(dp.logger.logs) == 3
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"name": {(0, 0)}}}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"name": {(0, 0)}}}
    assert dp.logger.logs[2] == {"op": Op.WRITE, "idx": {"name": {(3, 6)}}}

def test_max_highlight():
    dp = DPArray(5, "name")
    dp[0] = 1
    dp[1] = 3
    dp[2] = 0
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"name": {0, 1, 2}}}

    dp.max(idx=3, refs=[0, 1, 2])
    assert dp.arr[3] == 3
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"name": {0, 1, 2}}}
    assert dp.logger.logs[2] == {"op": Op.HIGHLIGHT, "idx": {"name": {1}}}
    assert dp.logger.logs[3] == {"op": Op.WRITE, "idx": {"name": {3}}}

    dp.max(idx=4, refs=[0, 1, 2, 3], preprocessing=lambda x: -(x - 1)**2)
    assert dp.arr[4] == 0
    assert dp.logger.logs[4] == {"op": Op.READ, "idx": {"name": {0, 1, 2, 3}}}
    assert dp.logger.logs[5] == {"op": Op.HIGHLIGHT, "idx": {"name": {0}}}
    assert dp.logger.logs[6] == {"op": Op.WRITE, "idx": {"name": {4}}}


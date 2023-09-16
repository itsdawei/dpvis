"""Tests the interaction between DPArray and Logger."""
# import numpy as np
import pytest

from dp import DPArray, Op


def test_duplicate_array_error():
    dp1 = DPArray(10, "duplicate_name")
    with pytest.raises(ValueError):
        _ = DPArray(10, "duplicate_name", logger=dp1.logger)


def test_read_write():
    dp = DPArray(10, "dp")

    dp[0] = 1
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0: 1}}}
    assert len(dp.logger.logs) == 1

    dp[1] = 2
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0: 1, 1: 2}}}

    temp = dp[1]
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0: 1, 1: 2}}}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"dp": {1: None}}}
    assert len(dp.logger.logs) == 2

    dp[2] = temp
    assert dp.logger.logs[2] == {"op": Op.WRITE, "idx": {"dp": {2: 2}}}
    assert len(dp.logger.logs) == 3


def test_2d_read_write():
    dp = DPArray((10, 10), "name")

    dp[0, 0] = 1
    assert len(dp.logger.logs) == 1

    temp = dp[0, 0]
    assert len(dp.logger.logs) == 2

    dp[3, 6] = temp
    assert len(dp.logger.logs) == 3
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"name": {(0, 0): 1}}}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"name": {(0, 0): None}}}
    assert dp.logger.logs[2] == {"op": Op.WRITE, "idx": {"name": {(3, 6): 1}}}


def test_multiple_arrays_logging():
    dp1 = DPArray(10, "dp_1")
    dp2 = DPArray(10, "dp_2", logger=dp1.logger)

    dp1[0] = 1
    dp2[0] = 2
    assert dp1.logger.logs[0] == {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {
                0: 1
            },
            "dp_2": {
                0: 2
            }
        }
    }

    dp1[1] = 3
    dp2[1] = dp1[1]  # READ happens before WRITE
    assert dp1.logger.logs[0] == {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {
                0: 1,
                1: 3
            },
            "dp_2": {
                0: 2
            }
        }
    }
    assert dp1.logger.logs[1] == {
        "op": Op.READ,
        "idx": {
            "dp_1": {
                1: None
            },
            "dp_2": {}
        }
    }
    assert dp1.logger.logs[2] == {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {},
            "dp_2": {
                1: 3
            }
        }
    }
    assert len(dp1.logger.logs) == 3


@pytest.mark.parametrize(
    "op",
    [Op.WRITE, Op.READ,
     pytest.param(Op.HIGHLIGHT, marks=pytest.mark.xfail)],  # Expected to fail
    ids=["w", "r", "h"])
def test_same_op_and_index(op):
    """Same operation with same index does not create additional log."""
    dp = DPArray(10, "dp")

    if op == Op.WRITE:
        dp[0] = 1
        dp[0] = 2
    elif op == Op.READ:
        _ = dp[0]
        _ = dp[0]
    elif op == Op.HIGHLIGHT:
        # TODO: Perform highlight action
        pass
    assert dp.logger.logs[0] == {
        "op": op,
        "idx": {
            "dp": {
                0: 2 if op == Op.WRITE else None
            }
        }
    }
    assert len(dp.logger.logs) == 1


# @pytest.mark.parametrize("s",
#                          [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
#                          ids=["a", "b", "c", "d", "e"])
# def test_slice_logging(s):
#     dp = DPArray(10)

#     dp[s] = 1
#     if isinstance(s, int):
#         s = np.s_[s:s + 1]
#     truth = set(i for i in range(*s.indices(10)))
#     assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp_array": truth}}
#     assert len(dp.logger.logs) == 1

# @pytest.mark.parametrize("slice_1",
#                          [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
#                          ids=["a", "b", "c", "d", "e"])
# @pytest.mark.parametrize("slice_2",
#                          [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 1],
#                          ids=["a", "b", "c", "d", "e"])
# def test_2d_slice_logging(slice_1, slice_2):
#     dp = DPArray((10, 10))

#     dp[slice_1, slice_2] = 1
#     if isinstance(slice_1, int):
#         slice_1 = np.s_[slice_1:slice_1 + 1]
#     if isinstance(slice_2, int):
#         slice_2 = np.s_[slice_2:slice_2 + 1]
#     truth = {(i, j)
#              for i in range(*slice_1.indices(10))
#              for j in range(*slice_2.indices(10))}
#     assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp_array": truth}}
#     assert len(dp.logger.logs) == 1

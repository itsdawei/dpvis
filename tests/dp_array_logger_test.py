"""Tests the interaction between DPArray and Logger."""
import numpy as np
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


def test_get_timesteps_one_array():
    dp = DPArray(3, "dp")
    dp[0] = 1
    dp[1] = 2

    timesteps = dp.get_timesteps()
    assert len(timesteps) == 1
    assert np.all(timesteps[0]["dp"]["contents"] == [1, 2, None])
    assert timesteps[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()

    dp[1] = 3
    timesteps2 = dp.get_timesteps()
    assert len(timesteps2) == 1
    assert np.all(timesteps2[0]["dp"]["contents"] == [1, 3, None])
    assert timesteps2[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()

    _ = dp[1]
    timesteps3 = dp.get_timesteps()
    assert len(timesteps3) == 2
    assert np.all(timesteps3[1]["dp"]["contents"] == [1, 3, None])
    assert timesteps3[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

    dp[1] = 3
    timesteps4 = dp.get_timesteps()
    assert len(timesteps4) == 2
    assert np.all(timesteps4[1]["dp"]["contents"] == [1, 3, None])
    assert timesteps4[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: {1},
        Op.HIGHLIGHT: set(),
    }.items()

    dp[0] = dp[1]
    timesteps5 = dp.get_timesteps()
    assert len(timesteps5) == 3
    assert np.all(timesteps5[2]["dp"]["contents"] == [3, 3, None])
    assert timesteps5[2]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: {0},
        Op.HIGHLIGHT: set(),
    }.items()


def test_get_timesteps_two_arrays():
    dp = DPArray(3, "dp")
    dp2 = DPArray(3, "dp2", logger=dp.logger)

    dp[0] = 1
    dp[1] = 2
    dp2[0] = 2
    timesteps = dp.get_timesteps()
    assert len(timesteps) == 1
    assert np.all(timesteps[0]["dp"]["contents"] == [1, 2, None])
    assert np.all(timesteps[0]["dp2"]["contents"] == [2, None, None])
    assert timesteps[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()
    assert timesteps[0]["dp2"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0},
        Op.HIGHLIGHT: set(),
    }.items()

    _ = dp[1]
    timesteps1 = dp.get_timesteps()
    assert len(timesteps1) == 2
    assert np.all(timesteps1[1]["dp"]["contents"] == [1, 2, None])
    assert np.all(timesteps1[1]["dp2"]["contents"] == [2, None, None])
    assert timesteps1[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()
    assert timesteps1[1]["dp2"].items() >= {
        Op.READ: set(),
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

    dp2[2] = 3
    timesteps2 = dp.get_timesteps()
    assert len(timesteps2) == 2
    assert np.all(timesteps2[1]["dp"]["contents"] == [1, 2, None])
    assert np.all(timesteps2[1]["dp2"]["contents"] == [2, None, 3])
    assert timesteps2[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()
    assert timesteps2[1]["dp2"].items() >= {
        Op.READ: set(),
        Op.WRITE: {2},
        Op.HIGHLIGHT: set(),
    }.items()
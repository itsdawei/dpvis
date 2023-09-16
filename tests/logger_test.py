"""Tests the methods in Logger."""
import pytest
import numpy as np

from dp import Logger, Op

# pylint: disable=redefined-outer-name


@pytest.fixture
def logger():
    """Returns a logger with one array."""
    logger = Logger()
    logger.add_array("dp1", 10)
    return logger


def test_array_not_found_error(logger):
    with pytest.raises(ValueError):
        logger.append(Op.READ, "name_doesnt_exist", 0)


def test_add_array(logger):
    logger.add_array("dp2", 10)
    assert logger.array_names == {"dp1", "dp2"}


def test_add_during_logging_error(logger):
    logger.append("dp1", Op.READ, 0, 0)
    with pytest.raises(ValueError):
        logger.add_array("dp2", 10)


def test_append(logger):
    logger.add_array("dp2", 10)

    # Append to dp1.
    logger.append("dp1", Op.READ, 0)
    assert logger.logs[0] == {
        "op": Op.READ,
        "idx": {
            "dp1": {
                0: None
            },
            "dp2": {}
        }
    }
    logger.append("dp1", Op.READ, 1)
    assert logger.logs[0] == {
        "op": Op.READ,
        "idx": {
            "dp1": {
                0: None,
                1: None
            },
            "dp2": {},
        }
    }
    assert len(logger.logs) == 1

    # Append to dp2.
    logger.append("dp2", Op.READ, 0)
    assert logger.logs[0] == {
        "op": Op.READ,
        "idx": {
            "dp1": {
                0: None,
                1: None
            },
            "dp2": {
                0: None
            },
        }
    }
    assert len(logger.logs) == 1

    logger.append("dp1", Op.WRITE, 0, 1)
    # Previous time-step is not updated.
    assert logger.logs[0] == {
        "op": Op.READ,
        "idx": {
            "dp1": {
                0: None,
                1: None
            },
            "dp2": {
                0: None
            },
        }
    }
    # Current time-step is updated.
    assert logger.logs[1] == {
        "op": Op.WRITE,
        "idx": {
            "dp1": {
                0: 1
            },
            "dp2": {},
        }
    }
    # Total time-step.
    assert len(logger.logs) == 2


@pytest.mark.parametrize("op", [Op.WRITE, Op.READ, Op.HIGHLIGHT],
                         ids=["w", "r", "h"])
def test_same_ops_and_index(logger, op):
    if op == Op.WRITE:
        logger.append("dp1", Op.WRITE, 0, 1)
        logger.append("dp1", Op.WRITE, 0, 2)
    elif op == Op.READ:
        logger.append("dp1", Op.READ, 0)
        logger.append("dp1", Op.READ, 0)
    elif op == Op.HIGHLIGHT:
        logger.append("dp1", Op.HIGHLIGHT, 0)
        logger.append("dp1", Op.HIGHLIGHT, 0)
    assert len(logger.logs) == 1
    assert logger.logs[0] == {
        "op": op,
        "idx": {
            "dp1": {
                0: 2 if op == Op.WRITE else None
            },
        }
    }


def test_to_timesteps_one_array():
    logger = Logger()
    logger.add_array("dp1", 3)

    logger.append("dp1", Op.WRITE, 0, 1)
    timesteps = logger.to_timesteps()
    assert len(timesteps) == 1
    assert np.all(timesteps[0]["dp1"]["contents"] == [1, None, None])
    assert timesteps[0]["dp1"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0},
        Op.HIGHLIGHT: set(),
    }.items()

    logger.append("dp1", Op.WRITE, 2, 3)
    logger.append("dp1", Op.WRITE, 2, 4)
    timesteps1 = logger.to_timesteps()
    assert len(timesteps1) == 1
    assert np.all(timesteps1[0]["dp1"]["contents"] == [1, None, 4])
    assert timesteps1[0]["dp1"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 2},
        Op.HIGHLIGHT: set(),
    }.items()

    logger.append("dp1", Op.READ, 1)
    timesteps2 = logger.to_timesteps()
    assert len(timesteps2) == 2
    assert np.all(timesteps2[1]["dp1"]["contents"] == [1, None, 4])
    assert timesteps2[1]["dp1"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

    logger.append("dp1", Op.HIGHLIGHT, 0)
    logger.append("dp1", Op.WRITE, 0, 5)
    timesteps3 = logger.to_timesteps()
    assert len(timesteps3) == 2
    assert np.all(timesteps3[1]["dp1"]["contents"] == [5, None, 4])
    assert timesteps3[1]["dp1"].items() >= {
        Op.READ: {1},
        Op.WRITE: {0},
        Op.HIGHLIGHT: {0},
    }.items()

    logger.append("dp1", Op.READ, 1)
    logger.append("dp1", Op.HIGHLIGHT, 2)
    logger.append("dp1", Op.READ, 2)
    timesteps4 = logger.to_timesteps()
    assert len(timesteps4) == 3
    assert np.all(timesteps4[2]["dp1"]["contents"] == [5, None, 4])
    assert timesteps4[2]["dp1"].items() >= {
        Op.READ: {1, 2},
        Op.WRITE: set(),
        Op.HIGHLIGHT: {2},
    }.items()


def test_to_timesteps_two_arrays():
    logger = Logger()
    logger.add_array("dp1", 3)
    logger.add_array("dp2", 3)
    logger.append("dp1", Op.WRITE, 0, 1)
    logger.append("dp1", Op.WRITE, 2, 3)
    logger.append("dp2", Op.WRITE, 0, 2)
    logger.append("dp2", Op.WRITE, 1, 4)
    logger.append("dp1", Op.READ, 1)
    logger.append("dp2", Op.READ, 1)
    logger.append("dp1", Op.HIGHLIGHT, 0)

    timesteps = logger.to_timesteps()
    assert len(timesteps) == 2
    assert np.all(timesteps[0]["dp1"]["contents"] == [1, None, 3])
    assert timesteps[0]["dp1"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 2},
        Op.HIGHLIGHT: set(),
    }.items()
    assert np.all(timesteps[0]["dp2"]["contents"] == [2, 4, None])
    assert timesteps[0]["dp2"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()
    assert np.all(timesteps[1]["dp1"]["contents"] == [1, None, 3])
    assert timesteps[1]["dp1"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: {0},
    }.items()
    assert np.all(timesteps[1]["dp2"]["contents"] == [2, 4, None])
    assert timesteps[1]["dp2"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

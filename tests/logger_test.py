"""Tests the methods in Logger."""
import pytest

from dp import Logger, Op

# pylint: disable=redefined-outer-name


@pytest.fixture
def logger():
    """Returns a logger with one array."""
    logger = Logger()
    logger.add_array("dp1")
    return logger


def test_array_not_found_error(logger):
    with pytest.raises(ValueError):
        logger.append(Op.READ, "name_doesnt_exist", 0)


def test_add_array(logger):
    logger.add_array("dp2")
    assert logger.array_names == {"dp1", "dp2"}


def test_add_during_logging_error(logger):
    logger.append("dp1", Op.READ, 0, 0)
    with pytest.raises(ValueError):
        logger.add_array("dp2")


def test_append(logger):
    logger.add_array("dp2")

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

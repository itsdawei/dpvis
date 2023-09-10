import pytest

from dp import DPArray, Logger, Op

# pylint: disable=redefined-outer-name


@pytest.fixture
def logger():
    """Returns a logger with one array."""
    logger = Logger()
    logger.add_array("dp1")
    return logger


def test_duplicate_array_error():
    dp1 = DPArray(10, "duplicate_name")
    with pytest.raises(ValueError):
        _ = DPArray(10, "duplicate_name", logger=dp1.logger)


def test_array_not_found_error(logger):
    with pytest.raises(ValueError):
        logger.append(Op.READ, "name_doesnt_exist", 0)


def test_add_array(logger):
    logger.add_array("dp2")
    assert logger.array_names == {"dp1", "dp2"}


def test_add_during_logging_error(logger):
    logger.append("dp1", Op.READ, 0)
    with pytest.raises(ValueError):
        logger.add_array("dp2")


def test_append(logger):
    logger.add_array("dp2")

    # Append to dp1.
    logger.append("dp1", Op.READ, 0)
    assert logger.logs[0] == {"op": Op.READ, "idx": {"dp1": {0}, "dp2": set()}}
    logger.append("dp1", Op.READ, 1)
    assert logger.logs[0] == {
        "op": Op.READ,
        "idx": {
            "dp1": {0, 1},
            "dp2": set()
        }
    }
    assert len(logger.logs) == 1

    # Append to dp2.
    logger.append("dp2", Op.READ, 0)
    assert logger.logs[0] == {
        "op": Op.READ,
        "idx": {
            "dp1": {0, 1},
            "dp2": {0},
        }
    }
    assert len(logger.logs) == 1

    logger.append("dp1", Op.WRITE, 0)
    # Previous time-step is not updated.
    assert logger.logs[0] == {
        "op": Op.READ,
        "idx": {
            "dp1": {0, 1},
            "dp2": {0},
        }
    }
    # Current time-step is updated.
    assert logger.logs[1] == {
        "op": Op.WRITE,
        "idx": {
            "dp1": {0},
            "dp2": set(),
        }
    }
    # Total time-step.
    assert len(logger.logs) == 2


@pytest.mark.parametrize(
    "op",
    [Op.WRITE, Op.READ,
     pytest.param(Op.HIGHLIGHT, marks=pytest.mark.xfail)], # Expected to fail
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
    assert dp.logger.logs[0] == {"op": op, "idx": {"dp": {0}}}
    assert len(dp.logger.logs) == 1

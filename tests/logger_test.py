import pytest

from dp import Logger, Operation

@pytest.fixture
def logger():
    """Returns a logger with one array."""
    logger = Logger()
    logger.add_array("test")
    return logger


def test_add_array(logger):
    logger.add_array("test2")
    assert logger.array_names == set(["test", "test2"])


def test_append(logger):
    logger.append("test", Operation.READ, 0)
    assert logger.logs[0]["op"] == Operation.READ
    assert logger.logs[0]["idx"] == {"test": set([0])}

    logger.append("test", Operation.READ, 1)
    assert logger.logs[0]["op"] == Operation.READ
    assert logger.logs[0]["idx"] == {"test": set([0, 1])}
    assert len(logger.logs) == 1

    logger.add_array("test2")
    logger.append("test2", Operation.READ, 0)
    assert logger.logs[0]["op"] == Operation.READ
    assert logger.logs[0]["idx"] == {"test": set([0, 1]), "test2": set([0])}
    assert len(logger.logs) == 1

    logger.append("test", Operation.WRITE, 0)
    assert logger.logs[0]["op"] == Operation.READ
    assert logger.logs[0]["idx"] == {"test": set([0, 1]), "test2": set([0])}
    assert logger.logs[1]["op"] == Operation.WRITE
    assert logger.logs[1]["idx"] == {"test": set([0])}
    assert len(logger.logs) == 2


def test_append_bad_array_name(logger):
    with pytest.raises(ValueError):
        logger.append(Operation.READ, "name_doesnt_exist", 0)

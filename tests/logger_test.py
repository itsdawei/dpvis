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
    assert logger.array_count == 2
    assert logger.array_names == ["test", "test2"]


def test_append(logger):
    logger.append(Operation.READ, "test", 0)
    assert logger.logs[0].operation == Operation.READ
    assert logger.logs[0].indices == {"test": set([0])}

    logger.append(Operation.READ, "test", 1)
    assert logger.logs[0].operation == Operation.READ
    assert logger.logs[0].indices == {"test": set([0, 1])}
    assert len(logger.logs) == 1

    logger.append(Operation.READ, "test2", 0)
    assert logger.logs[0].operation == Operation.READ
    assert logger.logs[0].indices == {"test": set([0, 1]), "test2": set([0])}
    assert len(logger.logs) == 1

    logger.append(Operation.WRITE, "test", 0)
    assert logger.logs[0].operation == Operation.READ
    assert logger.logs[0].indices == {"test": set([0, 1]), "test2": set([0])}
    assert logger.logs[1].operation == Operation.WRITE
    assert logger.logs[1].indices == {"test": set([0])}
    assert len(logger.logs) == 2


def test_append_bad_array_name(logger):
    with pytest.raises(ValueError):
        logger.append(Operation.READ, "name_doesnt_exist", 0)


def test_eq(logger):
    logger2 = Logger()
    assert logger == logger2

    logger2.append(Operation.READ, "test", 0)
    assert not logger == logger2

    logger.append(Operation.READ, "test", 0)
    assert logger == logger2


def test_ne(logger):
    logger2 = Logger()
    assert not logger != logger2

    logger2.append(Operation.READ, "test", 0)
    assert logger != logger2

    logger.append(Operation.READ, "test", 0)
    assert not logger != logger2

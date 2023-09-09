import pytest

from dp import Logger


def test_constructor():
    logger = Logger("test")
    assert logger.array_count == 1
    assert logger.array_names == ["test"]


@pytest.fixture
def logger():
    """Returns a logger with one array."""
    return Logger("test")


def test_add_array(logger):
    logger.add_array("test2")
    assert logger.array_count == 2
    assert logger.array_names == ["test", "test2"]


def test_append(logger):
    logger.append(Logger.Operation.READ, "test", 0)
    assert logger.logs[0].operation == Logger.Operation.READ
    assert logger.logs[0].indices == {"test": set([0])}

    logger.append(Logger.Operation.READ, "test", 1)
    assert logger.logs[0].operation == Logger.Operation.READ
    assert logger.logs[0].indices == {"test": set([0, 1])}
    assert len(logger.logs) == 1

    logger.append(Logger.Operation.READ, "test2", 0)
    assert logger.logs[0].operation == Logger.Operation.READ
    assert logger.logs[0].indices == {"test": set([0, 1]), "test2": set([0])}
    assert len(logger.logs) == 1

    logger.append(Logger.Operation.WRITE, "test", 0)
    assert logger.logs[0].operation == Logger.Operation.READ
    assert logger.logs[0].indices == {"test": set([0, 1]), "test2": set([0])}
    assert logger.logs[1].operation == Logger.Operation.WRITE
    assert logger.logs[1].indices == {"test": set([0])}
    assert len(logger.logs) == 2


def test_append_bad_array_name(logger):
    with pytest.raises(ValueError):
        logger.append(Logger.Operation.READ, "name_doesnt_exist", 0)


def test_eq(logger):
    logger2 = Logger("test")
    assert logger == logger2

    logger2.append(Logger.Operation.READ, "test", 0)
    assert not logger == logger2

    logger.append(Logger.Operation.READ, "test", 0)
    assert logger == logger2


def test_ne(logger):
    logger2 = Logger("test")
    assert not logger != logger2

    logger2.append(Logger.Operation.READ, "test", 0)
    assert logger != logger2

    logger.append(Logger.Operation.READ, "test", 0)
    assert not logger != logger2


def test_log_constructor():
    log = Logger.Log(Logger.Operation.READ, "test", 0)
    assert log.operation == Logger.Operation.READ
    assert log.indices == {"test": set([0])}


def test_log_add_indice():
    log = Logger.Log(Logger.Operation.READ, "test", 0)
    log.add_indice("test", 1)
    assert log.indices == {"test": set([0, 1])}

    log.add_indice("test2", 0)
    assert log.indices == {"test": set([0, 1]), "test2": set([0])}


def test_log_is_same_operation():
    log = Logger.Log(Logger.Operation.READ, "test", 0)
    assert log.is_same_operation(Logger.Operation.READ)
    assert not log.is_same_operation(Logger.Operation.WRITE)

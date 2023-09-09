"""This file provides the Logger class."""
import enum as Enum


class Logger:
    """Logger class.

    Args:
        array_name (str): The name of the array to be logged.

    Attributes:
        _logs (list): Contains the logs.
        _array_count (int): The number of arrays logged.
        _array_names (list): The names of the arrays logged.

    """

    def __init__(self, array_name):
        self._array_count = 1
        self._array_names = [array_name]
        self._logs = []

    def add_array(self, array_name):
        """Adds an array to the logger.

        Args:
            array_name (str): The name of the array to be added.
        Returns:
            None
        """
        self.array_count += 1
        self.array_names.append(array_name)

    def append(self, array_name, operation, indice):
        """Appends a log to the logger.

        Args:
            operation (Operation): The operation to be logged.
            indice (int): The indice to be logged.
        Returns:
            None
        Raises:
            ValueError: Array name not found in logger / array not tracked by logger. 
        """
        if array_name not in self.array_names:
            raise ValueError("Array name not found in logger.")
        if len(self._logs) == 0:
            self._logs.append(self.Log(operation, array_name, indice))
        else:
            if self._logs[-1].is_same_operation(operation):
                self._logs[-1].add_indice(array_name, indice)
            else:
                self._logs.append(self.Log(operation, array_name, indice))

    @property
    def logs(self):
        """Returns the logs."""
        return self._logs

    @property
    def array_names(self):
        """Returns the array names."""
        return self._array_names

    @property
    def array_count(self):
        """Returns the array count."""
        return self._array_count

    class Operation(Enum):
        """Enum for the operation of the log."""
        READ = 1
        WRITE = 2
        HIGHLIGHT = 3

    class Log:
        """Wraps the operation and indices of the log.

        Args:
            operation (Operation): "READ", "WRITE", or "HIGHLIGHT"
            array_name (str): The name of the array to be logged.
            indice (int): The indice to be logged.

        Attributes:
            operation (Operation): The operation of the log.
            indices (dict): The indices of the log. Key: array_name, Value: list of indices.

        """

        def __init__(self, operation, array_name, indice):
            self.operation = operation
            self.indices = dict()
            self.indices[array_name] = [indice]

        def is_same_operation(self, operation):
            """Returns True if the operation is the same as the log's operation."""
            return self.operation == operation

        def add_indice(self, array_name, indice):
            """Adds an indice to the log."""
            if array_name not in self.indices:
                self.indices[array_name] = [indice]
            else:
                self.indices[array_name].append(indice)

"""This file provides the Logger class."""
import enum as Enum
import numpy as np


class Logger:
    """Logger class.

    """

    def __init__(
        self,
        array_name
    ):
        self._logs = []
        self.array_count = 1
        self.array_names = [array_name]

    def add_array(
        self,
        array_name
    ):
        """Adds an array to the logger.

        Args:
            array_name (str): The name of the array to be added.
        Returns:
            None
        """
        self.array_count += 1
        self.array_names.append(array_name)

    def append(
        self,
        array_name,
        operation,
        indice
    ):
        """Appends a log to the logger.

        Args:
            operation (Operation): The operation to be logged.
            indice (int): The indice to be logged.
        Returns:
            None
        """
        if self._logs.is_empty():
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
        READ = 1
        WRITE = 2
        HIGHLIGHT = 3

    class Log:
        """Wraps the operation and indices of the log.
        Operation: READ | WRITE | HIGHLIGHT
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

"""This file provides the Logger class."""
from enum import IntEnum


class Operation(IntEnum):
    """The allowed operation on the DPArray."""
    READ = 1
    WRITE = 2
    HIGHLIGHT = 3


class Logger:
    """Logger class.

    Attributes:
        _logs (list): Contains the logs.
        _array_names (list): The names of the arrays logged.

    """

    def __init__(self):
        self._array_names = set()
        self._logs = []

    def add_array(self, array_name):
        """Adds an array to the logger.

        Args:
            array_name (str): The name of the array to be added.
        """
        if array_name in self.array_names:
            raise ValueError(f"Array name {array_name} already exists in"
                             f"logger.")
        self.array_names.add(array_name)

    def append(self, array_name, operation, idx):
        """Appends an operation to the log.

        Args:
            operation (Operation): Operation performed.
            idx (int): Index of the array.

        Raises:
            ValueError: Array name not recognized by logger. 
        """
        if array_name not in self.array_names:
            raise ValueError(f"Array name {array_name} not recognized by"
                             f"logger. Make sure logger is passed to the"
                             f"constructor of {array_name}")
        if len(self._logs) == 0 or self._logs[-1]["op"] != operation:
            self._logs.append({"op": operation, "idx": {array_name: idx}})
        else:
            self._logs[-1]["idx"][array_name].append(idx)

    @property
    def logs(self):
        """Returns the logs."""
        return self._logs

    @property
    def array_names(self):
        """Returns the array names."""
        return self._array_names

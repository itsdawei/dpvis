"""This file provides the Logger class."""
from enum import IntEnum

import numpy as np


class Op(IntEnum):
    """The allowed operation on the DPArray."""
    READ = 1
    WRITE = 2
    HIGHLIGHT = 3


class Logger:
    """Logger class.

    # TODO: add explanation of the format of the log

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
        if len(self._logs) > 0:
            raise ValueError(
                f"Cannot add array {array_name} to a non-empty logger.")
        self.array_names.add(array_name)

    def append(self, array_name, operation, idx, values=None):
        """Appends an operation to the log.

        Args:
            operation (Operation): Operation performed.
            idx (list of tuple/int): Index of the array.
            values (list): Values updated, 

        Raises:
            ValueError: Array name not recognized by logger. 
        """
        if array_name not in self.array_names:
            raise ValueError(f"Array name {array_name} not recognized by"
                             f"logger. Make sure logger is passed to the"
                             f"constructor of {array_name}")

        idx_list = idx
        if isinstance(idx, np.ndarray):
            idx_list = idx.tolist()
        elif isinstance(idx, int):
            idx_list = [idx]

        if values:
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif isinstance(values, (int, float, np.float32, np.float64)):
                values = [values]
            if len(idx_list) != len(values):
                raise ValueError(f"Length of idx {idx_list} and values {values}"
                                 f"do not match.")
        elif values is None and operation == Op.WRITE:
            raise ValueError(f"Values must be provided for {operation}.")

        # Initialize new operation.
        if len(self._logs) == 0 or self._logs[-1]["op"] != operation:
            self._logs.append({
                "op": operation,
                "idx": {
                    # array_name: {idx1: value1, idx2: value2, ...}
                    name: dict() for name in self._array_names
                },
            })
        self._logs[-1]["idx"][array_name] |= dict(zip(idx_list, values) if values else zip(idx_list, [None] * len(idx_list)))

    @property
    def logs(self):
        """Returns the logs."""
        return self._logs

    @property
    def array_names(self):
        """Returns the array names."""
        return self._array_names

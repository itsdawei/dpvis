"""This file provides the Logger class."""
from enum import IntEnum

import numpy as np
from colorama import Fore, Style


class Op(IntEnum):
    """The allowed operation on the DPArray."""
    READ = 1
    WRITE = 2
    HIGHLIGHT = 3
    ANNOTATE = 4


class Logger:
    """Logger class.

    The log format is as follows:
    {
        "op": Op.READ/Op.WRITE/Op.HIGHLIGHT/Op.ANNOTATE,
        "idx": {
            "array1": {idx1: value1, idx2: value2, ...},
            "array2": {idx1: value1, idx2: value2, ...},
            ...
        },
    }
    note that values are None for READ and HIGHLIGHT.

    Attributes:
        _logs (list): Contains the logs.
        _array_shapes (dict): The shapes of the arrays associated with 
            a logger instance. Key: Array name, Value: Array shape.
    """

    def __init__(self):
        """Initializes an empty logger."""
        self._array_shapes = {}
        self._logs = []

    def add_array(self, array_name, shape):
        """Adds an array to the logger.

        Args:
            array_name (str): The name of the array to be added.
            shape (int or tuple): The shape of the array to be added.
        """
        if array_name in self._array_shapes:
            raise ValueError(f"Array name {array_name} already exists in"
                             f"logger.")
        if len(self._logs) > 0:
            raise ValueError(
                f"Cannot add array {array_name} to a non-empty logger.")
        self._array_shapes[array_name] = shape

    def append(self, array_name, operation, idx, values=None):
        """Appends an operation to the log.

        Args:
            operation (Operation): Operation performed.
            idx (list of tuple/int): Index of the array.
            values (list): Values updated, 

        Raises:
            ValueError: Array name not recognized by logger. 
        """
        if array_name not in self._array_shapes:
            raise ValueError(f"Array name {array_name} not recognized by"
                             f"logger. Make sure logger is passed to the"
                             f"constructor of {array_name}")

        idx_list = idx
        if isinstance(idx, np.ndarray):
            idx_list = idx.tolist()
        elif isinstance(idx, int):
            idx_list = [idx]

        if values is not None:
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif isinstance(values,
                            (int, float, np.float32, np.float64, np.int64)):
                values = [values]
            if len(idx_list) != len(values):
                raise ValueError(f"Length of idx {idx_list} and values {values}"
                                 f" do not match.")
        elif values is None and operation == Op.WRITE:
            raise ValueError(f"Values must be provided for {operation}.")

        # Initialize new operation.
        if len(self._logs) == 0 or self._logs[-1]["op"] != operation:
            self._logs.append({
                "op": operation,
                "idx": {
                    # array_name: {idx1: value1, idx2: value2, ...}
                    name: {} for name in self._array_shapes
                },
            })

        values = [None] * len(idx_list) if values is None else values
        self._logs[-1]["idx"][array_name].update(dict(zip(idx_list, values)))

    def append_annotation(self, array_name, annotation, idx=None):
        """Appends an annotated operation to the log.

        Args:
            array_name (str): Name of the array associated with this operation.
            operation (Operation): Type of operation performed.
            annotation (str): Annotations associated with this operation.
            idx (int): Index of the array.

        Raises:
            ValueError: Array name not recognized by logger. 
            AttributeError: Appending annotations to an empty logger.
        """
        if array_name not in self._array_shapes:
            raise ValueError(f"Array name {array_name} not recognized by"
                             f"logger. Make sure logger is passed to the"
                             f"constructor of {array_name}")
        if len(self._logs) == 0:
            raise AttributeError(
                "Cannot append annotations to an empty logger.")

        # Add ANNOTATE log.
        if self._logs[-1]["op"] != Op.ANNOTATE:
            self._logs.append({"op": Op.ANNOTATE, "idx": {array_name: {}}})

        # Array-wise annotation uses the key "ARRAY".
        if idx is None:
            idx = "ARRAY"

        # NOTE: Currently overwrites the annotation on the same cell when
        # called multiple times.
        self._logs[-1]["idx"][array_name][idx] = str(annotation)

    def to_timesteps(self):
        """Converts the logs to timesteps.

        Raises:
            ValueError: If the logs are not in the correct format.

        Returns:
            list of timestep dicts
            timestep: {
                "array_name": {
                    "annotations": array annotations at this timestep which are
                        not associated with any cell but the entire array.
                    "cell_annotations": array cell annotations at this timestep,
                    "contents": array contents at this timestep,
                    Op.READ: [idx1, idx2, ...],
                    Op.WRITE: [idx1, idx2, ...],
                    Op.HIGHLIGHT: [idx1, idx2, ...],
                },
                "array_2": {
                    ...
                },
            }
        """
        array_contents = {
            name: np.full(shape, None)
            for name, shape in self._array_shapes.items()
        }

        # For each consecutive sequence of Op.WRITE, find the last index.
        last_write_indices = []
        for i, log in enumerate(self._logs):
            if log["op"] != Op.WRITE:
                continue
            if last_write_indices and i == last_write_indices[-1] + 1:
                continue
            last_write_indices.append(i)

        # NOTE: If annotation is added after the last write, it will be group
        # in a timestep without a write. This fixes that issue by forcing the
        # last timestep to include everything after the last write as well.
        # However, this may cause issues with backtracing.
        last_write_indices[-1] = len(self.logs)

        # Split the logs into batches based on the last write indices.
        log_batches = np.split(self._logs, np.array(last_write_indices) + 1)

        contents = {
            name: np.full(shape, None)
            for name, shape in self._array_shapes.items()
        }

        # Create a new timestep for each batch.
        timesteps = []
        for batch in log_batches:
            timesteps.append({
                name: {
                    "contents": array_contents[name].copy(),
                    Op.READ: {},
                    Op.WRITE: {},
                    Op.HIGHLIGHT: {},
                    Op.ANNOTATE: {},
                } for name in self._array_shapes
            })
            for log in batch:
                op = log["op"]
                for name, indices in log["idx"].items():
                    timesteps[-1][name][op] |= indices
                    # For WRITE, track the changes to the DP array.
                    if op == Op.WRITE:
                        for idx, val in indices.items():
                            contents[name][idx] = val
                        timesteps[-1][name]["contents"] = contents[name].copy()
        return timesteps

    def print_timesteps(self):
        """Prints the timesteps in color. Currently works for 1D arrays only.

        Raises:
            ValueError: If the array shapes are not 1D.
        """
        # if array_shapes are not 1D, raise error
        for name, shape in self.array_shapes.items():
            if not isinstance(shape, int):
                raise ValueError("must be 1D array to print timesteps.")

        timesteps = self.to_timesteps()

        for i, ts in enumerate(timesteps):
            print(i)
            for name, shape in self.array_shapes.items():  # assume 1d
                # print name, then contents of the array,
                # if an item's index is in ts[name][Op.WRITE] then highlight it
                print("\t", name, ": ", end="")
                print("\t[", end="")
                for i in range(shape):
                    if i in ts[name][Op.WRITE]:
                        print(Fore.RED,
                              f"{ts[name]['contents'][i]:>2.0f}",
                              end="")
                    elif i in ts[name][Op.HIGHLIGHT]:
                        print(Fore.GREEN,
                              f"{ts[name]['contents'][i]:>2.0f}",
                              end="")
                    elif i in ts[name][Op.READ]:
                        print(Fore.YELLOW,
                              f"{ts[name]['contents'][i]:>2.0f}",
                              end="")
                    elif ts[name]["contents"][i] is None:
                        print("   ", end="")
                    else:
                        print(f"{ts[name]['contents'][i]:>3.0f}", end="")
                    print(Style.RESET_ALL, end="")
                    print(",", end="")
                print("]")

    @property
    def logs(self):
        """Returns the logs."""
        return self._logs

    @property
    def array_shapes(self):
        """Returns the array shapes."""
        return self._array_shapes

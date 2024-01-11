"""This file provides the Logger class."""
from enum import IntEnum

import numpy as np
from colorama import Fore, Style


class Op(IntEnum):
    """Allowed operations on the array to be recorded by Logger.

    Attributes:
        READ (int): Reading from indices in the array. No values required.
        WRITE (int): Writing to indices in the array. Requires an value for each
            index.
        MAXMIN (int): Maximum or minimum on indices in the array. No values
            required.
    """
    READ = 1
    WRITE = 2
    MAXMIN = 3


class Logger:
    """Records the operations performed on the DPArray.

    [`Visualizer`][dp._visualizer.Visualizer] uses the
    [`Logger`][dp._logger.Logger] to produce a frame-by-frame animation of the
    states of the DPArray.

    Each Logger is associated with some [`DPArray`][dp._dp_array] objects. The
    logger stores a list of "logs," where each log contains information
    regarding an [`operations`][dp._logger.Op] performed on the `DPArray`.
    For example, for a DP problem with two arrays (say, `OPT` and `Values`),
    consider the following operations on the corresponding arrays:

    1. **`OPT`**: `READ` operation at index `(1, 1)`, `(1, 2)`, and `(1, 3)`.
    1. **`Values`**: `READ` operation at index `(0, 0)`.
    1. **`OPT`**: `WRITE` operation of value `10` at index `(3, 3)` and
       `(4, 4)`.
    1. **`Values`**: `MAXMIN` operation at index `(3, 3)` and `(4, 4)`.

    ```json
    [{
        "op": Op.READ,
        "idx": {
            "OPT": {
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
            },
            "Values": { (0, 0): None },
        },
    },
    {
        "op": Op.WRITE,
        "idx": {
            "OPT": {
                (3, 3): 10,
                (4, 4): 10,
            },
            "Values": {},
        },
    },
    {
        "op": Op.MAXMIN,
        "idx": {
            "OPT": {
                (3, 3): None,
                (4, 4): None,
            },
            "Values": {},
        },
    }]
    ```

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
            raise ValueError(f"Cannot add array {array_name} to a non-empty"
                             f"logger.")
        self._array_shapes[array_name] = shape

    def append(self, array_name, operation, idx, values=None):
        """Appends an operation to the log.

        Args:
            operation (Op): Operation performed.
            idx (list of indices): Indices of the array. For 1D arrays, this
                is a list of int. For higher dimensional arrays, this is a list
                of tuples.
            values (list): Values of the array that is updated with for
                [Op.WRITE][dp._logger.Op].

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

        if values is None:
            values = [None] * len(idx_list)
        self._logs[-1]["idx"][array_name].update(dict(zip(idx_list, values)))

    def append_annotation(self, array_name, annotation, idx=None):
        """Appends an annotated operation to the log.

        ```json
        {
            "annotations": {
                name_1: "annotation1",
                name_2: "annotation1",
            }
            "cell_annotations": {
                name_1: {
                    idx1: "annotation1", 
                    idx2: "annotation2",
                    ...,
                },
                ...
            }
        }
        ```

        Args:
            array_name (str): Name of the array associated with this operation.
            annotation (str): Annotations associated with this operation.
            idx (int or tuple): Index of the element to be annotated on. None 
                if the annotation associated with the entire array.

        Raises:
            ValueError: Array name not recognized by logger. 
        """
        if array_name not in self._array_shapes:
            raise ValueError(f"Array name {array_name} not recognized by"
                             f"logger. Make sure logger is passed to the"
                             f"constructor of {array_name}")

        # Create or overwrite annotation.
        if idx is None:
            if "annotations" not in self._logs[-1]:
                self._logs[-1]["annotations"] = {}
            self._logs[-1]["annotations"][array_name] = annotation
        else:
            if "cell_annotations" not in self._logs[-1]:
                self._logs[-1]["cell_annotations"] = {
                    name: {} for name in self._array_shapes
                }
            self._logs[-1]["cell_annotations"][array_name][idx] = annotation

    def to_timesteps(self):
        """Converts the logs to timesteps.

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
                    Op.MAXMIN: [idx1, idx2, ...],
                },
                "array_2": {
                    ...
                },
            }

        Raises:
            ValueError: If the logs are not in the correct format.
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

        # Split the logs into batches based on the last write indices.
        log_batches = np.split(self._logs, np.array(last_write_indices) + 1)
        # Remove the last batch if it is empty.
        # TODO: investigate the source of this behavior and potentially optimize
        #   how batches are created to fix this.
        if log_batches[-1].size == 0:
            log_batches = log_batches[:-1]

        contents = {
            name: np.full(shape, None)
            for name, shape in self._array_shapes.items()
        }

        # Create a new timestep for each batch.
        timesteps = []
        for batch in log_batches:
            timesteps.append({
                name: {
                    "annotations": "",
                    "cell_annotations": {},
                    "contents": array_contents[name].copy(),
                    Op.READ: set(),
                    Op.WRITE: set(),
                    Op.MAXMIN: set(),
                } for name in self._array_shapes
            })
            for log in batch:
                op = log["op"]
                for name, indice_dict in log["idx"].items():
                    timesteps[-1][name][op] |= indice_dict.keys()
                    # For WRITE, track the changes to the DP array.
                    if op == Op.WRITE:
                        for idx, val in indice_dict.items():
                            contents[name][idx] = val
                            array_contents[name][idx] = val
                        timesteps[-1][name]["contents"] = contents[name].copy()

                for annotate_key in ["annotations", "cell_annotations"]:
                    if annotate_key not in log:
                        continue
                    for name, annotation in log[annotate_key].items():
                        timesteps[-1][name][annotate_key] = annotation
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
                    elif i in ts[name][Op.MAXMIN]:
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

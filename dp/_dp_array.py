"""This file provides the DPArray class."""
import warnings

import numpy as np

from dp._index_converter import _nd_slice_to_indices
from dp._logger import Logger, Op


class DPArray:
    """DPArray class.

    Args:
        shape (array-like): The dimensions of the array.
        array_name (str): Name of the array, this is used by ``dp.Logger`` when
            the DP algorithm interacts with multiple arrays. The array name is
            displayed as the figure title when the array is visualized.
        logger (dp.Logger): Logger object that tracks the actions performed on
            this array, including READ, WRITE, and HIGHLIGHT. This object is
            used to reproduce frame-by-frame animation of the DP algorithm.
        dtype (str or data-type): Data type of the DPArray. We only support
            ``"f"`` / ``np.float32`` and ``"d"`` / ``np.float64``.

    Attributes:
        _arr (np.array): Contains the values of the DP array.
        _occupied_arr (np.array): A mask that indicates which index is filled.
    """

    def __init__(
        self,
        shape,
        array_name="dp_array",
        *,
        logger=None,
        dtype=np.float64,
    ):
        """Initializes the DPArray."""
        self._dtype = self._parse_dtype(dtype)

        self._arr = np.full(shape, dtype=self._dtype, fill_value=np.nan)
        self._occupied_arr = np.zeros_like(self._arr, dtype=bool)

        self._logger = Logger() if logger is None else logger
        self._logger.add_array(array_name, shape)

        self._array_name = array_name

    @staticmethod
    def _parse_dtype(dtype):
        """Parses the dtype passed into the constructor.

        Returns:
            np.float32 and np.float64
        Raises:
            ValueError: Unsupported dtype.
        """
        # First convert str dtype's to np.dtype.
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        # np.dtype is not np.float32 or np.float64, but it compares equal.
        if dtype == np.float32:
            return np.float32
        if dtype == np.float64:
            return np.float64
        if dtype == int:
            return int

        raise ValueError("Unsupported dtype. Must be np.float32 or"
                         "np.float64")

    def annotate(self, annotation, idx=None):
        """Annotates the array or a cell of the array. 
        This annotation will be associated with the last regular operation.

        Args:
            annotation (str): The annotation to be added.
            idx (int or tuple of ints): The index of the array. If None, the
                annotation will be associated with the entire array.
        """
        self._logger.append_annotation(self._array_name, annotation, idx)

    def get_timesteps(self):
        """Retrieve the timesteps of all DPArrays associated with this array's 
            logger.

        Returns:
            list of timesteps where each timestep is:
            timestep: {
                "array_name": {
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
        return self._logger.to_timesteps()

    def print_timesteps(self):
        """Prints the timesteps in color. Currently works for 1D arrays only.

        Raises:
            ValueError: If the array shapes are not 1D.
        """
        self._logger.print_timesteps()

    def __getitem__(self, idx):
        """Retrieve an item using [] operator.

        Args:
            idx (int): The index of the array.

        Returns:  
            self.dtype or np.ndarray: corresponding item

        Warnings:
            Raises an warning when an undefined index is referenced.
        """
        if not np.all(self._occupied_arr[idx]):
            read_indices = np.full(self._arr.shape, False)
            read_indices[idx] = True
            undef_read_indices = np.flatnonzero(
                np.asarray(~self.occupied_arr & read_indices))
            warnings.warn(
                f"Referencing undefined elements in "
                f"'{self._array_name}'. Undefined elements: "
                f"{undef_read_indices}.",
                category=RuntimeWarning)
        log_idx = _nd_slice_to_indices(self._arr, idx)
        self._logger.append(self._array_name, Op.READ, log_idx)
        return self._arr[idx]

    def __setitem__(self, idx, value):
        """Set an item using the assignment operator.

        Args:
            idx (int): The index of the array.
            value (self.dtype or array-like): The assigned value.

        Raises:
            ValueError: If ``idx`` is a slice object.
        """
        log_idx = _nd_slice_to_indices(self._arr, idx)

        value = self.dtype(value)
        if isinstance(value, self.dtype):
            value = np.full(len(log_idx), value)

        self._arr[idx] = value.reshape(self._arr[idx].shape)
        self._occupied_arr[idx] = True
        self._logger.append(self._array_name, Op.WRITE, log_idx, value)

    def __eq__(self, other):
        """Equal to operator.

        Args:
            other (DPArray or array-like): Other container.

        Returns:
            np.ndarray: True/False mask.
        """
        if isinstance(other, DPArray):
            return self.arr == other.arr
        return self.arr == other

    def __ne__(self, other):
        """Not equal to operator.

        Args:
            other (DPArray or array-like): Other container.

        Returns:
            np.ndarray: True/False mask.
        """
        if isinstance(other, DPArray):
            return self.arr != other.arr
        return self.arr != other

    def _cmp(self, cmp, indices, elements=None):
        """Helper function for comparing a list of elements.

        Iterates through a list of element and outputs the "largest" element
        according to the cmp function. The indices corresponding to the final
        output will be highlighted in the DP array. To use this function,
        provide a list of elements and a list of indices for each element.
        For example,
        ```
        cmp = lambda x, y: x > y
        elements = [0, 1, 2, 3, 4, 4]
        indices = [None, 2, 4, 6, 8, 1]
        ```
        The output of this function will be `4` and indices `8` and `1` will be
        highlighted.

        Args:
            cmp (callable): A callable that returns boolean. cmp(x, y) == True
                if x is larger than y. For example, x > y for maximum and
                x < y for minimum.
            indices (array-like): An array of indices of the elements. This
                array has the same shape as elements.
            elements (array-like): An array of elements to be compared.
                These can be elements directly from the array (i.e. arr[0]), or
                modified elements (i.e. arr[0] + 1). If elements is None, the
                value of array at the indices queried is used.

        Returns:
            dtype: Final result of the comparisons

        Raises:
            ValueError: Indices and elements must have same length.
            ValueError: Indices and elements cannot be empty.
        """
        # Elements is an optional argument
        if elements is None:
            elements = [self[idx] for idx in indices]
        # TODO shape check for slices.
        if len(indices) != len(elements):
            raise ValueError("indices and elements must have same length")
        if len(elements) == 0 or len(indices) == 0:
            raise ValueError("indices and elements cannot be empty")

        best_indices = [indices[0]]
        best_element = elements[0]
        for i, e in zip(indices, elements):
            # Unravel when index is a slice.
            if isinstance(i, slice) and isinstance(e, np.ndarray):
                # Get argmax indices.
                slice_max_idx = e.flatten().argmax()
                slice_indices = _nd_slice_to_indices(self._arr, i)
                i = slice_indices[slice_max_idx]
                # Get max element.
                e = np.max(e)
            else:
                # Make index into a singleton.
                i = [i]

            # If new best index/element is found.
            if cmp(e, best_element):
                best_indices = i
                best_element = e

            # If index has equivalent element to the best element.
            elif e == best_element:
                best_indices.extend(i)

        # Highlight and write value.
        self.logger.append(self._array_name, Op.MAXMIN, best_indices)
        return best_element

    def max(self, indices, elements=None):
        """Outputs the maximum value and highlight its corresponding index.

        Args:
            elements (array-like): An array of elements to be compared.
                These can be elements directly from the array (i.e. arr[0]), or
                modified elements (i.e. arr[0] + 1).
            indices (array-like): An array of indices of the elements.
                indices[i] correspond to elements[i]. If elements[i] is not an
                element of the DP array, item[i] should be None. If elements 
                is None, the value of array at the indices queried is used.

        Returns:
            self.dtype: Maximum value of the elements.
        """
        return self._cmp(lambda x, y: x > y, indices, elements)

    def min(self, indices, elements=None):
        """Outputs the minimum value and highlight its corresponding index.

        Args:
            indices (array-like): An array of indices of the elements.
                indices[i] correspond to elements[i]. If elements[i] is not an
                element of the DP array, item[i] should be None.
            elements (array-like): An array of elements to be compared.
                These can be elements directly from the array (i.e. arr[0]), or
                modified elements (i.e. arr[0] + 1). If elements is None, the
                value of array at the indices queried is used.

        Returns:
            self.dtype: Minimum value of the elements.
        """
        return self._cmp(lambda x, y: x < y, indices, elements)

    def add_traceback_path(self, path):
        """Add a traceback path to this DPArray object.

        Paths added to a DPArray object will be displayed when calling display()
        on that DPArray object. The path will appear on the last frame of the
        visualization window (slider is in the rightmost position).

        Args:
            path (list of tuples): A list of indices to be displayed.
            Indices should be tuples with as many elements as the dimension
            of the array (the number of dimensions is given by len(shape)).
        """
        log_idx = _nd_slice_to_indices(self._arr, path)
        self._logger.append(self._array_name, Op.READ, log_idx)

    @property
    def arr(self):
        """Returns the np.ndarray that contains the DP array."""
        return np.array(self._arr, copy=True)

    @property
    def shape(self):
        """Returns the shape of the DP array."""
        return self._arr.shape

    @property
    def occupied_arr(self):
        """Returns the np.ndarray that contains the occupied mask."""
        return np.array(self._occupied_arr, copy=True)

    @property
    def logger(self):
        """Returns the np.ndarray that contains all the computations."""
        return self._logger

    @property
    def dtype(self):
        """Returns the data type of the array."""
        return self._dtype

    @property
    def array_name(self):
        """Returns the array name."""
        return self._array_name

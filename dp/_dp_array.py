"""This file provides the DPArray class."""
import warnings

import numpy as np

from dp._logger import Logger, Op


class DPArray:
    """DPArray class.

    Args:
        shape (array-like): The dimensions of the array.
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
        description_string=None,
        row_labels=None,
        column_labels=None,
        colors=None,
        dtype=np.float64,
    ):
        """Initializes the DPArray."""
        self._dtype = self._parse_dtype(dtype)

        self._arr = np.full(shape, dtype=self._dtype, fill_value=np.nan)
        self._occupied_arr = np.zeros_like(self._arr, dtype=bool)

        self._logger = Logger() if logger is None else logger
        self._logger.add_array(array_name, shape)

        self._array_name = array_name
        self._description_string = description_string
        self._row_labels = row_labels
        self._column_labels = column_labels
        self._colors = colors

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

        raise ValueError("Unsupported dtype. Must be np.float32 and np.float64")

    def _nd_slice_to_indices(self, nd_slice):
        """Converts a nd-slice to indices.

        Calculate the indices from the slices.
        1. Get the start, stop, and step values for each slice.
        2. Use np.arange to create arrays of indices for each slice.
        3. Create 1D array of indices using meshgrid and column_stack.

        Args:
            nd_slice (slice/int, list/tuple of slice/int):
                nd_slice can be anything used to index a numpy array. For
                example:
                - Direct indexing: (0, 0, ...)
                - 1d slice: slice(0, 10, 2)
                - nd slice: (slice(0, 10, 2), slice(1, 5, 1), ...)
                - Mixture: (slice(0, 10, 2), 5, 1)

        Returns:
            list of tuples/integer: length n list of d-tuples, where n is the 
                number of indices and d is the dimension DPArray. If d = 1, 
                then the list will contain integers instead.

        Raises:
            ValueError: If ``nd_slice`` is not a slice object, a list of slice
                objects, or a list of tuples of integers.
            ValueError: If any element in ``nd_slice`` is not a valid slice
                object or integer.
        """
        if isinstance(nd_slice, (slice, int)):
            # Convert 1d slice to nd slice.
            nd_slice = (nd_slice,)
        if not isinstance(nd_slice, (list, tuple)):
            raise ValueError(f"'nd_slice' has type {type(nd_slice)}, must be "
                             f"a slice object, a list/tuple of slice objects,"
                             f"or a list/tuple of integers.")

        slice_indices = []
        for dim, size in enumerate(self._arr.shape):
            s = nd_slice[dim]
            if isinstance(s, slice):
                # Handle slice objects.
                slice_indices.append(np.arange(*s.indices(size)))
            elif isinstance(s, int):
                # Handle tuple of integers for direct indexing.
                slice_indices.append(s)
            else:
                raise ValueError("Each element in 'nd_slice' must be a valid"
                                 "slice object or integer.")

        # Generate the meshgrid of indices and combine indices into
        # n-dimensional index tuples.
        mesh_indices = np.meshgrid(*slice_indices, indexing="ij")
        indices = np.stack(mesh_indices,
                           axis=-1).reshape(-1, len(slice_indices))

        # Convert to tuple if index is > 1D, otherwise remove the last
        # dimension.
        indices_tuples = ([tuple(row) for row in indices] if indices.shape[1]
                          != 1 else np.squeeze(indices, axis=1))
        return indices_tuples

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

        Warning:
            Warns if an undefined index is referenced.
        """
        if not np.all(self._occupied_arr[idx]):
            read_indices = np.full(self._arr.shape, False)
            read_indices[idx] = True
            undef_read_indices = np.flatnonzero(
                np.asarray(~self.occupied_arr & read_indices is True))
            warnings.warn(
                f'Referencing undefined elements in "{self._array_name}". \
                          Undefined elements: {undef_read_indices}.',
                category=RuntimeWarning)

        log_idx = self._nd_slice_to_indices(idx)
        self._logger.append(self._array_name, Op.READ, log_idx)
        return self._arr[idx]

    def __setitem__(self, idx, value):
        """Set an item using the assignment operator.

        Args:
            idx (int): The index of the array.
            value (self.dtype): The assigned value.

        Raises:
            ValueError: If ``idx`` is a slice object.
        """
        log_idx = self._nd_slice_to_indices(idx)
        if isinstance(idx, slice):
            raise ValueError("Slice assignment not currently supported.")
        # TODO: potentially support slice writes
        # TODO: match values to log_idx?
        converted_val = self.dtype(value)
        self._logger.append(self._array_name, Op.WRITE, log_idx, converted_val)
        self._arr[idx] = self.dtype(converted_val)
        self._occupied_arr[idx] = True

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

    def _cmp(self, cmp, indices, elements):
        """Helper function for comparing a list of elements.

        Iterates through a list of element and outputs the "largest" element
        according to the cmp function. The index corresponding to the final
        output will be highlighted in the DP array. To use this function,
        provide a list of elements and a list of indices for each element.
        For example,
        ```
        cmp = lambda x, y: x > y
        elements = [0, 1, 2, 3, 4]
        indices = [None, 2, 4, 6, 8]
        ```
        The output of this function will be `4` and index `8` will be
        highlighted.

        Args:
            cmp (callable): A callable that returns boolean. cmp(x, y) == True
                if x is larger than y. For example, x > y for maximum and
                x < y for minimum.
            indices (array-like): An array of indices of the elements. This
                array has the same shape as elements.
            elements (array-like): An array of elements to be compared.
                These can be elements directly from the array (i.e. arr[0]), or
                modified elements (i.e. arr[0] + 1).

        Returns:
            dtype: Final result of the comparisons

        Raises:
            ValueError: TODO
        """
        # TODO shape check for slices.
        if len(indices) != len(elements):
            raise ValueError("indices and elements must have same length")
        if len(elements) == 0 or len(indices) == 0:
            raise ValueError("indices and elements cannot be empty")

        best_index = indices[0]
        best_element = elements[0]
        for i, e in zip(indices, elements):
            # Unravel when index is a slice.
            if isinstance(i, slice) and isinstance(e, np.ndarray):
                slice_indices = self._nd_slice_to_indices(i)
                slice_max_idx = e.flatten().argmax()
                e = e[slice_max_idx]
                i = slice_indices[slice_max_idx]

            if cmp(e, best_element):
                best_index = i
                best_element = e

        # Highlight and write value.
        if best_index is not None:
            self.logger.append(self._array_name, Op.HIGHLIGHT, best_index)
        return best_element

    def max(self, indices, elements):
        """Outputs the maximum value and highlight its corresponding index.

        Args:
            elements (array-like): An array of elements to be compared.
                These can be elements directly from the array (i.e. arr[0]), or
                modified elements (i.e. arr[0] + 1).
            indices (array-like): An array of indices of the elements.
                indices[i] correspond to elements[i]. If elements[i] is not an
                element of the DP array, item[i] should be None.

        Returns:
            self.dtype: Maximum value of the elements.
        """
        return self._cmp(lambda x, y: x > y, indices, elements)

    def min(self, indices, elements):
        """Outputs the minimum value and highlight its corresponding index.

        Args:
            indices (array-like): An array of indices of the elements.
                indices[i] correspond to elements[i]. If elements[i] is not an
                element of the DP array, item[i] should be None.
            elements (array-like): An array of elements to be compared.
                These can be elements directly from the array (i.e. arr[0]), or
                modified elements (i.e. arr[0] + 1).

        Returns:
            self.dtype: Minimum value of the elements.
        """
        return self._cmp(lambda x, y: x < y, indices, elements)

    @property
    def arr(self):
        """Returns the np.ndarray that contains the DP array."""
        return np.array(self._arr, copy=True)

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

"""This file provides the DPArray class."""
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
        self._dtype = self._parse_dtype(dtype)

        self._arr = np.empty(shape, dtype=self._dtype)
        self._occupied_arr = np.zeros_like(self._arr, dtype=bool)

        self._logger = Logger() if logger is None else logger
        self._logger.add_array(array_name)

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
            list of tuples/integer: length n list of d-tuples, where n is the number of
                indices and d is the dimension DPArray. If d = 1, then the list will
                contain integers instead.

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
            raise ValueError(f"'nd_slice' has type {type(nd_slice)}, must be"
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
        indices = np.stack(mesh_indices, axis=-1).reshape(-1, len(slice_indices))

        # Convert to tuple if index is > 1D, otherwise remove the last
        # dimension.
        indices_tuples = ([tuple(row) for row in indices] if indices.shape[1]
                          != 1 else np.squeeze(indices, axis=1))
        return indices_tuples

    def __getitem__(self, idx):
        """Retrieve an item using [] operator.

        Args:
            idx (int): The index of the array.

        Returns:
            self.dtype or np.ndarray:
        """
        # TODO: Check if idx is occupied
        log_idx = self._nd_slice_to_indices(idx)
        self._logger.append(self._array_name, Op.READ, log_idx)
        return self._arr[idx]

    def __setitem__(self, idx, value):
        """Set an item using the assignment operator.

        Args:
            idx (int): The index of the array.
            value (self.dtype): The assigned value.
        """
        log_idx = self._nd_slice_to_indices(idx)
        self._logger.append(self._array_name, Op.WRITE, log_idx)
        self._arr[idx] = self.dtype(value)

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

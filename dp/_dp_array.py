"""This file provides the DPArray class."""
import numpy as np


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

        if logger is None:
            # TODO: Create logger
            # self._logger = Logger()
            pass
        else:
            self._logger = logger

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

    def __getitem__(self, idx):
        """Retrieve an item using [] operator.

        Args:
            idx (int): The index of the array.

        Returns:
            self.dtype or np.ndarray:
        """
        # TODO: Check if idx is occupied
        # TODO: Record READ in logger
        return self._arr[idx]

    def __setitem__(self, idx, value):
        """Set an item using the assignment operator.

        Args:
            idx (int): The index of the array.
            value (self.dtype): The assigned value.
        """
        # TODO: Record WRITE in logger
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

    # @property
    # def logger(self):
    #     """Returns the np.ndarray that contains all the computations."""
    #     return self._logger

    @property
    def dtype(self):
        """Returns the data type of the array."""
        return self._dtype

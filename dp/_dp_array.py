"""This file provides the DPArray class."""
import numpy as np


class DPArray:
    """DPArray class.

    Args:
        a (tuple or numpy.ndarray): If given tuple, this is interpreted as the
            dimensions of the array. If given array, it is used as the intial
            array of the DP.
        dtype (str or data-type): Data type of the DPArray. We only support
            ``"f"`` / ``np.float32`` and ``"d"`` / ``np.float64``.

    Attributes:
        _solution_dim (int): See ``solution_dim`` arg.
    """

    def __init__(
        self,
        a,
        logger=None,
        description_string=None,
        row_labels=None,
        column_labels=None,
        colors=None,
        dtype=np.float64,
    ):
        self._dtype = self._parse_dtype(dtype)
        if isinstance(a, tuple):
            self._arr = np.array(a, dtype=self._dtype)
        elif isinstance(a, np.ndarray):
            self._arr = a.astype(self._dtype)
        else:
            raise ValueError("'a' must be a tuple or np.ndarray")

        if logger is None:
            # TODO: Create logger
            # self._logger = Logger()
            pass
        else:
            self._logger=logger

        self._description_string=description_string
        self._row_labels=row_labels
        self._column_labels=column_labels
        self._colors=colors

    @staticmethod
    def _parse_dtype(dtype):
        """Parses the dtype passed into the constructor.

        Returns:
            np.float32 or np.float64
        Raises:
            ValueError: There is an error in the bounds configuration.
        """
        # First convert str dtype's to np.dtype.
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        # np.dtype is not np.float32 or np.float64, but it compares equal.
        if dtype == np.float32:
            return np.float32
        if dtype == np.float64:
            return np.float64

        raise ValueError("Unsupported dtype. Must be np.float32 or np.float64")

    def __getitem__(self, idx):
        """Retrieve an item using [] operator.

        Args:
            idx (int): The index of the array.

        Returns:
            self.dtype or np.ndarray:
        """
        return self._arr[idx]

    def __setitem__(self, idx, value):
        """Set an item using the assignment operator.

        Args:
            idx (int): The index of the array.
            value (self.dtype): The assigned value.
        """
        self._arr[idx] = self.dtype(value)

    @property
    def arr(self):
        """Returns the np.ndarray that contains all the computations."""
        return self._arr

    # @property
    # def logger(self):
    #     """Returns the np.ndarray that contains all the computations."""
    #     return self._logger

    @property
    def dtype(self):
        """Returns the data type of the array."""
        return self._dtype

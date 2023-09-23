"""This file provides the DPArray class."""
from typing import Iterable
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

        self._arr = np.empty(shape, dtype=self._dtype)
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

<<<<<<< HEAD
    def print_timesteps(self):
        """Prints the timesteps in color. Currently works for 1D arrays only.
        
        Raises:
            ValueError: If the array shapes are not 1D.
        """
        self._logger.print_timesteps()

=======
>>>>>>> 02bd9137e4e96e9fc67230c1a565cda7bc6ed87f
    def __getitem__(self, idx):
        """Retrieve an item using [] operator.

        Args:
            idx (int): The index of the array.

        Returns:  
            self.dtype or np.ndarray: corresponding item
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
        # TODO: match values to log_idx?
        self._logger.append(self._array_name, Op.WRITE, log_idx, value)
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

    def _max_min(self, cmp, refs, preprocessing, const):
        """
        Args:
            cmp (callable): Use x > y for max and x < y for min
            idx: The index to assign the calculated value to
            refs (iterable of indices): Indices to retreive values
                from to use in the max/min function. Must be an
                iterable even if the iterable is a singleton
            preprocessing (callable or iterable of callables): 
                If callable preprocessing will be applied to each
                ref value before applying the max/min function. 
                If iterable of callables, then it is requried the
                len(refs) = len(preprocessing). preprocessing[i]
                will be applied to refs[i] before applying the max function.
            const (float): A constant value to use in the min/max operation
                
        Returns:
            The max/min value of the references after applying preprocessing  
        """
        # Error handling
        if not isinstance(refs, Iterable) or len(refs) == 0:
            raise ValueError(
                "Expecting reference to be Iterable of length " + \
                "at least one."
            )
        if not callable(preprocessing) and len(preprocessing) != len(refs):
            raise ValueError(
                "Expected refs and preprocessing of same length or single " + \
                "preprocessing callable."
            )

        # Make iterable to iterate over
        if callable(preprocessing):
            itr = [(ref, preprocessing) for ref in refs]
        else:
            itr = zip(refs, preprocessing)

        # Find max/min value and corresponding idx
        best_idx = None
        best_val = const
        for ref, func in itr:
            val = func(self[ref])

            if isinstance(val, np.ndarray):
                slice_indices = self._nd_slice_to_indices(ref)
                val = val.flatten()
                slice_max_idx = val.argmax()
                val = val[slice_max_idx]
                ref = slice_indices[slice_max_idx]

            if best_val is None or cmp(val, best_val):
                best_idx = ref
                best_val = val

        # Highlight and write value
        if best_idx is not None:
            self.logger.append(self._array_name, Op.HIGHLIGHT, best_idx)
        return best_val

    def max(self, refs, preprocessing=(lambda x: x), const=None):
        """
        Args:
            idx: The index to assign the calculated value to
            refs (iterable of indices): Indicies to retreive
                values from to use in the max function
            preprocessing (callable or iterable of callables): 
                If callable preprocessing will be applied to
                each refs value before applying the max function. 
                If iterable of callables, then it is requried the
                len(refs) = len(preprocessing). preprocessing[i]
                will be applied to refs[i] before applying the max
                function.
            const (float): A constant value to use in the min/max
                operation
                
        Returns:
            The maximum value after applying preprocessing to refs             
        """
        return self._max_min(cmp=lambda x, y: x > y,
                             refs=refs,
                             preprocessing=preprocessing,
                             const=const)

    def min(self, refs, preprocessing=(lambda x: x), const=None):
        """
        Args:
            idx: The index to assign the calculated value to
            refs (iterable of indices): Indicies to retreive
                values from to use in the min function
            preprocessing (callable or iterable of callables): 
                If callable preprocessing will be applied to
                each refs value before applying the min function. 
                If iterable of callables, then it is requried the
                len(refs) = len(preprocessing). preprocessing[i]
                will be applied to refs[i] before applying the min
                function.
            const (float): A constant value to use in the min/max
                operation
                
        Returns:
            The minimum value after applying preprocessing to refs       
        """
        return self._max_min(cmp=lambda x, y: x < y,
                             refs=refs,
                             preprocessing=preprocessing,
                             const=const)

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

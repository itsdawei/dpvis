import numpy as np

def _indices_to_np_indices(indices):
    """Get an Iterable of tuples representing indices and convert it into numpy
    indicies.

    Example input: [(0, 1), (2, 3), (4, 5)]
    Example output: ([0, 2, 4], [1, 3, 5])

    Args:
        indices(Iterable): Set of indices. It is expected that the indices are
        integers for 1D arrays and tuples of integers for arrays of greater than two dimensions
        (the number of intergers in the tuples should be equal to the dimension of the array).

    Returns:
        formatted_indices: outputs the given indices in numpy form:
        a list of values on the first dimension and a list of values on
        the second dimension.
    """
    if not isinstance(indices, list):
        indices = list(indices)

    np.ndarray(indices)


    # # ignore if 1-d or no indicies
    # if len(indices) <= 0 or isinstance(list(indices)[0], int):
    #     return list(indices)

    # x, y = [], []
    # for i in indices:
    #     x.append(i[0])
    #     y.append(i[1])
    # return x, y

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
            - Direct indexing: (0, 1, ...) or [0, 1, ...]
            - 1d slice: slice(0, 10, 2)
            - nd slice: (slice(0, 10, 2), slice(1, 5, 1), ...)
            - Mixture: (slice(0, 10, 2), 5, 1)

    Returns:
        list of tuples/integer: length n list of d-tuples, where n is the 
            number of indices and d is the dimension DPArray. If d = 1, 
            then the list will contain integers instead.

    Raises:
        ValueError: ``nd_slice`` is not a slice object, a list of slice
            objects, or a list of tuples of integers.
        ValueError: Some element in ``nd_slice`` is not a valid slice
            object or integer.
    """
    if isinstance(nd_slice, (slice, int)):
        # Convert 1d slice to nd slice.
        nd_slice = (nd_slice,)
    if isinstance(nd_slice, list):
        return nd_slice
    if not isinstance(nd_slice, (list, tuple)):
        raise ValueError(f"'nd_slice' has type {type(nd_slice)}, must be "
                            f"a slice object, a list/tuple of slice objects,"
                            f" or a list/tuple of integers.")

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
            raise ValueError("Each element in 'nd_slice' must be a valid "
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
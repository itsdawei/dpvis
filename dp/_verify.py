"""This file contains a method to verify traceback solutions"""
import numpy as np
from dp._logger import Op
from dp._index_converter import _indices_to_np_indices


def verify_traceback_path(arr, path):
    """Verify that solution is a valid traceback solution of the DPArray.

    Args:
        dp (DPArray): A DPArray that should be initialzed according 
            to an optimization problem. A traceback solution is not
            well defined for non-optimization dynamic programming
            (i.e. does not use max or min).
        path (list of indices): A list of indices.
            For 1D DPArrays, this should be a list of integers.
            For 2D DPArrays, this should be a list of tuples.
        
    Return:
        bool: False if the given path is not correct and True
            if it is correct. Empty paths will raise a Value Error.
            Incomplete paths (in which an unexplored predecessor
            still exists) are incorrect. Paths that contain the
            index of an unitialized element are incorrect.
    
    Raises:
        ValueError: Path is empty (length 0).
    """
    # Handle trivial case.
    if len(path) == 0:
        raise ValueError("Path must be non-empty.")

    # Ensure each index in the path is initialized.
    if not np.all(arr.occupied_arr[_indices_to_np_indices(path)]):
        return False

    # Go through time steps and check predecessors.
    i = len(path) - 1
    name = arr.array_name
    for timestep in reversed(arr.get_timesteps()):
        # If writing path[i], analyze predecessors.
        if path[i] in timestep[name][Op.WRITE]:
            # Check that path[0] has no predecessors.
            if i == 0:
                return len(timestep[name][Op.MAXMIN]) == 0

            # path[i - 1] is not a predecessor.
            # The given path is not correct.
            if path[i - 1] not in timestep[name][Op.MAXMIN]:
                return False
            i = i - 1
    return True

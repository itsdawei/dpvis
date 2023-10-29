"""This file contains a method to verify traceback solutions"""
import numpy as np
from dp._logger import Op
from dp._index_converter import _indices_to_np_indices


@staticmethod
def verify_traceback_solution(arr, solution):
    """
    Check if solution is a valid traceback of DPArray dp.

    Args:
        dp (DPArray): A DPArray that should be initialzed according to an
        optimization problem. A traceback solution is not well defined
        for non-optimization dynamic programming (i.e. does not use
        max or min).

        solution (list of indices): S list of indices.
        For 1D DPArrays, this should be a list of integers.
        For 2D DPArrays, this should be a list of tuples.
        
    Return:
        bool: False if the given solution is not correct
        and True if it is correct. Empty solutions are considered correct.
        Incomplete solutions (in which an unexplored predecessor still
        exists) are incorrect. Solutions that contain the index of an
        unitialized element are incorrect.
    """
    # Handle trivial case.
    if len(solution) == 0:
        return True

    # Ensure each index in the solution is initialized.
    if not np.all(arr.occupied_arr[*_indices_to_np_indices(solution)]):
        return False

    # Go through time steps and check predecessors.
    i = len(solution) - 1
    for timestep in reversed(arr.get_timesteps()):
        # If writing solution[i], analyze predecessors.
        if solution[i] in timestep[arr.array_name][Op.WRITE]:
            # Check that solution[0] has no predecessors.
            if i == 0:
                return len(timestep[arr.array_name][Op.HIGHLIGHT]) == 0

            # solution[i - 1] is not a predecessor.
            # The given solution is not correct.
            if solution[i - 1] not in \
                    timestep[arr.array_name][Op.HIGHLIGHT]:
                return False
            i = i - 1
    return True

import numpy as np
from dp._logger import Op
from dp._index_converter import _indices_to_np_indices

@staticmethod
def is_traceback(dp, solution):
    """
    Check if solution is a valid traceback of DPArray dp.

    Args:
        dp (DPArray): a DPArray that should be initialzed according to an optimization problem.
        Traceback is not well defined for non-optimization dynamic programming (i.e. does not use max min)

        solution (list of indices): a list of indices.
        For 1D DPArrays, this should be a list of integers.
        For 2D DPArrays, this should be a list of tuples.
        
    Return:
        Returns False if the given solution is not correct
        and true if it is correct. Empty solutions are considered correct.
        Incomplete solutions (in which an unexplored predecessor still exists) are incorrect.
        Solutions that contain the index of an unitialized element are incorrect.
    """
    # Handle trivial case
    if(len(solution) == 0):
        return True
    
    # Ensure each index in the solution is initialized
    if not np.all(dp._occupied_arr[_indices_to_np_indices(solution)]):
        return False

    # Go through time steps and check predecessors
    i = len(solution) - 1
    for timestep in reversed(dp.get_timesteps()):
        # If writing solution[i], analyze predecessors
        if solution[i] in timestep[dp.array_name][Op.WRITE]:
            # Check that solution[0] has no predecessors
            if i == 0:
                return len(timestep[dp.array_name][Op.HIGHLIGHT]) == 0
            
            # Check that solution[i - 1] is a predecessor of solution[i]
            else:
                # solution[i - 1] is not a predecessor, so the given solution is not correct
                if solution[i - 1] not in timestep[dp.array_name][Op.HIGHLIGHT]:
                    return False
                i = i - 1
    return True
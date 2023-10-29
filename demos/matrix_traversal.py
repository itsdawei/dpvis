import numpy as np

from dp import DPArray, display
from dp._verify import verify_traceback_solution


def matrix_traversal(M):
    """Dynamic program that solves the Matrix Traversal problem.

    Given a matrix M of shape (n + 1, m + 1), where M[x, y] is the cost of
    traveling to (x, y). An agent begins at index (0, 0), and find a least-cost
    path to (n, m). The agent is only able to move South or Eest from its
    current position. 
    """
    OPT = DPArray(shape=M.shape)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # Base case.
            if i == 0 and j == 0:
                OPT[0, 0] = M[0, 0]
                continue

            # Assume that the agent have arrived at cell (i, j). The agent
            # must have arrived here from cell (i-1, j) or (i, j-1), since the
            # agent can only travel from South or East.

            indices = []
            elements = []
            if i > 0:
                indices.append((i - 1, j))
                elements.append(OPT[i - 1, j])
            if j > 0:
                indices.append((i, j - 1))
                elements.append(OPT[i, j - 1])

            # We take the better path between the optimal path to (i-1, j) and
            # (i, j-1).
            OPT[i, j] = M[i, j] + OPT.min(indices=indices, elements=elements)

    OPT.enable_logger(False)
    current = (M.shape[0] - 1, M.shape[1] - 1)
    solution = [current]
    while current != (0, 0):
        if (current[1] < 1 or OPT[current[0] - 1, current[1]]
                <= OPT[current[0], current[1] - 1]):
            current = (current[0] - 1, current[1])
        else:
            current = (current[0], current[1] - 1)
        solution.append(current)
    solution = solution[::-1]

    OPT.add_backtrack_solution(solution)
    
    # TODO:
    # import verify_solution_set from verifaction
    # verify_solution_set(OPT, index_of_final_solution, proposed_path)
    # Print info on command line.

    row_labels = [str(i) for i in range(M.shape[0])]
    column_labels = [str(j) for j in range(M.shape[1])]
    display(OPT, row_labels=row_labels, column_labels=column_labels)

    return OPT[M.shape[0] - 1, M.shape[1] - 1]


if __name__ == "__main__":
    M = np.array([
        [5, 4, 2, 5, 4, 3, 4],
        [1, 4, 3, 4, 2, 0, 1],
        [3, 5, 0, 2, 4, 3, 4],
        [2, 4, 4, 4, 1, 3, 0],
        [4, 2, 5, 0, 5, 0, 5],
    ])
    matrix_traversal(M)

import pytest
import numpy as np
from dp import DPArray
from dp._verify import is_traceback

@pytest.mark.parametrize(
        "matrix, solutions, not_solutions",
        [(np.array([[0, 2, 1],
                    [1, 0, 1],
                    [1, 2, 1],]),
                    ([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
                     [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
                     [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
                     ),
                    ([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
                     [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
                     [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
                     ),
                    )],
        ids=["simple"]
)           
def test_is_backtrack(matrix, solutions, not_solutions):
    (h, w) = matrix.shape
    dp = DPArray((h, w))

    dp[0, 0] = matrix[0, 0]
    for i in range(h):
        for j in range(w):
            indices = []
            elements = []
            if i == 0 and j == 0:
                continue

            if i > 0:
                indices.append((i - 1, j))
                elements.append(dp[i - 1, j])

            if j > 0:
                indices.append((i, j - 1))
                elements.append(dp[i, j - 1])

            dp[i, j] = dp.max(indices, elements) + matrix[i, j]
    
    for solution in solutions:
        assert is_traceback(dp, solution)

    for not_solution in not_solutions:
        assert not is_traceback(dp, not_solution)
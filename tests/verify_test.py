"""Tests for verifying backtracking paths."""
import pytest
import numpy as np
from dp import DPArray
from dp._verify import verify_traceback_path


@pytest.mark.parametrize("matrix, paths, not_paths", [
    (
        np.array([
            [0, 2, 1],
            [1, 0, 1],
            [1, 2, 1],
        ]),
        (
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        ),
        (
            [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
            [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
        ),
    ),
    (
        np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 0], [1, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1]]),
        ([(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5),
          (3, 5), (4, 5)], [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 1),
                            (4, 2), (4, 3), (4, 4), (4, 5)]),
        ([(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5),
          (3, 5), (4, 5)], [(0, 0), (1, 1), (1, 2)]),
    )
],
                         ids=["a", "b"])
def test_is_backtrack(matrix, paths, not_paths):
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

    for path in paths:
        assert verify_traceback_path(dp, path)

    for not_path in not_paths:
        assert not verify_traceback_path(dp, not_path)

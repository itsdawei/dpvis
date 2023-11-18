import numpy as np
import pytest

from dp._index_converter import _indices_to_np_indices, _nd_slice_to_indices


@pytest.mark.parametrize("indices, np_indices",
                         [([], []), ([1, 2, 3], [1, 2, 3]),
                          ([(1, 2), (3, 4), (5, 6),
                            (7, 8)], ((1, 3, 5, 7), (2, 4, 6, 8))),
                          ([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)],
                           ((1, 4, 7, 10), (2, 5, 8, 11), (3, 6, 9, 12)))],
                         ids=["Empty", "1D", "2D", "3D"])
def test_indices_to_np_indices(indices, np_indices):
    assert np_indices == _indices_to_np_indices(indices)


@pytest.mark.parametrize("slice_1",
                         [np.s_[::2], np.s_[4:6], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
@pytest.mark.parametrize("slice_2",
                         [np.s_[::2], np.s_[4:6], np.s_[4:], np.s_[:6], 1],
                         ids=["a", "b", "c", "d", "e"])
def test_nd_slice_to_indices(slice_1, slice_2):
    nd_slice = (slice_1, slice_2)
    arr = np.zeros((100, 2))
    arr[nd_slice] = 1

    indices = _nd_slice_to_indices(arr, nd_slice)
    for i in range(100):
        for j in range(2):
            ans = (1 if (i, j) in indices else 0)
            assert arr[(i, j)] == ans

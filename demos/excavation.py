import numpy as np
from dp._visualizer import Visualizer

from dp import DPArray


def excavate(v, M):
    """Dynamic program that solves the Excavate problem.

    There are N mining sites, where each site has L layers. Excavating level
    l of site i returns a value of v[i][l]. It takes one month to excavate one
    level from one mining site. The layers of the site must be excavated in
    order, i.e., you must excavate levels 0,...,l-1 before excavating level l.
    Find the maximum value that can be obtained from excavating if you have M
    months to mine.

    Args:
        v (array-like): v[i][l] represent the value gained from excavating level
            l of site i.
        M (int): Number of months allowed.

    Returns:
        int: Maximum value that can be attained from v given M months.
    """

    # We have enough months to excavate everything, so DP is not needed.
    if v.shape[0] * v.shape[1] < M:
        return np.sum(v)

    # OPT[i, m] is the maximum value that can be obtained by spending m months
    # on mining sites up to the ith site.
    OPT = DPArray((v.shape[0] + 1, M + 1), array_name="Excavation")

    V = DPArray(shape=v.shape, array_name="V", logger=OPT.logger)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            V[i, j] = v[i, j]

    # Base case:
    OPT[0, :] = 0  # Given no mining sites.
    OPT[:, 0] = 0  # Given zero months.

    for i in range(1, v.shape[0] + 1):  # For the ith mining site.
        for m in range(1, M + 1):  # Consider a budget of m <= M months.
            indices = []
            elements = []
            # Consider the maximum value obtained if we had decided to dig
            # l <= m layers on the ith site.
            for l in range(m + 1):
                indices.append((i - 1, m - l))

                # Assume that we have dug l layers from the ith site, and
                # obtained value of np.sum(v[i-1, :l]). Since we have dug
                # l layers from the ith site, we only have m-l months left for
                # the rest of the sites. We have already computed the maximum
                # value given m-l months and up to the (i-1)st site as
                # OPT[i - 1, m-l]. Hence, we can simply add these values
                # together as follows:
                elements.append(OPT[i - 1, m - l] + np.sum(V[i - 1, :l]))
            # After considering all possible choices of digging l layers from
            # the ith site, we can take choice that yields the maximum value.
            OPT[i, m] = OPT.max(indices=indices, elements=elements)

            OPT.annotate(f"OPT[{i}, {m}] = {OPT[i, m]}")
            V.annotate(f"V[{i - 1}, :{m}] = {V[i - 1, :m]}")

    # TODO: Implement backtracking.
    # backtrack(OPT, indices)

    row_labels = [f"{i}th Mine" for i in range(v.shape[0] + 1)]
    if len(row_labels) > 0:
        row_labels[0] = "0th Mine"
    if len(row_labels) > 1:
        row_labels[1] = "1st Mine"
    if len(row_labels) > 2:
        row_labels[2] = "2nd Mine"
    if len(row_labels) > 3:
        row_labels[3] = "3rd Mine"
    column_labels = [f"{i} Months" for i in range(M + 1)]
    visualizer = Visualizer()
    visualizer.add_array(OPT,
                         column_labels=column_labels,
                         row_labels=row_labels)
    visualizer.add_array(V)
    visualizer.show()

    return OPT[v.shape[0], M]


if __name__ == "__main__":
    v = np.array([
        [5, 4, 2, 5, 4, 3, 4],
        [1, 4, 3, 4, 2, 0, 1],
        [3, 5, 0, 2, 4, 3, 4],
        [2, 4, 4, 4, 1, 3, 0],
        [4, 2, 5, 0, 5, 0, 5],
    ])
    M = 10
    excavate(v, M)

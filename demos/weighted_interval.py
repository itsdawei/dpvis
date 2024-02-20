import numpy as np

from dp import DPArray
from dp._visualizer import display, Visualizer


def solve(intervals):
    """Dynamic program for Weighted Interval Selection.

    Args:
        intervlas (array-like): a list of intervals. Each interval is
        represented as a list of [start time, finish time, weight].

    Returns:
        int: Maximum value that can be attained from v given M months.
    """
    # Sort intervals by finish time.
    intervals = sorted(intervals, key=lambda x: x[1])

    # Maximum of intervals selected
    N = len(intervals)

    OPT = DPArray(N + 1, array_name="Weighted Interval Scheduling", dtype=int)

    # Compute p[i] = largest index j < i s.t. interval i is compatible with j.
    # p = [0] * (N + 1)  # [-1, ..., -1]
    p = DPArray(N+1, logger=OPT.logger, array_name="P", dtype=int)
    p[:] = 0
    for i, a in enumerate(intervals, start=1):
        # Search up to the ith interval.
        for j, b in enumerate(intervals[:i], start=1):
            # Check that a and b are compatible.
            if min(a[1], b[1]) - max(a[0], b[0]) < 1:
                p[i] = j

    # Base Case.
    OPT[0] = 0

    for i, a in enumerate(intervals, start=1):
        OPT[i] = max(a[2] + OPT[p[i]], OPT[i - 1])

    column_labels = [f"{i} intervals" for i in range(N + 1)]
    description = ("| Interval | Start | Finish | Weight | p |\n"
                   "| :------: | :---: | :----: | :----: | - |\n")
    for i, a in enumerate(intervals, start=1):
        description += f"| {i} | {a[0]} | {a[1]} | {a[2]} | {p._arr[i]} |\n"
    visualizer = Visualizer()
    visualizer.add_array(OPT,
                         column_labels=column_labels,
                         description=description)
    visualizer.add_array(p, description="P")
    visualizer.show()

    return OPT[N]


if __name__ == "__main__":
    # Definition of interval: [start time, finish time, value]
    intervals = [
        [0, 3, 3],
        [1, 4, 2],
        [0, 5, 4],
        [3, 6, 1],
        [4, 7, 2],
        [3, 9, 5],
        [5, 10, 2],
        [8, 10, 1],
    ]
    solve(intervals)

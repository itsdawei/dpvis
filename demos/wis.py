"""Dynamic program visualization for weighted interval selection.

To run this example: python3 weighted_interval.py
"""
from dp import DPArray
from dp._visualizer import Visualizer, display


def weighted_interval_scheduling(intervals, OPT, p):
    """Dynamic program for Weighted Interval Selection.

    Args:
        intervals (array-like): a list of intervals. Each interval is
        a list with the following format [start time, finish time, weight].

    Returns:
        int: Maximum sum of weight of compatible intervals.
    """
    # Sort intervals by finish time
    intervals = sorted(intervals, key=lambda x: x[1])

    # Maximum of intervals selected
    N = len(intervals)

    # Compute p[i] = largest index j < i s.t. interval i is compatible with j
    # p = [0] * (N + 1)  # [-1, ..., -1]
    p[:] = 0
    for i, int_i in enumerate(intervals, start=1):
        # Search up to the ith interval
        for j, int_j in enumerate(intervals[:i], start=1):
            # Check that int_i and int_j are compatible
            if min(int_i[1], int_j[1]) - max(int_i[0], int_j[0]) < 1:
                p[i] = j

    # Base Case
    OPT[0] = 0

    for i, int_i in enumerate(intervals, start=1):
        OPT[i] = max(int_i[2] + OPT[p[i]], OPT[i - 1])

    return OPT[N]


# [start time, finish time, weight]
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

N = len(intervals)
OPT = DPArray(N + 1, array_name="Weighted Interval Scheduling", dtype=int)
p = DPArray(N + 1, logger=OPT.logger, array_name="P", dtype=int)

weighted_interval_scheduling(intervals, OPT, p)

column_labels = [f"{i} intervals" for i in range(N + 1)]
description = ("|     | Start | Finish | Weight |  p  |\n"
               "| :-: | :---: | :----: | :----: | :-: |\n")
for i, a in enumerate(intervals, start=1):
    description += f"| {i} | {a[0]} | {a[1]} | {a[2]} | {p._arr[i]} |\n"
visualizer = Visualizer()
visualizer.add_array(OPT, column_labels=column_labels, description=description)
visualizer.add_array(p)

app = visualizer.create_app()
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True)

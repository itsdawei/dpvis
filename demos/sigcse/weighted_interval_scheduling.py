import numpy as np
from dp import DPArray, display

def precompute_predecessors(sorted_ints):
    p = [0] * (len(sorted_ints) + 1)
    for i, int_i in enumerate(sorted_ints, start=1):
        # Search up to the ith interval.
        for j, int_j in enumerate(sorted_ints[:i], start=1):
            # Check that int_i and int_j are compatible.
            if min(int_i[1], int_j[1]) - max(int_i[0], int_j[0]) < 1:
                p[i] = j
    return p

# sorted_ints is ordered by increasing finish time.
def solve(sorted_ints, w):
    OPT = DPArray(len(sorted_ints) + 1)

    # PRECOMPUTE: p is a list where the ith element is the last interval that ends before interval i
    # p[i] = largest index j < i s.t. interval i is compatible with j.
    p = precompute_predecessors(sorted_ints)

    OPT[0] = 0
    for i in range(1, len(sorted_ints) + 1):
        OPT[i] = max(OPT[i-1], w[i] + OPT[p[i]])

    # print(OPT[len(sorted_ints)])
    # description= "$$\max(OPT(i-1), w_i + OPT(p_i))$$"
    display(OPT)

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

    # Sort intervals by finish time.
    intervals = sorted(intervals, key=lambda x: x[1])

    # Note that for ease of indexing, w is 1 indexed.
    w = [int_i[2] for int_i in intervals]
    w.insert(0, 0)

    solve(intervals, w)

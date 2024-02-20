# Weighted Interval Scheduling

- Given $N$ intervals, where interval $i$ starts at $s_i$, finishes at $f_i$,
  and has weight $v_i$.
- Two intervals are compatible if they don't overlap.
- Goal: find maximum weight subset of mutually compatible intervals.

## Greedy Algorithm Fails!

TODO

## Dynamic Programming Solution

> A fully executable example can be found on our [GitHub](https://github.com/itsdawei/dpvis/tree/main/demos/weighted_interval.py).

Let $OPT[i]$ denote the value of the optimal solution consisting of the
intervals $1, 2, ..., i$.

**CASE 1: $OPT$ includes the interval $i$.** In this case, $OPT$ cannot select
any jobs incompatible with $i$, and $OPT$ must include optimal solution to
problem consisting of remaining compatible jobs $1, 2, ..., p(i)$, where $p(i)$
denotes the last job that is compatible with $i$.

**CASE 2: $OPT$ doesn't includes the interval $i$.** In this case, $OPT$ must
include the optimal solution consisting of intervals $1, ..., i$.

We can implement this as follows:

```python linenums="1"
def WIS_DP(intervals):
    """Weighted interval selection dynamic program.

    Args:
        intervals (array-like): a list of intervals. Each interval is
        a list with the following format [start time, finish time, weight].

    Returns:
        int: Maximum sum of weight of compatible intervals.
    """
    # Number of intervals.
    N = len(intervals)

    # Sort intervals by finish time.
    intervals = sorted(intervals, key=lambda x: x[1])

    # OPT[i] = value of the optimal solution consisting of intervals 1,...,i
    OPT = [0] * (N + 1)

    # p[i] = largest index j < i s.t. interval i is compatible with j.
    p = [0] * (N + 1)
    for i, a in enumerate(intervals, start=1):
        # Search up to the ith interval.
        for j, b in enumerate(intervals[:i], start=1):
            # Check that a and b are compatible.
            if min(a[1], b[1]) - max(a[0], b[0]) < 1:
                p[i] = j

    # Base Case.
    OPT[0] = 0

    # Compute OPT.
    for i, a in enumerate(intervals, start=1):
        OPT[i] = max(a[2] + OPT[p[i]], OPT[i - 1])

    return OPT[N]
```

To visualize the dynamic program, replace the python lists with `DPArray`

```python linenums="1" hl_lines="1-2 21 24-25 39-50"
from dp import DPArray
from dp._visualizer import Visualizer, display

def solve(intervals):
    """Dynamic program for Weighted Interval Selection.

    Args:
        intervals (array-like): a list of intervals. Each interval is
        represented as a list of [start time, finish time, weight].

    Returns:
        int: Maximum sum of weight of compatible intervals.
    """
    # Number of intervals.
    N = len(intervals)

    # Sort intervals by finish time.
    intervals = sorted(intervals, key=lambda x: x[1])

    # OPT[i] = value of the optimal solution consisting of intervals 1,...,i
    OPT = DPArray(N + 1, array_name="Weighted Interval Scheduling", dtype=int)

    # Compute p[i] = largest index j < i s.t. interval i is compatible with j.
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

    # Add more information for the visualization.
    column_labels = [f"{i} intervals" for i in range(N + 1)]
    description = ("| Interval | Start | Finish | Weight | p |\n"
                   "| :------: | :---: | :----: | :----: | - |\n")
    for i, a in enumerate(intervals, start=1):
        description += f"| {i} | {a[0]} | {a[1]} | {a[2]} | {p._arr[i]} |\n"
    visualizer = Visualizer()
    visualizer.add_array(OPT,
                         column_labels=column_labels,
                         description=description)
    visualizer.add_array(p)
    visualizer.show()

    return OPT[N]
```

## Try it yourself!

You can use this example, for which the correct $OPT$ array should be `[0, 3,
3, 4, 4, 5, 8, 8, 8]`.
```python
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
```
Alternatively, make your own example and confirm that the above implementation of WIS is correct!

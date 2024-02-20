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
include the optimal solution consisting of intervals $1, ..., i-1$.

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
    for i, int_i in enumerate(intervals, start=1):
        # Search up to the ith interval.
        for j, int_j in enumerate(intervals[:i], start=1):
            # Check that int_i and int_j are compatible.
            if min(int_i[1], int_j[1]) - max(int_i[0], int_j[0]) < 1:
                p[i] = j

    # Base Case.
    OPT[0] = 0

    for i, int_i in enumerate(intervals, start=1):
        OPT[i] = max(int_i[2] + OPT[p[i]], OPT[i - 1])

    return OPT[N]
```

## Visualization with `dpvis`

To visualize the dynamic program, replace the python lists with `DPArray` and
add additional information to the visualization.

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
    for i, int_i in enumerate(intervals, start=1):
        # Search up to the ith interval.
        for j, int_j in enumerate(intervals[:i], start=1):
            # Check that int_i and int_j are compatible.
            if min(int_i[1], int_j[1]) - max(int_i[0], int_j[0]) < 1:
                p[i] = j

    # Base Case.
    OPT[0] = 0

    for i, int_i in enumerate(intervals, start=1):
        OPT[i] = max(int_i[2] + OPT[p[i]], OPT[i - 1])

    # Add more information for the visualization.
    column_labels = [f"{i} intervals" for i in range(N + 1)]
    visualizer = Visualizer()
    visualizer.add_array(OPT, column_labels=column_labels)
    visualizer.show()

    return OPT[N]
```

This is what you will see when you execute the above code.

<img src="../images/dparray_empty.png" width="75%"/>

On the top of the page is a slider to control what timestep is being
visualized. The slider can be used to show different timesteps by clicking and
dragging or using the <span
style="color:white;background-color:black">PLAY</span> and <span
style="color:white;background-color:black">STOP</span> buttons. Below the
slider is a visualization presenting the elements of the array on the current
timestep. The zeroth timestep shows the base case. The visualization shows that
the zeroth element of the array is set to zero, which corresponds to `OPT[0]
= 0` in our code.

Try dragging the slider to timestep 5.

<img src="../images/dparray_partial.png" width="75%"/>

Now the visual shows the array with the first six elements set. On this
timestep, elements two and four of the `OPT` array are <span
style="background-color:#b7609a">READ</span>, meaning we accessed the values of
those elements on this timestep. We also <span
style="background-color:#5c53a5">WRITE</span> a value of `5` to element five of
the `OPT` array. This corresponds to line 37 in our code when `i=5`:
```python
# a[2] is the weight of interval i
max(a[2] + OPT[p[i]], OPT[i - 1])
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

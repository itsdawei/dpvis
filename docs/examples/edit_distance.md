# Edit Distance

- Given two strings `str1` and `str2`, what is the minimum number of edits
- to convert `str1` to `str2`?
- Each edit costs 1, and is either an add, delete, or replace.
- Goal: find a minimum cost set of edits that converts `str1` to `str2`.

## Dynamic Programming Solution

> A fully executable example can be found on our [GitHub](https://github.com/itsdawei/dpvis/tree/main/demos/edit_distance_2d.py).

Let $OPT[i, j]$ denote the cheapest conversion between the first $i$ characters of `str1`
and the first $j$ characters of `str2`.

**BASE CASES: $i$ or $j$ is $0$.** At least one of our strings is empty, 
so we should just pay the cost to remove the remaining letters in the other string. So
we should record that $OPT[i, j] = max(i, j)$.

**CASE 1: `str1[i] == str2[j]`.** The last letter of `str1` is the same as
the last letter of `str2`. So we should leave the last letters alone and convert
`str1[i-1]` to `str2[j-1]` as cheaply as possible. So, $OPT[i, j] = OPT[i-1, j-1]$.

**CASE 2: `str1[i] != str2[j]`.** The last letter of `str1` is not the same as
the last letter of `str2`. At this point, we have to either add the last letter of 
`str2` to `str1`, delete the last letter of `str1`, or replace the last letter of 
`str1` with the last letter of `str2`. Each option is an edit, so we must record
$OPT[i, j] = 1 + min(OPT[i, j - 1], OPT[i - 1, j], OPT[i - 1, j - 1])$.

## Visualization with `dpvis`

We can visualize this with `dpvis` as follows:

```python linenums="1"  hl_lines="1 17 60-62"
from dp import DPArray, display

def edit_distance(str1, str2):
    """
    Edit Distance Problem:
    Given two strings str1 and str2 of lengths m and n, respectively, what is 
    the cost of the cheapest set of actions that converts str1 into str2. 
    The following actions are possible:
    add before/after index i, remove before/after index i, replace at index i

    Solution adapted from Bhavya Jain's solution:
    https://www.geeksforgeeks.org/edit-distance-dp-5/
    """
    m = len(str1)
    n = len(str2)
    # Initialize an m+1 x n+1 array
    OPT = DPArray((m + 1, n + 1), array_name="Edit Distance", dtype=int)

    # Base cases: either str1 or str2 is empty
    # Then we have to pay to remove/add the remaining letters
    for i in range(m + 1):
        OPT[i, 0] = i
    for j in range(n + 1):
        OPT[0, j] = j
    OPT.annotate("Base cases: no remaining letters in str1 or str2.")

    # Fill OPT[][] iteratively
    for i in range(m + 1):
        for j in range(n + 1):
            # Base case: either string is empty and has already been handled.
            if i == 0 or j == 0:
                pass

            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            elif str1[i - 1] == str2[j - 1]:
                OPT[i, j] = OPT[i - 1, j - 1]
                OPT.annotate("Last character same: pay OPT cost for remaining "
                             "strings.")

            # At this point the last characters are different, so consider
            # each possible action and pick the cheapest.
            else:
                indices = [
                    (i, j - 1),  # Insert
                    (i - 1, j),  # Remove
                    (i - 1, j - 1)  # Replace
                ]
                OPT[i, j] = 1 + OPT.min(indices=indices)
                OPT.annotate("Last characters different: test between insert, "
                             "remove, replace.")

    return OPT

if __name__ == "__main__":
    # Test Example
    str1 = "sunday"
    str2 = "saturday"

    dp_array = edit_distance(str1, str2)

    display(dp_array, row_labels="_" + str1, column_labels="_" + str2)
```

This is what you will see when you execute the above code.

<img src="../images/edit_distance_empty.png" width="75%"/>

On the top of the page is a slider to control what timestep is being
visualized. The slider can be used to show different timesteps by clicking and
dragging or using the <span
style="color:white;background-color:black">PLAY</span> and <span
style="color:white;background-color:black">STOP</span> buttons. Below the
slider is a visualization presenting the elements of the array on the current
timestep. The zeroth timestep shows the base cases (when one of the strings is
completely empty).

Try dragging the slider to timestep 5.

<img src="../images/edit_distance_partial.png" width="75%"/>

Now the visual shows that we're comparing `"s"` from `str1` to `"satur"` from 
`str2`. Since the last letters of the partial strings are not equal in this 
iteration, we pay one and choose between removing a letter from `str1`, 
removing a letter from `str2` or replacing a the last letter of `str1` 
with `str2`. In this case, removing the last letter of `str2` is the 
cheapest option with a cost of 3, so we pay 1 to delete the last letter of 
`str2` and then we can pay 3 more to convert from there onwards (TODO: clean 
up).
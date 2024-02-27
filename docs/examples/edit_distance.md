# Edit Distance

- The *edit distance* between two strings is the minimum total cost of the
  operations required to convert `str1` to `str2`?
- The allowed operations and their corresponding costs are as follows:
    - Inserting a letter costs $\alpha$.
    - Deleting a letter costs $\beta$.
    - Replacing a letter with a different letter costs $\gamma$.
- **Question**: Given two strings `str1` and `str2`, what is the *edit distance*
  between `str1` and `str2`?

## Dynamic Programming Solution

> A fully executable example can be found on our [GitHub](https://github.com/itsdawei/dpvis/tree/main/demos/edit_distance.py).

Let $OPT[i, j]$ denote the cheapest distance from the first $i$ characters of `str1`
to the first $j$ characters of `str2`.

**BASE CASE: $i$ is $0$.** `str1` is empty,
so we should just pay the cost of adding the remaining letters in `str2`. So
we should record that $OPT[i, j] = \alpha \cdot j$.

**BASE CASE: $j$ is $0$.** `str2` is empty,
so we should just pay the cost to removing the remaining letters in `str1`. So
we should record that $OPT[i, j] = \beta \cdot i$.

**CASE 1: `str1[i] == str2[j]`.** The last letter of `str1` is the same as
the last letter of `str2`. So we should leave the last letters alone and convert
`str1[i-1]` to `str2[j-1]` as cheaply as possible. So, $OPT[i, j] = OPT[i-1, j-1]$.

**CASE 2: `str1[i] != str2[j]`.** The last letter of `str1` is not the same as
the last letter of `str2`. At this point, we have to either add the last letter of
`str2` to `str1`, delete the last letter of `str1`, or replace the last letter of
`str1` with the last letter of `str2`. Each option is an edit, so we must record
$OPT[i, j] = \min(\alpha + OPT[i, j - 1], \beta + OPT[i - 1, j], \gamma + OPT[i - 1, j - 1])$.

## Visualization with `dpvis`

We can visualize this with `dpvis` as follows:

```python linenums="1"
from dp import DPArray, display


def edit_distance(str1, str2, alpha, beta, gamma):
    """Solution adapted from Bhavya Jain's solution: https://www.geeksforgeeks.org/edit-distance-dp-5/"""
    m = len(str1)
    n = len(str2)

    # Initialize an (m+1)x(n+1) array
    OPT = DPArray((m + 1, n + 1), array_name="Edit Distance", dtype=int)

    # Base cases: either str1 or str2 is empty
    # Then we have to pay to insert/delete the remaining letters
    for i in range(m + 1):
        OPT[i, 0] = beta * i
    for j in range(n + 1):
        OPT[0, j] = alpha * j

    # Fill OPT[][] iteratively
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            if str1[i - 1] == str2[j - 1]:
                OPT[i, j] = OPT[i - 1, j - 1]
                arr = OPT.arr
                continue

            # At this point the last characters are different, so consider
            # each possible action and pick the cheapest.
            indices = [
                (i, j - 1),  # insert
                (i - 1, j),  # remove
                (i - 1, j - 1)  # replace
            ]
            elements = [
                OPT[i, j - 1] + alpha,  # insert
                OPT[i - 1, j] + beta,  # remove
                OPT[i - 1, j - 1] + gamma,  # replace
            ]

            OPT[i, j] = OPT.min(indices=indices, elements=elements)
    return OPT


if __name__ == '__main__':
    str1 = "sunday"
    str2 = "saturday"
    # FOR CSCI 270 HW6: DO NOT CHANGE THE COSTS
    ALPHA = 10
    BETA = 12
    GAMMA = 7

    dp_array = edit_distance(str1, str2, ALPHA, BETA, GAMMA)

    display(dp_array,
            row_labels=f"_{str1}",
            column_labels=f"_{str2}")
```

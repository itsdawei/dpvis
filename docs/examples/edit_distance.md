# Edit Distance

- Given two strings `str1` and `str2`, what is the minimum number of edits
  to convert `str1` to `str2`?
- We can insert a letter in `str1` for a cost of $\alpha$, delete a letter from
  `str1` for a cost of $\beta$, or substitue a letter of `str1` with a letter
  from `str2` for a cost of $\gamma$.
- Goal: find a minimum cost set of edits that converts `str1` to `str2`.

## Dynamic Programming Solution

> A fully executable example can be found on our [GitHub](https://github.com/itsdawei/dpvis/tree/main/demos/edit_distance.py).

Let $OPT[i, j]$ denote the cheapest distance from the first $i$ characters of `str1`
to the first $j$ characters of `str2`.

**BASE CASE: $i$ is $0$.** `str1` is empty,
so we should just pay the cost of adding the remaining letters in `str2`. So
we should record that $OPT[i, j] = \alpha * j$.

**BASE CASE: $j$ is $0$.** `str2` is empty,
so we should just pay the cost to removing the remaining letters in `str1`. So
we should record that $OPT[i, j] = \beta * i$.

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
    """Adapted from Bhavya Jain's solution: https://www.geeksforgeeks.org/edit-distance-dp-5/"""
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
    OPT.annotate("No remaining letters in str1 or str2.")

    # Fill OPT[][] iteratively
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Base case: either string is empty and has already been handled.
            annotate_string = "str1: " + str1[:i]
            annotate_string += ", str2: " + str2[:j] + " " + ("_" * 50) + " "

            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            if str1[i - 1] == str2[j - 1]:
                OPT[i, j] = OPT[i - 1, j - 1]
                annotate_string += "Last character same: pay OPT cost for remaining strings."

            # At this point the last characters are different, so consider
            # each possible action and pick the cheapest.
            else:
                indices = [
                    (i, j - 1),  # Insert
                    (i - 1, j),  # Remove
                    (i - 1, j - 1)  # Replace
                ]
                elements = [
                    OPT[i, j - 1] + alpha, OPT[i - 1, j] + beta,
                    OPT[i - 1, j - 1] + gamma
                ]

                OPT[i, j] = OPT.min(indices=indices, elements=elements)
                arr = OPT.arr
                if min(arr[i, j - 1] + alpha, arr[i - 1, j] + beta,
                       arr[i - 1, j - 1] + gamma) == arr[i, j - 1] + alpha:
                    annotate_string += "Insert the last letter str2 to end of str1, obtaining str1 = " + str1[:
                                                                                                              i]
                    annotate_string += str2[
                        j -
                        1] + " and str2 = " + str2[:j] + ". Then, since the last letters"
                    annotate_string += " are the same, iterate to str1 = " + str1[:
                                                                                  i] + " and str2 = " + str2[:j
                                                                                                             -
                                                                                                             1] + "."
                elif min(arr[i, j - 1] + alpha, arr[i - 1, j] + beta,
                         arr[i - 1, j - 1] + gamma) == arr[i - 1, j] + beta:
                    annotate_string += "Delete the last letter of str1."
                else:
                    annotate_string += "Substitute the last letter of str1 with the last letter of str2, obtaining str1 = "
                    annotate_string += str1[:i - 1] + str2[
                        j -
                        1] + " and str2 = " + str2[:j] + ". Then, since the last letters"
                    annotate_string += " are the same, iterate to str1 = " + str1[:i
                                                                                  -
                                                                                  1] + " and str2 = " + str2[:j
                                                                                                             -
                                                                                                             1] + "."

            OPT.annotate(annotate_string)

    return OPT


# Test Example
str1 = "sunday"
str2 = "saturday"
ALPHA = 10
BETA = 12
GAMMA = 7

dp_array = edit_distance(str1, str2, ALPHA, BETA, GAMMA)

description = "# Edit Distance \n\n"
description += "Change \"*" + str1 + "*\" to \"*" + str2 + "*\""
description += "\n\n Cost of inserting to string 1: " + str(ALPHA)
description += "\n\n Cost of deleting from string 1: " + str(BETA)
description += "\n\n Cost of substituting last letter of string 1: " + str(
    GAMMA)

display(dp_array,
        description=description,
        row_labels="_" + str1,
        column_labels="_" + str2)
```

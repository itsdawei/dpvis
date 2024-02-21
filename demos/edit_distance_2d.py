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
    OPT = DPArray((m + 1, n + 1), array_name="Edit Distance")

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


# Test Example
str1 = "sunday"
str2 = "saturday"

dp_array = edit_distance(str1, str2)

display(dp_array, row_labels="_" + str1, column_labels="_" + str2)
"""
Question for student:
1. How can we optimize the above DP to use a 2 x m array?
2. How can we optimize the above DP to use just a 1d array?
3. The above solution works when the cost of each action is 1, how
   would you modify it if the cost of an add is alpha, the cost
   of a remove is beta, and the cost of a replace is tau?
"""

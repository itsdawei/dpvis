from dp import DPArray, display


def edit_distance(str1, str2, alpha, beta, gamma):
    """
    Edit Distance Problem:
    Given two strings str1 and str2 of lengths m and n, respectively, what is 
    the cost of the cheapest set of actions that converts str1 into str2?
    The following actions are possible:
    add before/after index i - costs alpha
    remove before/after index i - costs beta
    replace at index i - costs gamma

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
        OPT[i, 0] = min(alpha, beta) * i
    for j in range(n + 1):
        OPT[0, j] = min(alpha, beta) * j
    OPT.annotate("No remaining letters in str1 or str2.")

    # Fill OPT[][] iteratively
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Base case: either string is empty and has already been handled.
            annotate_string = "str1: " + str1[:i]
            annotate_string += ", str2: " + str2[:j] + " " + ("_"*50)
            
            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            if str1[i - 1] == str2[j - 1]:
                OPT[i, j] = OPT[i - 1, j - 1]
                annotate_string += "Last character same: pay OPT cost for remaining strings."

            # At this point the last characters are different, so consider
            # each possible action and pick the cheapest.
            else:
                OPT[i, j] = min(OPT[i, j - 1] + alpha, 
                                OPT[i-1, j] + beta,
                                OPT[i-1, j - 1] + gamma)
                arr = OPT.arr
                if min(arr[i, j - 1] + alpha, arr[i-1, j] + beta, arr[i-1, j - 1] + gamma) == arr[i, j-1] + alpha:
                    annotate_string += "Add the last letter str2 to end of str1."
                elif min(arr[i, j - 1] + alpha, arr[i-1, j] + beta, arr[i-1, j - 1] + gamma) == arr[i-1, j] + beta:
                    annotate_string += "Remove the last letter of str1."
                else:
                    annotate_string += "Replace the last letter of str1 with the last letter of str2."

            OPT.annotate(annotate_string)

    return OPT


# Test Example
str1 = "sunday"
str2 = "saturday"
ALPHA = 10
BETA = 12
GAMMA = 3


dp_array = edit_distance(str1, str2, ALPHA, BETA, GAMMA)


description = "# Edit Distance \n\n"
description += "Change \"*" + str1 + "*\" to \"*" + str2 + "*\""
description += "\n\n Cost of add: " + str(ALPHA)
description += "\n\n Cost of remove: " + str(BETA)
description += "\n\n Cost of swap: " + str(GAMMA)

display(dp_array, description=description,  row_labels="_" + str1, column_labels="_" + str2)
"""
Question for student:
1. How can we optimize the above DP to use a 2 x m array?
2. How can we optimize the above DP to use just a 1d array?
3. The above solution works when the cost of each action is 1, how
   would you modify it if the cost of an add is alpha, the cost
   of a remove is beta, and the cost of a replace is tau?
"""

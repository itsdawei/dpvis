from dp import DPArray, display


def sequence_alignment(str1, str2, alpha, delta):
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
    OPT = DPArray((m + 1, n + 1), array_name="String Alignment", dtype=int)

    # Base cases: either str1 or str2 is empty
    # Then we have to pay to remove/add the remaining letters
    for i in range(m + 1):
        OPT[i, 0] = delta * i
    for j in range(n + 1):
        OPT[0, j] = delta * j
    OPT.annotate("No remaining letters in str1 or str2.")

    # Fill OPT[][] iteratively
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Base case: either string is empty and has already been handled.
            annotate_string = "str1: " + str1[:i]
            annotate_string += ", str2: " + str2[:j] + " " + ("_"*50) + " "
            
            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            if str1[i - 1] == str2[j - 1]:
                OPT[i, j] = OPT[i - 1, j - 1]
                annotate_string += "Last character same: pay OPT cost for remaining strings."

            # At this point the last characters are different, so consider
            # each possible action and pick the cheapest.
            else:
                OPT[i, j] = min(OPT[i, j - 1] + delta, 
                                OPT[i-1, j] + delta,
                                OPT[i-1, j - 1] + alpha)
                arr = OPT.arr
                if min(arr[i, j - 1] + delta, arr[i-1, j] + delta, arr[i-1, j - 1] + alpha) == arr[i, j-1] + delta:
                    annotate_string += "Add a gap at the end of string 1. Match the last letter of string 2 with that gap."
                elif min(arr[i, j - 1] + delta, arr[i-1, j] + delta, arr[i-1, j - 1] + alpha) == arr[i-1, j] + delta:
                    annotate_string += "Add a gap at the end of string 2. Match the last letter of string 1 with that gap."
                else:
                    annotate_string += "Align the last letter of string 1 to the last letter of string 2. Pay cost of misalignment."

            OPT.annotate(annotate_string)

    return OPT


# Test Example
str1 = "sunday"
str2 = "saturday"
# Alpha is the cost of mismatched letters.
ALPHA = 10
# Delta is the cost of adding a gap.
DELTA = 12


dp_array = sequence_alignment(str1, str2, ALPHA, DELTA)


description = "# Sequence Alignment \n\n"
description += "Find the min-cost alignment between \"*" + str1 + "*\" and \"*" + str2 + "*\""
description += "\n\n Cost of mismatching two letters: " + str(ALPHA)
description += "\n\n Cost of adding a gap: " + str(DELTA)

display(dp_array, description=description,  row_labels="_" + str1, column_labels="_" + str2)
"""
Question for student:
1. How can we optimize the above DP to use a 2 x m array?
2. How can we optimize the above DP to use just a 1d array?
3. The above solution works when the cost of each action is 1, how
   would you modify it if the cost of an add is alpha, the cost
   of a remove is beta, and the cost of a replace is tau?
"""

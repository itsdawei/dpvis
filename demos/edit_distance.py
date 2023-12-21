from dp import DPArray, display

"""
Edit Distance Problem:
Given two strings str1 and str2 of lengths m and n, respectively, what is 
the cost of the cheapest set of actions that converts str1 into str2. 
The following actions are possible:
add before/after index i, remove before/after index i, replace at index i

Solution adapted from Bhavya Jain's solution:
https://www.geeksforgeeks.org/edit-distance-dp-5/
"""


def edit_distance(str1, str2, m, n):
    # Initialize an m+1 x n+1 array
    OPT = DPArray((m+1, n+1), array_name="Edit Distance")

    # __import__("pdb").set_trace();

    # Base cases: either str1 or str2 is empty
    # Then we have to pay to remove/add the remaining letters
    for i in range(m+1):
        OPT[i, 0] = i
    for j in range(n+1):
        OPT[0, j] = j

    # Fill OPT[][] iteratively
    for i in range(m+1):
        for j in range(n+1):
            # Base case: either string is empty and has already been handled.
            if i == 0 or j == 0:
                pass

            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            elif str1[i-1] == str2[j-1]:
                OPT[i, j] = OPT[i-1, j-1]

            # At this point the last characters are different, so consider
            # each possible action and pick the cheapest.
            else:
                indices = [
                    (i, j-1),       # Insert
                    (i-1, j),       # Remove
                    (i-1, j-1)      # Replace
                ]
                elements = [
                    OPT[i, j-1],       # Insert
                    OPT[i-1, j],       # Remove
                    OPT[i-1, j-1]      # Replace
                ]
                OPT[i, j] = 1 + OPT.min(indices=indices, elements=elements)
    
    return OPT

# Test Example
str1 = "sunday"
str2 = "saturday"


dp_array = edit_distance(str1, str2, len(str1), len(str2))
display(dp_array)

"""
Question for student:
1. How can we optimize the above DP to use a 2*len(str1) array?
2. How can we optimize the above DP to use just a 1d array?
3. The above problem works when the cost of each action is 1, how
   would you modify it if the cost of an add is alpha, the cost
   of a remove is beta, and the cost of a replace is tau?
"""

def edit_distance_1d(str1, str2, alpha, beta, tau):
    pass
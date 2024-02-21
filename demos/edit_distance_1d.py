from dp import DPArray, display

def edit_distance_1d(str1, str2):
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
    curr = DPArray(n+1, array_name="OPT")
    for j in range(n+1):
        curr[j] = j
    
    previous = 0

    for i in range(1, m+1):
        # Store the current value at the beginning of the row.
        previous = curr[0]
        curr[0] = i

        # Loop through the columns of the DPArray
        for j in range(1, n+1):
            # Store the current value in a temporary variable.
            temp = curr[j]
            
            # Check if the characters at the current positions in str1 and str2 are the same
            if str1[i-1] == str2[j-1]:
                curr[j] = previous
            else:
                # Update the current cell with the minimum pf the three adjacent cells
                curr[j] = 1 + min(previous, curr[j-1], curr[j])

            # Update the previous variable with the temporary value
            previous = temp
    
    # The value in the last cell represents the minimum number of operations
    display(curr)
    return curr[n]

str1 = "sit"
str2 = "kiit"

ans = edit_distance_1d(str1, str2)
print(ans)

def edit_distance_different_costs(str1, str2, alpha, beta, tau):
    pass
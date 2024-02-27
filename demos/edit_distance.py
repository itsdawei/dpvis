from dp import DPArray, display


def edit_distance(str1, str2, alpha, beta, gamma):
    """
    Edit Distance Problem:
    Given two strings str1 and str2 of lengths m and n, respectively, what is 
    the cost of the cheapest set of actions that converts str1 into str2?
    The following operations are possible:
    - Insert a letter:  costs alpha
    - Delete a letter: costs beta
    - Replace a letter: costs gamma

    Solution adapted from Bhavya Jain's solution:
    https://www.geeksforgeeks.org/edit-distance-dp-5/
    """
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
    OPT.annotate("**Base cases:** `str1 = ''` or `str2 = ''`")

    # Fill OPT[][] iteratively
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Base case: either string is empty and has already been handled.
            annotation = f"`str1 = '{str1[:i]}'` and `str2 = '{str2[:j]}'`\n\n"

            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            if str1[i - 1] == str2[j - 1]:
                OPT[i, j] = OPT[i - 1, j - 1]
                arr = OPT.arr
                annotation += (
                    f"`'{str1[i-1]}'` and `'{str2[j-1]}'` are equal\n\n"
                    f"Now we invoke the optimal substructure "
                    f"`OPT['{str1[:i-1]}']['{str2[:j-1]}'] = {arr[i-1, j-1]}`")
                OPT.annotate(annotation)
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
            arr = OPT.arr
            annotation += "The most efficient operation is to "
            if arr[i, j] == arr[i, j - 1] + alpha:
                a = str1[:i]
                b = str2[:j - 1]
                opt_ab = arr[i, j - 1]
                annotation += f"**append `'{str2[j-1]}'` to `str1`**\n\n"
            elif arr[i, j] == arr[i - 1, j - 1] + gamma:
                a = str1[:i - 1]
                b = str2[:j - 1]
                opt_ab = arr[i - 1, j - 1]
                annotation += (f"**replace `str1[{i}] = '{str1[i-1]}'` with "
                               f"`'{str2[j-1]}'`**\n\n")
            elif arr[i, j] == arr[i - 1, j] + beta:
                a = str1[:i - 1]
                b = str2[:j]
                opt_ab = arr[i - 1, j]
                annotation += f"**delete `'{str1[i-1]}'` from `str1`**\n\n"
            annotation += (f"Now, we need to find the edit distance between "
                           f"`'{a}'` and `'{b}'`. Using the DP array, we can "
                           f"read off the optimal value for the substructure "
                           f"`OPT['{a}']['{b}'] = {opt_ab}`")
            OPT.annotate(annotation)
    return OPT


if __name__ == '__main__':
    str1 = "sunday"
    str2 = "saturday"
    # FOR CSCI 270 HW6: DO NOT CHANGE THE COSTS
    ALPHA = 10
    BETA = 12
    GAMMA = 7

    dp_array = edit_distance(str1, str2, ALPHA, BETA, GAMMA)

    desc = (f"# Edit Distance\n\n"
            f"**PROBLEM**: Change `str1 = {str1}` to `str2 = '{str2}'`\n\n"
            f"**OPERATIONS ALLOWED ON `str1` AND THEIR COSTS**:\n"
            f"- Inserting ($$\\alpha$$) = {ALPHA}\n"
            f"- Deleting ($$\\beta$$) = {BETA}\n"
            f"- Replacing ($$\\gamma$$) = {GAMMA}\n")

    display(dp_array,
            description=desc,
            row_labels=f"_{str1}",
            column_labels=f"_{str2}")

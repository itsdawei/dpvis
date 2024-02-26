from dp import DPArray, display


def edit_distance(str1, str2, alpha, beta, gamma):
    """
    Edit Distance Problem:
    Given two strings str1 and str2 of lengths m and n, respectively, what is 
    the cost of the cheapest set of actions that converts str1 into str2?
    The following actions are possible:
    insert before/after index i - costs alpha
    delete before/after index i - costs beta
    substitute at index i - costs gamma

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
    OPT.annotate("Base case: str1 or str2 is empty")

    # Fill OPT[][] iteratively
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Base case: either string is empty and has already been handled.
            annotate_string = f"str1: {str1[:i]}, str2: {str2[:j]} \n\n"

            # If last characters are the same, pay nothing and pay the optimal
            # costs for the remaining strings.
            if str1[i - 1] == str2[j - 1]:
                OPT[i, j] = OPT[i - 1, j - 1]
                annotate_string += ("Last character same: pay OPT cost for "
                                    "remaining strings.")

            # At this point the last characters are different, so consider
            # each possible action and pick the cheapest.
            else:
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
                if OPT[i, j] == arr[i, j - 1] + alpha:
                    annotate_string += (f"Insert the last letter str2 (\'{str1[i]}\') to end "
                                        f"of str1, obtaining str1 = {str1[:i]} "
                                        f"{str2[j-1]} and str2 = {str2[:j]}."
                                        f"Then, since the last letters are "
                                        f"the same, iterate to str1 = "
                                        f"{str1[:i]} str2 = {str2[:j - 1]}.")
                elif OPT[i, j] == arr[i - 1, j - 1] + gamma:
                    annotate_string += (f"Substitute the last letter of str1 "
                                        f"with the last letter of str2, "
                                        f"obtaining str1 = {str1[:i - 1]}"
                                        f" {str2[j - 1]} and str2 = "
                                        f"{str2[:j]}. Then, since the last "
                                        f"letters are the same, iterate to str1"
                                        f" = {str1[:i - 1]} and str2 = "
                                        f"{str2[:j-1]}.")
                elif OPT[i, j] == arr[i - 1, j] + beta:
                    annotate_string += "Delete the last letter of str1."

            OPT.annotate(annotate_string)

    return OPT


if __name__ == '__main__':
    str1 = "sunday"
    str2 = "saturday"
    # FOR CSCI 270 HW6: DO NOT CHANGE THE COSTS
    ALPHA = 10
    BETA = 12
    GAMMA = 7

    dp_array = edit_distance(str1, str2, ALPHA, BETA, GAMMA)

    desc = (f"# Edit Distance \n\n"
            f"Change \"*{str1}*\" to \"*{str2}*\""
            f"\n\n Cost of inserting to string 1: {ALPHA}"
            f"\n\n Cost of deleting from string 1: {BETA}"
            f"\n\n Cost of substituting last letter of string 1: {GAMMA}")

    display(dp_array,
            description=desc,
            row_labels=f"_{str1}",
            column_labels=f"_{str2}")

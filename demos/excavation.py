from dp import DPArray, display
import numpy as np
import fire

# Problem
# There are N different mining areas: {1,...,n}
# Each mining areas has L different layers
# Level l of mining area i provides the miner value v[i, l]
# It takes the miner one month to mine one level of one mining area
# Before the miner mines level 1,...,l-1 they must have mined level l
# Goal: find the maximum value that the miner can achieve if they have M months


# v has an added row and column of zeroes so 1-indexing can be used
# M is the number of months available to the miner
def excavate(v=np.array([[0, 0, 0, 0, 0, 0, 0], [5, 4, 2, 5, 4, 3, 4],
                         [1, 4, 3, 4, 2, 0, 1], [3, 5, 0, 2, 4, 3, 4],
                         [2, 4, 4, 4, 1, 3, 0], [4, 2, 5, 0, 5, 0, 5],]),
             M=10):

    # Define problem constants and height and width of OPT
    N = v.shape[0] - 1
    L = v.shape[1] - 1
    # Make DPArray height and width
    h, w = (N + 1, min(N * L, M) + 1
           )  # Note that padding is added so everything is one indexed

    # Initialize DPArray
    # OPT[i, m] is the maximum value the miner can get from mines {1,...,i} in no more than m months
    row_labels = ["Padding" if i == 0 else f'Mine {i}' for i in range(h)]
    column_labels = ["Padding" if i == 0 else f'Month {i}' for i in range(w)]
    OPT = DPArray((h, w), row_labels=row_labels, column_labels=column_labels)

    # Add padding
    OPT[:, 0] = 0
    OPT[0, :] = 0

    # For each mining area
    for i in range(1, h):
        # For each possible number of months
        for m in range(1, w):
            indices = []
            elements = []
            # for each number of levels to dig in mining area i (digging l levels in i)
            for l in range(0, m):
                indices.append((i - 1, m - l))
                # np.sum(v[i, 1:(l + 2)]) is the value the miner gets from digging l levels in mine i
                # OPT[i - i, m - l] is the value the miner gets from digging in the mines 1,...,i-1 for no more than m - l months
                elements.append(OPT[i - 1, m - l] + np.sum(v[i, 1:(l + 1)]))
            OPT[i, m] = OPT.max(indices=indices, elements=elements)

    display(OPT)


if __name__ == "__main__":
    fire.Fire(excavate)

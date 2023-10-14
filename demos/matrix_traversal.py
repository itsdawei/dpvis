from dp import DPArray, display
import numpy as np
import fire

# Problem
# There is a matrix c of size n x m, where c[x, y] is the cost of the element at (x, y)
# A traveler starts at index (1, 1) (assume 1-indexing)
# The goal of the traveler is to reach element (n, n) while incurring the lowest cost possible
# The traveler's movement is constrained.: f they are at element (x, y), then they may only travel to (x+1, y) or (x, y+1)


# v has an added row and column of zeroes so 1-indexing can be used
def excavate(c=np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 4, 2, 5, 4, 3, 4],
                         [0, 1, 4, 3, 4, 2, 0, 1], [0, 3, 5, 0, 2, 4, 3, 4],
                         [0, 2, 4, 4, 4, 1, 3, 0], [0, 4, 2, 5, 0, 5, 0, 5]])):

    row_labels = ['Padding' if i == 0 else str(i) for i in range(c.shape[0])]
    column_labels = ['Padding' if j == 0 else str(j) for j in range(c.shape[1])]
    OPT = DPArray(shape=c.shape, row_labels=row_labels, column_labels=column_labels)

    # Add padding
    OPT[:, 0] = np.inf
    OPT[0, :] = np.inf
    OPT[1, 1] = c[1, 1]

    for i in range(1, c.shape[0]):
        for j in range(1, c.shape[1]):
            if i == 1 and j == 1:
                continue
            OPT[i,
                j] = c[i, j] + OPT.min(indices=[(i - 1, j), (i, j - 1)],
                                       elements=[OPT[i - 1, j], OPT[i, j - 1]])
    display(OPT)


if __name__ == "__main__":
    fire.Fire(excavate)

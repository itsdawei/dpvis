from dp import DPArray, display
from dp._visualizer import Visualizer
import numpy as np


# An item is a 2-tuple with (space, value)
# items is a list of items in the problem instance.
def knapsack(items, capacity):
    # Initialize DPArray

    OPT = DPArray((len(items) + 1, capacity + 1))
    DP_items = DPArray(len(items), array_name="Items", logger=OPT.logger)
    for i in range(len(items)):
        DP_items[i] = i

    # Put in base cases
    OPT[0, :] = 0
    OPT[:, 0] = 0
    # Recurrence: OPT(i, C) = max(OPT(i-1, C), OPT(i-1, C-i.space) + i.val)
    for idx in range(len(items) + 1):
        for rem in range(capacity + 1):
            # Base case: 0 value if there are no items left or if there is no space.
            if idx == 0 or rem == 0:
                continue
            # Normal case: There is an item to add and space remaining
            item = items[idx - 1]
            if idx >= 1 and rem - item[0] >= 0:
                OPT[idx, rem] = max(OPT[idx - 1, rem], OPT[idx - 1, rem-item[0]] + item[1])
            # Case: adding item is not possible
            elif idx >= 1 and rem - item[0] < 0:
                OPT[idx, rem] = OPT[idx - 1, rem]

    print(OPT[len(items), capacity])
    display(OPT)
    # visualizer = Visualizer()
    # visualizer.add_array(OPT)
    # visualizer.add_array(DP_items)
    # visualizer.show()

if __name__ == "__main__":
    items = [(2, 4), (4, 3), (7, 12), (5, 6), (13, 13)]
    max_capacity = 14
    knapsack(items, max_capacity)

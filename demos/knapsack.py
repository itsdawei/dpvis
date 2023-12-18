from dp import DPArray
from dp._visualizer import Visualizer


# An item is a 2-tuple with (space, value)
# items is a list of items in the problem instance.
def knapsack(items, capacity):
    # Initialize DPArray
    OPT = DPArray((len(items) + 1, capacity + 1), array_name="Knapsack")
    DP_items = DPArray(shape=len(items), array_name="Items", logger=OPT.logger)
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
                _ = DP_items[idx - 1]
                # OPT[idx, rem] = max(OPT[idx - 1, rem], OPT[idx - 1, rem-item[0]] + item[1])
                indices = [
                    (idx - 1, rem),
                    (idx - 1, rem - item[0]),
                ]
                elements = [
                    OPT[idx - 1, rem],
                    OPT[idx - 1, rem - item[0]] + item[1],
                ]
                OPT[idx, rem] = OPT.max(indices=indices, elements=elements)
                OPT.annotate(f"Comparing: max({indices[0]}, {indices[1]})")
            # Edge case: adding item is not possible
            elif idx >= 1 and rem - item[0] < 0:
                _ = DP_items[idx - 1]
                OPT[idx, rem] = OPT[idx - 1, rem]
                OPT.annotate("Item does not fit in remaining capacity.")

    # Make a copy of data in OPT to prevent visualization of future operations.
    arr = OPT.arr

    # Recover a traceback path.
    current = (arr.shape[0] - 1, arr.shape[1] - 1)
    path = [current]
    solution = []  # List of items.

    # While the path is not fully constructed.
    while current[0] != 0 and current[1] != 0:
        i, rem = current
        (capacity, value) = items[i - 1]

        # Find the predecessor of current.
        # Case 1: adding item is possible and more optimal
        if rem - capacity >= 0 and arr[i - 1, rem] < arr[i - 1, rem -
                                                        capacity] + value:
            current = (i - 1, rem - capacity)
            path.append(current)
            solution.append(i)

        # Case 2: there is no capacity for item or not adding item is more optimal
        else:
            current = (i - 1, rem)
            path.append(current)

    path = path[::-1]
    solution = solution[::-1]
    OPT.add_traceback_path(path)

    # Define labels.
    row_labels = [f"Item {i+1}: {item}" for i, item in enumerate(items)]
    row_labels.insert(0, "No item")
    column_labels = [f"Capacity {i}" for i in range(capacity + 1)]
    description = """
Recurrence: $OPT(i, C) = \max(OPT(i-1, C), OPT(i-1, C-c(i)) + v(i))$
```python
if idx >= 1 and rem - item[0] >= 0:
    OPT[idx, rem] = max(OPT[idx - 1, rem], OPT[idx - 1, rem-item[0]] + item[1])
elif idx >= 1 and rem - item[0] < 0:
    OPT[idx, rem] = OPT[idx - 1, rem]
```
"""

    # Visualize with the items array
    visualizer = Visualizer()
    visualizer.add_array(OPT,
                         row_labels=row_labels,
                         column_labels=column_labels,
                         description=description)
    item_col_labels = [f"Item {i}: {item}" for i, item in enumerate(items)]
    visualizer.add_array(DP_items,
                         row_labels=" ",
                         column_labels=item_col_labels)
    visualizer.show()


if __name__ == "__main__":
    items = [(2, 4), (4, 3), (7, 12), (5, 6), (13, 13)]
    capacity = 14
    knapsack(items, capacity)

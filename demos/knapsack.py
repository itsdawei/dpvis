from dp import DPArray, display


# An item is a 2-tuple with (space, value)
# items is a list of items in the problem instance.
def knapsack(items, capacity):
    # Adding a filler element since python is 0-indexed.
    items.insert(0, (0, -1))

    # Initialize DPArray.
    OPT = DPArray((len(items), capacity + 1))

    # Put in base cases.
    OPT[0, :] = 0
    OPT[:, 0] = 0
    # Recurrence: OPT(i, C) = max(OPT(i-1, C), OPT(i-1, C-i.space) + i.val).
    for idx, item in enumerate(items):
        for rem in range(capacity + 1):
            # Base case: 0 value if there are no items left or
            # if there is no space.
            if idx == 0 or rem == 0:
                continue

            # Item is not added
            indices = [(idx - 1, rem)]
            elements = [OPT[idx - 1, rem]]

            # If adding item is possible
            if rem - item[0] >= 0:
                # Item is added
                indices.append((idx - 1, rem - item[0]))
                elements.append(OPT[idx - 1, rem - item[0]] + item[1])
            
            OPT[idx, rem] = OPT.max(indices=indices, elements=elements)

    # Make a copy of data in OPT to prevent visualization of future operations.
    arr = OPT.arr

    # Recover a traceback path.
    current = (arr.shape[0] - 1, arr.shape[1] - 1)
    path = [current]
    solution = []  # Format: list of items.

    # While the path is not fully constructed.
    while current[0] != 0 and current[1] != 0:
        idx = current[0]
        rem = current[1]
        item = items[idx]

        # Find the predecessor of current.
        # Case 1: adding item is possible and more optimal
        if rem - item[0] >= 0 and arr[idx - 1, rem] < arr[idx - 1, rem - item[0]] + item[1]:
            current = (idx - 1, rem - item[0])
            path.append(current)
            solution.append(idx)

        # Case 2: there is no capacity for item or not adding item is more optimal
        else:
            current = (idx - 1, rem)
            path.append(current)

    path = path[::-1]
    solution = solution[::-1]
    OPT.add_traceback_path(path)

    # Define labels.
    row_labels = [f"Item {i}: {item}" for i, item in enumerate(items)]
    column_labels = [f"Capacity {i}" for i in range(capacity + 1)]
    display(OPT, row_labels=row_labels, column_labels=column_labels)


if __name__ == "__main__":
    items = [(2, 4), (4, 3), (7, 12), (5, 6), (13, 13)]
    capacity = 14
    knapsack(items, capacity)

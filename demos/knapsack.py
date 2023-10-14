from dp import DPArray, display
import fire


# An item is a 2-tuple with (space, value)
# items is a list of items in the problem instance.
def knapsack(items=[(2, 4), (4, 3), (7, 12), (5, 6), (13, 13)], capacity=14):
    # Adding a filler element since python is 0-indexed
    items.insert(0, (0, -1))

    # Initialize DPArray
    OPT = DPArray((len(items), capacity + 1))

    # Recurrence: OPT(i, C) = max(OPT(i-1, C), OPT(i-1, C-i.space) + i.val)
    for idx, item in enumerate(items):
        for rem in range(capacity + 1):
            # Base case: 0 value if there are no items left or if there is no space.
            if idx == 0 or rem == 0:
                OPT[idx, rem] = 0
            # Normal case: There is an item to add and space remaining
            if idx >= 1 and rem - item[0] >= 0:
                elements = [
                    OPT[idx - 1, rem], OPT[idx - 1, rem - item[0]] + item[1]
                ]
                indices = [(idx - 1, rem), (idx - 1, rem - item[0])]
                OPT[idx, rem] = OPT.max(indices=indices, elements=elements)
            # Edge case: adding item is not possible
            elif idx >= 1 and rem - item[0] < 0:
                OPT[idx, rem] = OPT.max(indices=[(idx - 1, rem)],
                                        elements=[OPT[(idx - 1, rem)]])

    display(OPT)


if __name__ == "__main__":
    fire.Fire(knapsack)

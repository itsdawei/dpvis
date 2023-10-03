from dp import DPArray, display
import fire


# An item is a 2-tuple with (space, value)
# items is a list of items in the problem instance.
def knapsack(items=[(2, 4), (4, 3), (7, 12), (5, 6), (13, 13)], capacity=14):
    OPT = DPArray((len(items), capacity))

    # Recurrence: OPT(i, C) = max(OPT(i-1, C), OPT(i-1, C-i.space) + i.val)
    for idx, item in enumerate(items):
        for rem in range(capacity):
            # Base case: 0 value if there are no items left or if there is no space.
            if idx == 0 or rem == 0:
                OPT[idx, rem] = 0
            # Normal case: There is an item to add and space remaining
            elif idx >= 1 and rem - item[0] >= 0:
                OPT[idx, rem] = max(OPT[idx - 1, rem],
                                    OPT[idx - 1, rem - item[0]] + item[1])
            # Edge case: adding item is not possible
            elif idx >= 1 and rem - item[0] < 0:
                OPT[idx, rem] = OPT[idx - 1, rem]
    display(OPT)


if __name__ == "__main__":
    fire.Fire(knapsack)

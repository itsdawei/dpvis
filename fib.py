from dp import DPArray, Op

def fib(n):
    """dynamic programming fibonacci
        fib(0) = 0
        fib(1) = 1
        fib(2) = 1
        ...
    """

    if n <= 1:
        return n

    arr = DPArray(n+1)
    arr[0] = 0
    print(arr.logger.logs)
    arr[1] = 1

    for i in range(2, n+1):
        arr[i] = arr[i-1] + arr[i-2]

    timesteps = arr.logger.to_timesteps()
    for i, ts in enumerate(timesteps):
        print(i)
        for name, shape in arr.logger.array_shapes.items(): # assume 1d
            # print name, then contents of the array, if the index of an item is in ts[name][Op.WRITE] then highlight it
            print("\t", name, ": ", end="")
            print("\t[", end="")
            for i in range(shape):
                if i in ts[name][Op.WRITE]:
                    print("*", end="")
                elif i in ts[name][Op.READ]:
                    print("%", end="")
                else:
                    print(ts[name]["contents"][i], end="")
                print(", ", end="")
            print("]")

    return arr[n]

print(fib(10))

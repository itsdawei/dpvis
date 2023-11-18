from dp import DPArray, display

# Number of iterations to run Fibonacci.
n = 10


# Fibonacci DP function
def fib(n):
    # Initialize a DPArray instead of an array/list
    arr = DPArray(n)

    # Base cases
    arr[0] = 1
    arr[1] = 1

    # Recurrence
    for i in range(2, n):
        arr[i] = arr[i - 1] + arr[i - 2]
        arr.annotate(f"arr[{i - 1}] + arr[{i - 2}]", idx=i)

    # Return the dp array
    return arr


dp_array = fib(n)

# Visualize.
display(dp_array)

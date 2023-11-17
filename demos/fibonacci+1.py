# Testing example for multiple writes per timestep.
from dp import DPArray, display

# Number of iterations to run Fibonacci.
n = 10


# Fibonacci DP function
def fib(n):
    # Initialize a DPArray instead of an array/list
    arr = DPArray((2, n))

    # Base cases
    arr[0, 0] = 1
    arr[0, 1] = 1

    arr[1, 0] = 1
    arr[1, 1] = 1

    # Recurrence
    for i in range(2, n):
        a = arr[0, i - 1] + arr[0, i - 2]
        b = arr[1, i - 1] + arr[1, i - 2]
        arr[0, i] = a
        arr[1, i] = b

    # Return the dp array
    return arr


dp_array = fib(n)

# Visualize.
display(dp_array)

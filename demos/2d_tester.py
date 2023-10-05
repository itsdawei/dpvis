from dp import DPArray, display

# Number of iterations to run Fibonacci.
n = 10


# Fibonacci DP function
def fib(n):
    # Initialize a DPArray instead of an array/list
    arr = DPArray((2, n))

    # pdb.set_trace()
    # Base cases
    arr[0, 0] = 1
    arr[1, 0] = 1
    arr[0, 1] = 1
    arr[1, 1] = 1

    # Recurrence
    for i in range(2, n):
        # If we want one timestep, we have to do the reads consecutively:
        val1 = arr[0, i - 1] + arr[0, i - 2]
        val2 = arr[1, i - 1] + arr[1, i - 2]
        arr[0, i] = val1
        arr[1, i] = val2
    # Return the entire array
    return arr


# dp_array is the fully filled out DPArray after running Fibonacci
dp_array = fib(n)

# Display the dp_array with maximum number of timesteps set to n
display(dp_array)

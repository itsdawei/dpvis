from dp import DPArray, display

# Number of iterations to run Fibonacci.
n = 10


# Fibonacci DP function
def fib(n):
    # Initialize a DPArray instead of an array/list
    arr = DPArray(n, column_labels=[f'fib {i + 1}' for i in range(n)])

    # Base cases
    arr[0] = 1
    arr[1] = 1

    # Recurrence
    for i in range(2, n):
        arr[i] = arr[i - 1] + arr[i - 2]

    # Return the entire array
    return arr


# dp_array is the fully filled out DPArray after running Fibonacci
dp_array = fib(n)

# Display the dp_array with maximum number of timesteps set to n
display(dp_array)

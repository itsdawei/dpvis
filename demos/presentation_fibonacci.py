from dp import DPArray, display

# Number of iterations to run Fibonacci.
n = 10

# Initialize a DPArray.
arr = DPArray(n)

arr[0] = 1
arr[1] = 1

for i in range(2, n):
    arr[i] = arr[i - 1] + arr[i - 2]

display(arr)

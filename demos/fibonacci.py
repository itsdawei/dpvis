from dp import DPArray, display

arr = DPArray(n, array_name="Fibonacci Array")
arr[0] = 1
arr[1] = 1
for i in range(2, n):
    arr[i] = arr[i-1]+arr[i-2]

display(dp_array, description="`OPT(n) = OPT(n-1) + OPT(n-2)`")

from dp import DPArray, _display_dp

n = 11

def fib(n):
    arr = DPArray(n)
    arr[0] = 1
    arr[1] = 1
    for i in range(2, n):
        arr[i] = arr[i-1] + arr[i-2]
    return arr

dp_array = fib(n)

import pdb; pdb.set_trace()
_display_dp(dp_array, n, start=2)
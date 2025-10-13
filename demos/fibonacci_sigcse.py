# import numpy as np
from dp import DPArray, display

def fib(n):
    arr = DPArray(n)
    arr[0] = 1
    arr[1] = 1

    for i in range(2, n):
        arr[i] = arr[i-2] + arr[i-1]
    
    print(arr[n-1])
    display(arr)
    

fib(10)
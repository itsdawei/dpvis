# Dynvis Quickstart

### The Basics

The main object in the dynvis library is [`DPArray`](). A [`DPArray`]() is a multidimensional array of fixed size, whose elements are floats. When [`DPArray`]() is used to facilitate a dynmaic programming algorithm, it records information about the execution of the algorithm that can be viewed later in an interactive display.

In this quickstart guide we use three operations in the [`DPArray`]() class:

1. Constructor
    
    In this guide we only use the shape parameter in the [`DPArray`]() constructor. Similarly to [`numpy.ndarray`](), the shape field is a tuple of integers. A [`DPArray`]() with *n* rows and *m* columns will have shape `(n,m)`.
2. Writing
    
    Elements of the [`DPArray`]() can be changed using brakets, similarly to [`numpy.ndarray`]() or a `list`. For instance, we can assign value `x` to the `i`th element of a one-dimensional [`DPArray`]() named `arr` using the code `arr[i] = 2`.
3. Reading
    
    Elements of the [`DPArray`]() can be retreived using brackets, similarly to [`numpy.ndarray`]() or a `list`. If we wanted to retrieve the `i`th element of the one-dimenional [`DPArray`]() named `arr` and store in the variable `x`, we use the code `x = arr[i]`.

### The Fibonacci Sequence with Dynamic Programming

This guide illustrates the basic usages of [`DPArray`]() in the context of generating the first `n` values of the Fibonacci sequence using dynamic programming. The Fibonacci sequence is defined recursively. The first two elements `arr(0)` and `arr(1)` are `0` and `1`, respectively. The `i`th element of the Fibonacci sequence is defined as `arr(i) = arr(i - 1) + arr(i - 2)`.

The following function returns a [`DPArray`]() of shape `n` that contains the fisrt `n` numbers in the Fibonacci sequence. Note that we will use the `display` import later to visualize this dynamic programming algorithm.

        # Import the DPArray class.
        from dp import DPArray, display

        # Fibonacci dynamic programming function.
        def fib(n):
            # Initialize a one-dimensional DPArray with n elements.
            arr = DPArray(n)

            # Write the first two Fibonacci numbers.
            arr[0] = 0
            arr[1] = 1

            # For each subsequent Fibonacci number
            for i in range(2, n):
                # Use the recurrance to calculate the ith Fibonacci number
                arr[i] = arr[i - 1] + arr[i - 2]

            # Return the DPArray
            return arr

### Visualizing the Dynamic Programming Algorithm

The DPArray class stores information about any program and can be used to produce an interactive visualization of a dynmic programming algorithm. To visualize how the above algorithm constructs an array containing the first `10` Fibonacci numbers, follow the above code with

        # Use the fib algorithm to find the first 10 Fibonacci numbers.
        dp_array = fib(10)
        # Display the array in an interactive visualization
        display(dp_array)
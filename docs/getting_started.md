# Getting Started

[dpvis](https://github.com/itsdawei/dpvis/) is designed to assist students
studying algorithm design gain a more thorough understanding of the
counter-intuitive yet beautiful technique in theoretical computer science known
as [Dynamic Programming
(DP)](https://en.wikipedia.org/wiki/Dynamic_programming).

Our library turns "standard" Python code into an interactive visualization. For
example, consider a dynamic program that computes the $n$th [Fibonnaci
number](https://en.wikipedia.org/wiki/Fibonacci_sequence). With only three
lines of changes, you can convert the standard python implementation to
leverage the functionalities of dpvis.

=== "dpvis"

    ```python linenums="1" hl_lines="1 3 9"
    from dp import DPArray, display

    arr = DPArray(n) # (1)!
    arr[0] = 1
    arr[1] = 1
    for i in range(2, n):
        arr[i] = arr[i-1]+arr[i-2]

    display(arr) # (2)!

    print(arr[-1]) # prints the last value
    ```

    1. Replaces the standard Python list with a [`DPArray`][dp.DPArray] object.
    2. Shows the visualization with [`display(arr)`][dp.display].

=== "Python"

    ```python linenums="1"
    arr = [0] * n # initialize an length n array
    arr[0] = 1
    arr[1] = 1
    for i in range(2, n):
        arr[i] = arr[i-1] + arr[i-2]

    print(arr[-1]) # prints the last value
    ```

dpvis supports additional features that enhances the visualization.
For a more comprehensive overview, see our [tutorials](examples/index.md).

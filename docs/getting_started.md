# Getting Started

[dpvis]() is designed to assist students studying algorithm design gain
a more thorough understanding of the counter-intuitive yet beautiful technique
in theoretical computer science known as [Dynamic Programming
(DP)](https://en.wikipedia.org/wiki/Dynamic_programming).

In particular, our library turns "standard" Python code into interactive DP
visualizations. For example, consider a dynamic program that computes the
Fibonnaci sequence.

```python linenums="1"
arr = []
arr[0] = 1
arr[1] = 1
for i in range(2, n):
    arr[i] = arr[i-1]+arr[i-2]
```

This is how you will modify it to display the visualizations.

```python linenums="1" hl_lines="1 3 9"
from dp import DPArray, display # (1)!

arr = DPArray(n) # (2)!
arr[0] = 1
arr[1] = 1
for i in range(2, n):
    arr[i] = arr[i-1]+arr[i-2]

display(arr) # (3)!
```

1. Imports the library.
2. Replaces the standard Python list with a [`DPArray`][dp.DPArray] object.
3. Shows the visualization with [`display(arr)`][dp.display].

```python linenums="1"
--8<-- "demos/fibonacci.py"
```

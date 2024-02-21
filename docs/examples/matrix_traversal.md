# Matrix Traversal

Given a matrix $M$ of dimensions $(n + 1, m + 1)$, where $M[x, y]$ is the cost
of traveling to $(x, y)$. An agent begins at index $(0, 0)$, and find
a least-cost path to $(n, m)$. The agent is only able to move South or East
from its current position. 

```python linenums="1"
--8<-- "demos/matrix_traversal.py"
```

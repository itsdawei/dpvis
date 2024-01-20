# Excavation

> This problem is taken from [Shaddin Dughmi](https://viterbi-web.usc.edu/~shaddin/)'s CSCI-270 (Fall 2021).

There are $N$ mining sites, where each site has L layers. Excavating level $l$
of site $i$ returns a value of $v[i][l]$. It takes one month to excavate one
level from one mining site. The layers of the site must be excavated in order,
i.e., you must excavate levels $0,\dots,l-1$ before excavating level $l$. **Find
the maximum value that can be excavated if you have a budget of $M$ months.**

```python linenums="1"
--8<-- "demos/excavation.py"
```

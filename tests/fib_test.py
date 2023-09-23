"""Testing dynamic programming implement for fibonacci"""
from dp import DPArray, Op


def fib(n):
    """dynamic programming fibonacci
        fib(0) = 0
        fib(1) = 1
        fib(2) = 1
        ...

    Returns:
        (int, DPArray): the nth fibonacci number and the 
            DPArray used to compute it
    """

    if n <= 1:
        return n, None

    arr = DPArray(n + 1)
    arr[0] = 0
    # print(arr.logger.logs)
    arr[1] = 1

    for i in range(2, n + 1):
        arr[i] = int(arr[i - 1] + arr[i - 2])

    arr.print_timesteps()

    return arr[n], arr


def test_fib():
    assert [fib(i)[0] for i in range(11)
           ] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def test_fib_dparray():
    _, arr = fib(10)

    timesteps = arr.get_timesteps()
    print(timesteps)
    assert len(timesteps) == 11  # 10 steps for computing, 11th read for return
    assert timesteps[0]["dp_array"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()
    for i in range(1, 10):
        assert timesteps[i]["dp_array"].items() >= {
            Op.READ: {i, i - 1},
            Op.WRITE: {i + 1},
            Op.HIGHLIGHT: set(),
        }.items()

"""Tests the interaction between DPArray and Logger."""
import numpy as np
import pytest

from dp import DPArray, Op


def test_duplicate_array_error():
    dp1 = DPArray(10, "duplicate_name")
    with pytest.raises(ValueError):
        _ = DPArray(10, "duplicate_name", logger=dp1.logger)


def test_read_write():
    dp = DPArray(10, "dp")

    dp[0] = 1
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0: 1}}}
    assert len(dp.logger.logs) == 1

    dp[1] = 2
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0: 1, 1: 2}}}

    temp = dp[1]
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp": {0: 1, 1: 2}}}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"dp": {1: None}}}
    assert len(dp.logger.logs) == 2

    dp[2] = temp
    assert dp.logger.logs[2] == {"op": Op.WRITE, "idx": {"dp": {2: 2}}}
    assert len(dp.logger.logs) == 3


def test_2d_read_write():
    dp = DPArray((10, 10), "name")

    dp[0, 0] = 1
    assert len(dp.logger.logs) == 1

    temp = dp[0, 0]
    assert len(dp.logger.logs) == 2

    dp[3, 6] = temp
    assert len(dp.logger.logs) == 3
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"name": {(0, 0): 1}}}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"name": {(0, 0): None}}}
    assert dp.logger.logs[2] == {"op": Op.WRITE, "idx": {"name": {(3, 6): 1}}}


def test_max_highlight():
    dp = DPArray(5, "name")
    dp[0] = 1
    dp[1] = 3
    dp[2] = 0
    assert dp.logger.logs[0].items() >= {
        "op": Op.WRITE,
        "idx": {
            "name": {
                0: 1,
                1: 3,
                2: 0
            }
        }
    }.items()

    indices = [0, 1, 2]
    # BUG: Indexing with a list of indices only logs the first read.
    # elements = dp[indices]
    elements = [dp[i] for i in indices]
    dp[3] = dp.max(indices, elements)
    assert dp.arr[3] == 3
    assert dp.logger.logs[1].items() >= {
        "op": Op.READ,
        "idx": {
            "name": {
                0: None,
                1: None,
                2: None
            }
        }
    }.items()
    assert dp.logger.logs[2].items() >= {
        "op": Op.HIGHLIGHT,
        "idx": {
            "name": {
                1: None
            }
        }
    }.items()
    assert dp.logger.logs[3].items() >= {
        "op": Op.WRITE,
        "idx": {
            "name": {
                3: 3
            }
        }
    }.items()

    indices = [0, 1, 2, 3]
    elements = [-(dp[i] - 1)**2 for i in indices]
    dp[4] = dp.max(indices, elements)
    assert dp.arr[4] == 0
    assert dp.logger.logs[4].items() >= {
        "op": Op.READ,
        "idx": {
            "name": {
                0: None,
                1: None,
                2: None,
                3: None
            }
        }
    }.items()
    assert dp.logger.logs[5].items() >= {
        "op": Op.HIGHLIGHT,
        "idx": {
            "name": {
                0: None
            }
        }
    }.items()
    assert dp.logger.logs[6].items() >= {
        "op": Op.WRITE,
        "idx": {
            "name": {
                4: 0
            }
        }
    }.items()


def test_min():
    """
    Test logger entries when using the DPArray min function.

    Setting:
    A city wants to place fire hydrants on a street.
    In front of each house, they can choose to build a fire hydrant.
    If the city builds a fire hydrant in front of house i, they incur
    a cost of c[i] due to construction costs. Law states that every house
    on the street must have a fire hydrant or be adjacent to a house with
    a fire hydrant. Find an optimal placement of fire hydrants so that the
    city spends as little as possible and the above law is satisfied.
    """
    c = [7, 6, 2, 9, 8, 10, 1, 3]
    highlight_ans = [None, None, None, 1, 2, 2, 4, [4, 5]]
    val_ans = [None, None, None, 8, 14, 14, 15, 15]
    dp = DPArray(8, "name")

    dp[0] = c[0]
    assert dp.logger.logs[0].items() >= {
        "op": Op.WRITE,
        "idx": {
            "name": {
                0: 7
            }
        }
    }.items()

    # Comparing dp[0] with a constant.
    dp[1] = dp.min([0, None], [dp[0], c[1]])
    assert dp.logger.logs[1].items() >= {
        "op": Op.READ,
        "idx": {
            "name": {
                0: None
            }
        }
    }.items()
    assert dp.logger.logs[2].items() >= {
        "op": Op.HIGHLIGHT,
        "idx": {
            "name": {
                None: None
            }
        }
    }.items()
    assert dp.logger.logs[3].items() >= {
        "op": Op.WRITE,
        "idx": {
            "name": {
                1: 6
            }
        }
    }.items()

    dp[2] = dp.min([0, 1], [dp[0] + c[2], dp[1]])
    assert dp.logger.logs[4].items() >= {
        "op": Op.READ,
        "idx": {
            "name": {
                0: None,
                1: None
            }
        }
    }.items()
    assert dp.logger.logs[5].items() >= {
        "op": Op.HIGHLIGHT,
        "idx": {
            "name": {
                1: None
            }
        }
    }.items()
    assert dp.logger.logs[6].items() >= {
        "op": Op.WRITE,
        "idx": {
            "name": {
                2: 6
            }
        }
    }.items()

    next_log = 7
    for i in range(3, 8):
        # Three options
        # Hydrant at i and then satisfy law for i - 2
        # Hydrant at i - 1 and satisfy law for i - 2
        # Hydrant at i - 1 and satisfy law for i - 3
        dp[i] = dp.min(
            [i - 2, i - 2, i - 3],
            [dp[i - 2] + c[i], dp[i - 2] + c[i - 1], dp[i - 3] + c[i - 1]])

        assert dp.logger.logs[next_log].items() >= {
            "op": Op.READ,
            "idx": {
                "name": {
                    i - 2: None,
                    i - 3: None
                }
            }
        }.items()

        # Construct argmin set
        if isinstance(highlight_ans[i], list):
            name = {j: None for j in highlight_ans[i]}
        else:
            name = {highlight_ans[i]: None}
        assert dp.logger.logs[next_log + 1].items() >= {
            "op": Op.HIGHLIGHT,
            "idx": {
                "name": name
            }
        }.items()
        assert dp.logger.logs[next_log + 2].items() >= {
            "op": Op.WRITE,
            "idx": {
                "name": {
                    i: val_ans[i]
                }
            }
        }.items()
        assert dp.arr[i] == val_ans[i]
        next_log += 3


def test_multiple_arrays_logging():
    dp1 = DPArray(10, "dp_1")
    dp2 = DPArray(10, "dp_2", logger=dp1.logger)

    dp1[0] = 1
    dp2[0] = 2
    assert dp1.logger.logs[0].items() >= {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {
                0: 1
            },
            "dp_2": {
                0: 2
            }
        }
    }.items()

    dp1[1] = 3
    dp2[1] = dp1[1]  # READ happens before WRITE
    assert dp1.logger.logs[0].items() >= {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {
                0: 1,
                1: 3
            },
            "dp_2": {
                0: 2
            }
        }
    }.items()
    assert dp1.logger.logs[1].items() >= {
        "op": Op.READ,
        "idx": {
            "dp_1": {
                1: None
            },
            "dp_2": {}
        }
    }.items()
    assert dp1.logger.logs[2].items() >= {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {},
            "dp_2": {
                1: 3
            }
        }
    }.items()
    assert len(dp1.logger.logs) == 3


@pytest.mark.parametrize("op", [Op.WRITE, Op.READ], ids=["w", "r"])
def test_same_op_and_index(op):
    """Same operation with same index does not create additional log.

    Highlight not tested since highlight operations typically require
    read operations before, which would split the logs into different 
    operation groups. 
    """
    dp = DPArray(10, "dp")

    if op == Op.WRITE:
        dp[0] = 1
        dp[0] = 2
    elif op == Op.READ:
        dp[0] = 1
        _ = dp[0]
        _ = dp[0]
    assert dp.logger.logs[0 if op == Op.WRITE else 1].items() >= {
        "op": op,
        "idx": {
            "dp": {
                0: 2 if op == Op.WRITE else None
            }
        }
    }.items()
    assert len(dp.logger.logs) == 1 if op == Op.WRITE else 2


@pytest.mark.parametrize("s", [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
def test_slice_reading(s):
    dp = DPArray(10)

    for i in range(10):
        dp[i] = i**2

    _ = dp[s]
    if isinstance(s, int):
        s = np.s_[s:s + 1]
    truth = {i: None for i in range(*s.indices(10))}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"dp_array": truth}}


def test_slice_reading_list_of_indices():
    dp = DPArray(10)
    for i in range(10):
        dp[i] = i**2

    indices = [1, 2, 3]
    _ = dp[indices]
    truth = {i: None for i in indices}
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"dp_array": truth}}


@pytest.mark.parametrize("s", [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
def test_slice_logging(s):
    dp = DPArray(10)
    dp[s] = 1
    if isinstance(s, int):
        s = np.s_[s:s + 1]
    truth = {i: 1 for i in range(*s.indices(10))}
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp_array": truth}}
    assert len(dp.logger.logs) == 1


@pytest.mark.parametrize("slice_1",
                         [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 5],
                         ids=["a", "b", "c", "d", "e"])
@pytest.mark.parametrize("slice_2",
                         [np.s_[::2], np.s_[:2], np.s_[4:], np.s_[:6], 1],
                         ids=["a", "b", "c", "d", "e"])
def test_2d_slice_logging(slice_1, slice_2):
    dp = DPArray((10, 10))

    dp[slice_1, slice_2] = 1
    if isinstance(slice_1, int):
        slice_1 = np.s_[slice_1:slice_1 + 1]
    if isinstance(slice_2, int):
        slice_2 = np.s_[slice_2:slice_2 + 1]
    truth = {
        (i, j): 1 for i in range(*slice_1.indices(10))
        for j in range(*slice_2.indices(10))
    }
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"dp_array": truth}}
    assert len(dp.logger.logs) == 1


def test_list_assignment():
    dp = DPArray(10)

    dp[::2] = [1, 1, 1, 1, 1]

    assert dp.logger.logs[0] == {
        "op": Op.WRITE,
        "idx": {
            "dp_array": {
                0: 1,
                2: 1,
                4: 1,
                6: 1,
                8: 1,
            }
        }
    }


def test_get_timesteps_one_array():
    dp = DPArray(3, "dp")
    dp[0] = 1
    dp[1] = 2

    timesteps = dp.get_timesteps()
    assert len(timesteps) == 1
    assert np.all(timesteps[0]["dp"]["contents"] == [1, 2, None])
    assert timesteps[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()

    dp[1] = 3
    timesteps2 = dp.get_timesteps()
    assert len(timesteps2) == 1
    assert np.all(timesteps2[0]["dp"]["contents"] == [1, 3, None])
    assert timesteps2[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()

    _ = dp[1]
    timesteps3 = dp.get_timesteps()
    assert len(timesteps3) == 2
    assert np.all(timesteps3[1]["dp"]["contents"] == [1, 3, None])
    assert timesteps3[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

    dp[1] = 3
    timesteps4 = dp.get_timesteps()
    assert len(timesteps4) == 2
    assert np.all(timesteps4[1]["dp"]["contents"] == [1, 3, None])
    assert timesteps4[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: {1},
        Op.HIGHLIGHT: set(),
    }.items()

    dp[0] = dp[1]
    timesteps5 = dp.get_timesteps()
    assert len(timesteps5) == 3
    assert np.all(timesteps5[2]["dp"]["contents"] == [3, 3, None])
    assert timesteps5[2]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: {0},
        Op.HIGHLIGHT: set(),
    }.items()


def test_get_timesteps_two_arrays():
    dp = DPArray(3, "dp")
    dp2 = DPArray(3, "dp2", logger=dp.logger)

    dp[0] = 1
    dp[1] = 2
    dp2[0] = 2
    timesteps = dp.get_timesteps()
    assert len(timesteps) == 1
    assert np.all(timesteps[0]["dp"]["contents"] == [1, 2, None])
    assert np.all(timesteps[0]["dp2"]["contents"] == [2, None, None])
    assert timesteps[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1},
        Op.HIGHLIGHT: set(),
    }.items()
    assert timesteps[0]["dp2"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0},
        Op.HIGHLIGHT: set(),
    }.items()

    _ = dp[1]
    timesteps1 = dp.get_timesteps()
    assert len(timesteps1) == 2
    assert np.all(timesteps1[1]["dp"]["contents"] == [1, 2, None])
    assert np.all(timesteps1[1]["dp2"]["contents"] == [2, None, None])
    assert timesteps1[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()
    assert timesteps1[1]["dp2"].items() >= {
        Op.READ: set(),
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

    dp2[2] = 3
    timesteps2 = dp.get_timesteps()
    assert len(timesteps2) == 2
    assert np.all(timesteps2[1]["dp"]["contents"] == [1, 2, None])
    assert np.all(timesteps2[1]["dp2"]["contents"] == [2, None, 3])
    assert timesteps2[1]["dp"].items() >= {
        Op.READ: {1},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()
    assert timesteps2[1]["dp2"].items() >= {
        Op.READ: set(),
        Op.WRITE: {2},
        Op.HIGHLIGHT: set(),
    }.items()


def test_to_timestep_2d():
    dp = DPArray((3, 3), "dp")
    dp[0, 0] = 1
    dp[1, 1] = 2

    timesteps = dp.get_timesteps()
    assert len(timesteps) == 1
    assert np.all(timesteps[0]["dp"]["contents"] ==
                  [[1, None, None], [None, 2, None], [None, None, None]])
    assert timesteps[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {(0, 0), (1, 1)},
        Op.HIGHLIGHT: set(),
    }.items()

    _ = dp[1, 1]
    timesteps1 = dp.get_timesteps()
    assert len(timesteps1) == 2
    print(dp.arr)
    print(timesteps1[1]["dp"]["contents"])
    assert np.all(timesteps1[1]["dp"]["contents"] ==
                  [[1, None, None], [None, 2, None], [None, None, None]])
    assert timesteps1[1]["dp"].items() >= {
        Op.READ: {(1, 1)},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()


def test_annotation():
    dp = DPArray(10, "dp")
    dp[0] = 1
    dp[1] = 2
    dp[2] = 3
    dp.annotate("hello world")

    timesteps0 = dp.get_timesteps()
    assert len(timesteps0) == 1
    assert timesteps0[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1, 2},
        Op.HIGHLIGHT: set(),
        "annotations": "hello world",
    }.items()

    dp.annotate("hello world again")
    dp.annotate("bye world")
    timesteps1 = dp.get_timesteps()
    assert len(timesteps1) == 1
    assert timesteps1[0]["dp"].items() >= {
        Op.READ: set(),
        Op.WRITE: {0, 1, 2},
        Op.HIGHLIGHT: set(),
        "annotations": "bye world",
    }.items()

    _ = dp[0]
    dp[6] = 10
    dp.annotate("hello cell", idx=6)
    timesteps2 = dp.get_timesteps()
    assert len(timesteps2) == 2
    assert timesteps2[1]["dp"].items() >= {
        Op.READ: {0},
        Op.WRITE: {6},
        Op.HIGHLIGHT: set(),
        "cell_annotations": {
            6: "hello cell"
        }
    }.items()

    dp.annotate("hello cell again", idx=6)
    dp.annotate("hello cell 0", idx=0)
    timesteps3 = dp.get_timesteps()
    assert len(timesteps3) == 2
    assert timesteps3[1]["dp"].items() >= {
        Op.READ: {0},
        Op.WRITE: {6},
        Op.HIGHLIGHT: set(),
        "cell_annotations": {
            6: "hello cell again",
            0: "hello cell 0"
        }
    }.items()

    dp.annotate("some annotation")
    timesteps4 = dp.get_timesteps()
    assert len(timesteps4) == 2
    assert timesteps4[1]["dp"].items() >= {
        Op.READ: {0},
        Op.WRITE: {6},
        Op.HIGHLIGHT: set(),
        "annotations": "some annotation",
        "cell_annotations": {
            6: "hello cell again",
            0: "hello cell 0"
        }
    }.items()

    _ = dp[0]
    timesteps5 = dp.get_timesteps()
    assert len(timesteps5) == 3
    assert timesteps5[2]["dp"].items() >= {
        Op.READ: {0},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

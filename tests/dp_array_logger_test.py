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
    assert dp.logger.logs[0] == {
        "op": Op.WRITE,
        "idx": {
            "name": {
                0: 1,
                1: 3,
                2: 0
            }
        }
    }

    indices = [0, 1, 2]
    # BUG: Indexing with a list of indices only logs the first read.
    # elements = dp[indices]
    elements = [dp[i] for i in indices]
    dp[3] = dp.max(indices, elements)
    assert dp.arr[3] == 3
    assert dp.logger.logs[1] == {
        "op": Op.READ,
        "idx": {
            "name": {
                0: None,
                1: None,
                2: None
            }
        }
    }
    assert dp.logger.logs[2] == {"op": Op.HIGHLIGHT, "idx": {"name": {1: None}}}
    assert dp.logger.logs[3] == {"op": Op.WRITE, "idx": {"name": {3: 3}}}

    indices = [0, 1, 2, 3]
    elements = [-(dp[i] - 1)**2 for i in indices]
    dp[4] = dp.max(indices, elements)
    assert dp.arr[4] == 0
    assert dp.logger.logs[4] == {
        "op": Op.READ,
        "idx": {
            "name": {
                0: None,
                1: None,
                2: None,
                3: None
            }
        }
    }
    assert dp.logger.logs[5] == {"op": Op.HIGHLIGHT, "idx": {"name": {0: None}}}
    assert dp.logger.logs[6] == {"op": Op.WRITE, "idx": {"name": {4: 0}}}


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
    assert dp.logger.logs[0] == {"op": Op.WRITE, "idx": {"name": {0: 7}}}

    # Comparing dp[0] with a constant.
    dp[1] = dp.min([0, None], [dp[0], c[1]])
    assert dp.logger.logs[1] == {"op": Op.READ, "idx": {"name": {0: None}}}
    assert dp.logger.logs[2] != {
        "op": Op.HIGHLIGHT,
        "idx": {
            "name": {
                None: None
            }
        }
    }
    assert dp.logger.logs[2] == {"op": Op.WRITE, "idx": {"name": {1: 6}}}

    dp[2] = dp.min([0, 1], [dp[0] + c[2], dp[1]])
    assert dp.logger.logs[3] == {
        "op": Op.READ,
        "idx": {
            "name": {
                0: None,
                1: None
            }
        }
    }
    assert dp.logger.logs[4] == {"op": Op.HIGHLIGHT, "idx": {"name": {1: None}}}
    assert dp.logger.logs[5] == {"op": Op.WRITE, "idx": {"name": {2: 6}}}

    next_log = 6
    for i in range(3, 8):
        # Three options
        # Hydrant at i and then satisfy law for i - 2
        # Hydrant at i - 1 and satisfy law for i - 2
        # Hydrant at i - 1 and satisfy law for i - 3
        dp[i] = dp.min(
            [i - 2, i - 2, i - 3],
            [dp[i - 2] + c[i], dp[i - 2] + c[i - 1], dp[i - 3] + c[i - 1]])

        assert dp.logger.logs[next_log] == {
            "op": Op.READ,
            "idx": {
                "name": {
                    i - 2: None,
                    i - 3: None
                }
            }
        }

        # Construct argmin set
        if isinstance(highlight_ans[i], list):
            name = {j: None for j in highlight_ans[i]}
        else:
            name = {highlight_ans[i]: None}
        assert dp.logger.logs[next_log + 1] == {
            "op": Op.HIGHLIGHT,
            "idx": {
                "name": name
            }
        }
        assert dp.logger.logs[next_log + 2] == {
            "op": Op.WRITE,
            "idx": {
                "name": {
                    i: val_ans[i]
                }
            }
        }
        assert dp.arr[i] == val_ans[i]
        next_log += 3


def test_min_select_indice_and_constant():
    dp = DPArray(2, "dp")
    dp[0] = 0
    dp[1] = 1

    # Basic check.
    min_val = dp.min([0, 1], constants=[2])
    assert min_val == 0

    min_val = dp.min([0, 1], constants=[-1])
    assert min_val == -1
    
    dp[1] = 2
    min_val = dp.min([0, 1], constants=[0, 1])
    assert min_val == 0
    timestep = dp.get_timesteps()[1]
    assert timestep["dp"][Op.READ] == {0, 1}
    assert timestep["dp"][Op.HIGHLIGHT] == {0}
    assert None not in timestep["dp"][Op.HIGHLIGHT]

def test_max_select_indice_and_constant():
    dp = DPArray(2, "dp")
    dp[0] = 0
    dp[1] = 1

    # Basic check.
    max_val0 = dp.max([0, 1], constants=[2])
    assert max_val0 == 2

    max_val1 = dp.max([0, 1], constants=[-1])
    assert max_val1 == 1
    
    dp[1] = 2
    max_val2 = dp.max([0, 1], constants=[0, 1])
    assert max_val2 == 2
    timestep = dp.get_timesteps()[2]
    assert timestep["dp"][Op.READ] == {0, 1}
    assert timestep["dp"][Op.HIGHLIGHT] == {1}
    assert None not in timestep["dp"][Op.HIGHLIGHT]

def test_min_select_elements_and_constant():
    dp = DPArray(2, "dp")
    dp[0] = 0
    dp[1] = 1

    min_val0 = dp.min([0, 1], elements=[dp[0]+1, dp[1]+1])
    assert min_val0 == 1
    timestep0 = dp.get_timesteps()[1]
    assert timestep0["dp"][Op.READ] == {0, 1}
    assert timestep0["dp"][Op.HIGHLIGHT] == {0}

    # Make a fresh timestep.
    dp[0] = 0
    min_val1 = dp.min([0, 1], elements=[2, 3])
    assert min_val1 == 2
    timestep1 = dp.get_timesteps()[2]
    assert timestep1["dp"][Op.READ] == set()
    assert timestep1["dp"][Op.HIGHLIGHT] == {0}

    dp[0] = 0
    min_val2 = dp.min([0, 1],
                        elements=[dp[0]+3, dp[1]-1],
                        constants=[2])
    assert min_val2 == 0
    timestep2 = dp.get_timesteps()[3]
    assert timestep2["dp"][Op.READ] == {0, 1}
    assert timestep2["dp"][Op.HIGHLIGHT] == {1}

    dp[0] = 0
    min_val3 = dp.min([0, 1],
                        elements=[dp[0]+3, dp[1]-1],
                        constants=[-1])
    assert min_val3 == -1
    timestep3 = dp.get_timesteps()[4]
    assert timestep3["dp"][Op.READ] == {0, 1}
    assert timestep3["dp"][Op.HIGHLIGHT] == set()


def test_max_select_elements_and_constant():
    dp = DPArray(2, "dp")
    dp[0] = 0
    dp[1] = 1

    max_val0 = dp.max([0, 1], elements=[dp[0]+1, dp[1]+1])
    assert max_val0 == 2
    timestep0 = dp.get_timesteps()[1]
    assert timestep0["dp"][Op.READ] == {0, 1}
    assert timestep0["dp"][Op.HIGHLIGHT] == {1}

    # Make a fresh timestep.
    dp[0] = 0
    max_val1 = dp.max([0, 1], elements=[2, 3])
    assert max_val1 == 3
    timestep1 = dp.get_timesteps()[2]
    assert timestep1["dp"][Op.READ] == set()
    assert timestep1["dp"][Op.HIGHLIGHT] == {1}

    dp[0] = 0
    max_val2 = dp.max([0, 1],
                        elements=[dp[0]+3, dp[1]-1],
                        constants=[10])
    assert max_val2 == 10
    timestep2 = dp.get_timesteps()[3]
    assert timestep2["dp"][Op.READ] == {0, 1}
    assert timestep2["dp"][Op.HIGHLIGHT] == set()

    dp[0] = 0
    max_val3 = dp.max([0, 1],
                        elements=[dp[0]+3, dp[1]-1],
                        constants=[-1])
    assert max_val3 == 3
    timestep3 = dp.get_timesteps()[4]
    assert timestep3["dp"][Op.READ] == {0, 1}
    assert timestep3["dp"][Op.HIGHLIGHT] == {0}


def test_multiple_arrays_logging():
    dp1 = DPArray(10, "dp_1")
    dp2 = DPArray(10, "dp_2", logger=dp1.logger)

    dp1[0] = 1
    dp2[0] = 2
    assert dp1.logger.logs[0] == {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {
                0: 1
            },
            "dp_2": {
                0: 2
            }
        }
    }

    dp1[1] = 3
    dp2[1] = dp1[1]  # READ happens before WRITE
    assert dp1.logger.logs[0] == {
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
    }
    assert dp1.logger.logs[1] == {
        "op": Op.READ,
        "idx": {
            "dp_1": {
                1: None
            },
            "dp_2": {}
        }
    }
    assert dp1.logger.logs[2] == {
        "op": Op.WRITE,
        "idx": {
            "dp_1": {},
            "dp_2": {
                1: 3
            }
        }
    }
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
    assert dp.logger.logs[0 if op == Op.WRITE else 1] == {
        "op": op,
        "idx": {
            "dp": {
                0: 2 if op == Op.WRITE else None
            }
        }
    }
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
    assert np.all(timesteps1[1]["dp"]["contents"] ==
                  [[1, None, None], [None, 2, None], [None, None, None]])
    assert timesteps1[1]["dp"].items() >= {
        Op.READ: {(1, 1)},
        Op.WRITE: set(),
        Op.HIGHLIGHT: set(),
    }.items()

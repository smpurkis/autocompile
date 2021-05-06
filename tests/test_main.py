import random
import string
from time import time

import numba
import numpy as np

from autocompile import *


def timeit(func, *args, **kwargs):
    @wraps(func)
    def benchmark():
        try:
            func_name = func.__name__  # will fail on cython compile functions
        except Exception as e:
            func_name = "func_cy"
        s = time()
        out = func(*args, **kwargs)
        print(f"\n{func_name} took: {round(time() - s, 3)} seconds")
        return out

    return benchmark()


def test_mixed_maths():
    # define test variables
    n0 = 1
    n = 10

    # warm up/compile functions
    maths_py(n0)
    maths_nb(n0)
    maths_cy(n0)
    maths_ac(n0)

    timeit(maths_py, n)
    timeit(maths_nb, n)
    timeit(maths_cy, n)
    timeit(maths_ac, n)


def test_list_type():
    # define test variables
    n0 = 1
    n = 1_000

    # warm up/compile functions
    lists_py(n0)
    lists_nb(n0)
    lists_cy(n0)
    lists_ac(n0)

    timeit(lists_py, n)
    timeit(lists_nb, n)
    timeit(lists_cy, n)
    timeit(lists_ac, n)


def test_mixed_types():
    # define test variables
    n0 = 1
    n = 100

    # warm up/compile functions
    mixed_py(n0)
    mixed_nb(n0)
    mixed_cy(n0)
    mixed_ac(n0)

    timeit(mixed_py, n)
    timeit(mixed_nb, n)
    timeit(mixed_cy, n)
    timeit(mixed_ac, n)


def test_np_arr_in_body():
    # define test variables
    n0 = 2
    n = 1000

    # warm up/compile functions
    np_array_in_body_py(n0)
    np_array_in_body_nb(n0)
    np_array_in_body_cy(n0)
    np_array_in_body_ac(n0)

    timeit(np_array_in_body_py, n)
    timeit(np_array_in_body_nb, n)
    timeit(np_array_in_body_cy, n)
    timeit(np_array_in_body_ac, n)


def test_np_arr_in_args():
    # define test variables
    n0 = np.random.rand(10, 10)
    n = np.random.rand(1000, 1000)

    # warm up/compile functions
    np_array_in_args_py(n0)
    np_array_in_args_nb(n0)
    np_array_in_args_np(n0)
    np_array_in_args_ac(n0)

    timeit(np_array_in_args_py, n)
    timeit(np_array_in_args_nb, n)
    timeit(np_array_in_args_np, n)
    timeit(np_array_in_args_ac, n)


def test_strings():
    # define test variables
    n0 = 10
    n = 100000

    # warm up/compile functions
    string_py(n0)
    string_nb(n0)
    string_cy(n0)
    string_ac(n0)

    timeit(string_py, n)
    timeit(string_nb, n)
    timeit(string_cy, n)
    timeit(string_ac, n)


def test_docstrings_and_comments():
    docstring_comments(1)


@autocompile
def docstring_comments(x: int):
    """

    Args:
        m: test (default: 4)

    Returns: wfwef = wefwfe {32r: 2r32}
    """
    i: int
    # hello world : can do
    for i in range(1):
        return


@autocompile
def mixed_ac(m: int):
    l: list
    x: dict
    t: float
    i: int
    j: int

    l = []
    for p in range(m):
        x = {}
        t = 0
        for i in range(m):
            for j in range(m):
                t += (i + m) ** 0.1
            x[i] = str(t)
        l.append(x)
    return l


@cython.compile
def mixed_cy(m):
    l = []
    for p in range(m):
        x = {}
        t = 0
        for i in range(m):
            for j in range(m):
                t += (i + m) ** 0.1
            x[i] = str(t)
        l.append(x)
    return l


@numba.jit(forceobj=True)
def mixed_nb(m):
    l = list()
    for p in range(m):
        x = dict()
        t = 0
        for i in range(m):
            for j in range(m):
                t += (i + m) ** 0.1
            x[i] = str(t)
        l.append(x)
    return l


def mixed_py(m):
    l = []
    for p in range(m):
        x = {}
        t = 0
        for i in range(m):
            for j in range(m):
                t += (i + m) ** 0.1
            x[i] = str(t)
        l.append(x)
    return l


@autocompile
def lists_ac(m: int):
    i: int
    j: int
    x: list
    y: list
    x = []
    for i in range(m):
        y = []
        for j in range(m):
            y.append(j)
        x.append(y)
    return x


@cython.compile
def lists_cy(m):
    x = []
    for i in range(m):
        y = []
        for j in range(m):
            y.append(j)
        x.append(y)
    return x


@numba.njit
def lists_nb(m):
    x = []
    for i in range(m):
        y = []
        for j in range(m):
            y.append(j)
        x.append(y)
    return x


def lists_py(m):
    x = []
    for i in range(m):
        y = []
        for j in range(m):
            y.append(j)
        x.append(y)
    return x


@autocompile
def maths_ac(x: float):
    i: int
    for i in range(10000000):
        x += (i + x) ** 0.1
    return [x]


@cython.compile
def maths_cy(x):
    for i in range(10000000):
        x += (i + x) ** 0.1
    return [x]


@numba.njit
def maths_nb(x):
    for i in range(10000000):
        x += (i + x) ** 0.1
    return [x]


def maths_py(x: float):
    for i in range(10000000):
        x += (i + x) ** 0.1
    return [x]


@autocompile(required_imports=globals())
def np_array_in_body_ac(x: int):
    total: float
    i: int
    j: int
    arr: np.ndarray

    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total


@cython.compile
def np_array_in_body_cy(x):
    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total


@numba.njit
def np_array_in_body_nb(x):
    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total


def np_array_in_body_py(x):
    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total


@autocompile(required_imports=globals())
def np_array_in_args_ac(x: np.ndarray):
    total: float
    i: int
    j: int

    total = 0
    x = x + np.random.rand(x.shape[0], x.shape[1])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            total += x[i][j]
    return total


def np_array_in_args_np(x):
    x = x + np.random.rand(x.shape[0], x.shape[1])
    total = np.sum(x)
    return total


@numba.njit
def np_array_in_args_nb(x):
    total = 0
    x = x + np.random.rand(x.shape[0], x.shape[1])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            total += x[i][j]
    return total


def np_array_in_args_py(x):
    total = 0
    x = x + np.random.rand(x.shape[0], x.shape[1])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            total += x[i][j]
    return total


@autocompile(required_imports=globals())
def string_ac(x: int):
    i: int
    s: str
    letters: str
    l: str

    s = ""
    letters = string.ascii_letters
    for i in range(x):
        l = random.choice(letters)
        s += l
    return s


@cython.compile
def string_cy(x):
    s = ""
    letters = string.ascii_letters
    for i in range(x):
        l = random.choice(letters)
        s += l
    return s


@numba.jit(forceobj=True)
def string_nb(x):
    s = ""
    letters = string.ascii_letters
    for i in range(x):
        l = random.choice(letters)
        s += l
    return s


def string_py(x):
    s = ""
    letters = string.ascii_letters
    for i in range(x):
        l = random.choice(letters)
        s += l
    return s

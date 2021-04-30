from AutoCompile import autocompile

from functools import wraps
from time import time

import cython
import numba
import numpy as np


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
    n = 1_000

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
    n = 3_000

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
    n = 200

    # warm up/compile functions
    mixed_py(n0)
    mixed_nb(n0)
    mixed_cy(n0)
    mixed_ac(n0)

    timeit(mixed_py, n)
    timeit(mixed_nb, n)
    timeit(mixed_cy, n)
    timeit(mixed_ac, n)


def test_np_arr():
    # define test variables
    n0 = 2
    n = 2000

    # warm up/compile functions
    np_array_py(n0)
    np_array_nb(n0)
    np_array_cy(n0)
    x = np_array_ac(n0)
    print(x)
    print(type(x))

    timeit(np_array_py, n)
    timeit(np_array_nb, n)
    timeit(np_array_cy, n)
    timeit(np_array_ac, n)


@autocompile()
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


@numba.jit
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


@autocompile()
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


@autocompile()
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


@autocompile
def np_array_ac(x: int):
    total: float
    i: int
    j: int

    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total


@cython.compile
def np_array_cy(x):
    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total


@numba.njit
def np_array_nb(x):
    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total


def np_array_py(x):
    arr = np.random.random((x, x))
    arr = arr.reshape(-1, *arr.shape[:-1]) * arr
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total += arr[i][j]
    return total

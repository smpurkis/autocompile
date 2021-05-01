# AutoCompile

TLDR; Speed up Python code that is marked with type hints (by converting it to Cython)

This is a package born slightly out of surprise when I found out that type hints don't 
speed up Python code at all, when all the information is there to be able to speed it up. 
So I decided to write this short package,  that analyzes the code of any function marked
with `@autocompile` and converts it into a Cython inline function. For example,

```python
def do_maths(x: float):
    i: int
    for i in range(10000000):
        x += (i + x) ** 0.1
    return x
```

will be converted to:

```cython
def maths_ac(double x):
    cdef long i  
    for i in range(10000000):
        x += (i + x) ** 0.1
    return x
```

## Documentation

`@autocompile` has the following arguments:
```
    mode: "inline" or "file", type: str, default: "inline"
        "inline": uses Cython inline as a backend, works with all imported libraries 
        "file": moves code to a tmp file and cythonizes it using subprocess, doesn't work with any imported libraries 
    infer_types: True or False, type: Bool, default: False
        Enable Cython infer type option
    checks_on: True or False, type: Bool, default: False
        Enable Cython boundary and wrapping checking
    required_imports: {} or globals(), type: Dict, default: {}
        This is required for access to the globals of the calling module. As Python in its infinite wisdom doesn't allow
        access without explicitly passing them.
        Example:
            @autocompile(required_imports=globals())
            def foo(bar: int):
                x = np.arange(bar)
                return x
        Without passing globals, Cython inline conversion will error, as it doesn't know what np (numpy) is
```

## Benchmark

Here are a few benchmarks of speed improvements (all code is in `tests` folder):

```
tests/test_main.py::test_mixed_maths 
maths_py took: 1.049 seconds
maths_nb took: 0.299 seconds
func_cy took: 1.595 seconds
maths_ac took: 0.298 seconds
PASSED

tests/test_main.py::test_list_type 
lists_py took: 0.626 seconds
lists_nb took: 0.311 seconds
func_cy took: 0.251 seconds
lists_ac took: 0.29 seconds
PASSED

tests/test_main.py::test_mixed_types
mixed_py took: 0.939 seconds
mixed_nb took: 1.268 seconds (had to force object mode)
func_cy took: 0.748 seconds
mixed_ac took: 0.173 seconds
PASSED

tests/test_main.py::test_np_arr
np_array_py took: 1.185 seconds
np_array_nb took: 0.053 seconds
func_cy took: 1.07 seconds
np_array_ac took: 1.141 seconds
PASSED
```
(note: this is using `cython.compile`, to compare against, as it is the closest function to `autocompile` (`ac`)).

As can be seen, `ac` is best at a mixture of base Python types, lists, dicts, numbers. It offers
no speed up for arrays at the moment.

Potential improvements:
- Add support for return types (relatively straightforward)
- Add support for automatically memory view (would solve array speed up issue)
- Add a backend like Nim or Julia (a lot of work)
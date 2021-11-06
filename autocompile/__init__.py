import copy
import hashlib
import inspect
from functools import wraps
from importlib import import_module
from pathlib import Path
from subprocess import check_call
from typing import get_type_hints

import cython
import numba
import numpy
from Cython.Build.Inline import cython_inline

cython_file_template = """
# cython: infer_types=True
# distutils: extra_compile_args = -Ofast -fopenmp
cimport cython
"""


class AutoCompile:
    """
    Main Class that handles the logic of the formatting the function code
    """

    def __init__(self, mode="inline", infer_types=False, checks_on=True, debug=False, force_memview=False):

        #  Simple 1 to 1 conversion of python types to cython types, aires on the side of caution
        self._type_conversion = {
            int: cython.long,
            "int": cython.long,
            float: cython.double,
            "float": cython.double,
            bool: cython.bint,
            "bool": cython.bint,
            str: "str",
            "str": "str",
            "float64": cython.double,
            "float32": cython.float,
            "int32": cython.int,
            "int64": cython.long
        }

        self.cythonized_functions = {}

        self.cython_file_template = cython_file_template

        self.mode = mode
        if self.mode == "file":
            self.setup_tmp_file()

        self.infer_types = infer_types
        self.checks_on = checks_on
        self.debug = debug
        self.force_memview = force_memview
        self.backend = "cython"

    def setup_tmp_file(self):
        self.BUILD_DIR_STR = "autocompile_tmp"
        self.BUILD_DIR = Path(self.BUILD_DIR_STR)
        self.BUILD_DIR.mkdir(parents=True, exist_ok=True)
        self.tempfile = Path(self.BUILD_DIR, "autocompile.pyx")
        with self.tempfile.open("w") as open_tempfile:
            open_tempfile.write(self.cython_file_template)

    def cythonize_func_file_method(self, func, function_lines, def_line_index, filepath):
        cython_function_code, cython_def_lines = self.format_function_code_to_cython(
            func, function_lines, def_line_index
        )
        with self.tempfile.open("a") as open_tempfile:
            open_tempfile.write(cython_function_code)

        proc_return = check_call(f"cythonize -i3 --inplace {filepath.as_posix()}".split())
        if proc_return != 0:
            raise Exception("Failed to Compile Function")

        autocythonfunctions = import_module(name=f"{self.BUILD_DIR_STR}.{self.tempfile.stem}")
        cythonized_func = autocythonfunctions.__dict__.get(func.__name__)

        return cythonized_func

    def extract_variables_from_definition(self, func, function_lines, def_line_index):
        vars = function_lines[def_line_index].split("(")[1].split(",")
        type_hints = get_type_hints(func)
        func_args = []

        for var in vars:
            cleaned_var = var.replace("):\n", "")
            var_name = cleaned_var.replace(" ", "").split(":")[0].split("=")[0]
            python_type = type_hints.get(var_name, None)
            cython_type = self._type_conversion.get(python_type, None)

            if python_type is None:
                cython_type = ""
            elif "ndarray" in str(python_type):
                self.backend = "numba"
                cython_type = ""
            else:
                if cython_type is None:
                    cython_type = str(python_type).split("'")[1]

            if "=" in var:
                default_value = var.replace(" ", "").split("=")[1]
                if "):\n" in default_value:
                    default_value = default_value[:-3]
                default_value = eval(default_value)
                if isinstance(default_value, str):
                    default_value_str = f'= "{default_value}"'
                    default_value = f'"{default_value}"'
                else:
                    default_value_str = f'= {default_value}'
                if python_type is None:
                    python_type = type(default_value)
                    cython_type = self._type_conversion.get(python_type, "")  # TODO make default clever

            else:
                default_value_str = ""
                default_value = None

            func_args.append({"name": var_name,
                              "python_type": python_type,
                              "cython_type": cython_type,
                              "default_value": default_value,
                              "default_value_str": default_value_str})

        func_return_type = {
            "python_type": None,
            "cython_type": None,
            "str": ""
        }
        if "return" in type_hints.keys():
            python_type = type_hints.get("return")
            cython_type = self._type_conversion.get(python_type, "")
            func_return_type = {
                "python_type": python_type,
                "cython_type": cython_type,
                "str": f" {cython_type}"
            }

        return func_args, func_return_type

    def extract_type_from_input_variables(self, func_args, args, kwargs):
        func_args_minus_kws = copy.copy(func_args)
        for kw in kwargs.keys():
            for var in func_args:
                if var["name"] == kw:
                    func_args_minus_kws.remove(var)

        for i, arg in enumerate(args):
            if func_args_minus_kws[i]["cython_type"] == "":
                func_args_minus_kws[i]["python_type"] = type(arg)
                # if "ndarray" in str(func_args_minus_kws[i]["python_type"]):
                #     np_type = arg.dtype
                #     cython_type = self._type_conversion.get(str(np_type), "")
                #     if cython_type == "":
                #         continue
                #     np_shape_len = len(arg.shape)
                #     func_args_minus_kws[i][
                #         "cython_type"] = f"{cython_type}[{':, '.join(['' for i in range(np_shape_len + 1)])[:-2]}]"
                # else:
                #     func_args_minus_kws[i]["cython_type"] = self._type_conversion.get(
                #         func_args_minus_kws[i]["python_type"], "")  # TODO make default clever
                func_args_minus_kws[i]["cython_type"] = self._type_conversion.get(
                    func_args_minus_kws[i]["python_type"], "")  # TODO make default clever
        return func_args

    def build_cython_function_definition(self, func, func_args, func_return_type):
        vars_string = ""
        for index, var in enumerate(func_args):
            if index == len(func_args) - 1:
                vars_string += f"{var['cython_type']} {var['name']}{'' if var['default_value_str'] == '' else ' ' + var['default_value_str']}"
            else:
                vars_string += f"{var['cython_type']} {var['name']}{'' if var['default_value_str'] == '' else ' ' + var['default_value_str']}, "
        if self.mode == "file":
            cython_def_lines = [f"cpdef{func_return_type['str']} {func.__name__}({vars_string}):\n"]
        else:
            cython_def_lines = [f"def {func.__name__}({vars_string}):\n"]
        return cython_def_lines

    def remove_comments_from_function_lines(self, func, function_lines):
        doc_lines = inspect.getdoc(func) or ""
        doc_lines = [l.lstrip() for l in doc_lines.split("\n")]
        cleaned_function_lines = []
        for line in function_lines:
            formatted_line = line.lstrip().replace("\n", "")
            if formatted_line not in doc_lines:
                if formatted_line[0] not in ["#", '"', "'"]:
                    cleaned_function_lines.append(line)
        return cleaned_function_lines

    def extract_function_body_type_hints(self, func, function_lines, cython_def_lines, def_line_index):
        cleaned_function_lines = self.remove_comments_from_function_lines(func, function_lines)

        remove_lines = []
        for line in cleaned_function_lines:
            if ":" in line and "def" not in line and "\'":
                var_string = ""
                leading_spaces = "".join([" " for i in range(len(line) - len(line.lstrip()))])
                split_line = line.replace(" ", "").replace("\n", "").split(":")
                var_name = split_line[0]
                if "\'" in var_name or '\"' in var_name or "(" in var_name:
                    break
                if "[" in var_name:
                    continue
                elif len(split_line) > 1:
                    if split_line[1] == "":
                        continue
                split_line = split_line[1].split("=")
                python_type = split_line[0]
                cython_type = self._type_conversion.get(python_type, None)
                if "ndarray" in python_type:
                    self.backend = "numba"
                    if self.force_memview:
                        pass
                    else:
                        continue
                if cython_type is None:
                    cython_type = python_type
                if len(split_line) > 1:
                    default_value_str = f"= {split_line[1]}"
                else:
                    default_value_str = ""
                var_string += f"{leading_spaces}cdef {cython_type} {var_name} {default_value_str} \n"
                cython_def_lines.append(var_string)
                remove_lines.append(line)

        if self.checks_on:
            checking_lines = []
        else:
            checking_lines = ["@cython.boundscheck(False)\n",
                              "@cython.wraparound(False)\n"]
        cython_function_lines = checking_lines + cython_def_lines + function_lines[def_line_index + 1:]
        [cython_function_lines.remove(line) for line in remove_lines]
        return cython_function_lines

    def format_function_code_to_cython(self, func, function_lines, def_line_index, hash_only=False, *args, **kwargs):
        """
        Walk through the stages of extracting optimisation information
        e.g. input variable types from arguments and type hints
        """
        func_args, func_return_type = self.extract_variables_from_definition(func, function_lines, def_line_index)
        func_args = self.extract_type_from_input_variables(func_args, args, kwargs)

        cython_def_lines = self.build_cython_function_definition(func, func_args, func_return_type)
        if hash_only:
            return cython_def_lines
        cython_function_lines = self.extract_function_body_type_hints(func, function_lines, cython_def_lines,
                                                                      def_line_index)

        cython_function_code = "".join(cython_function_lines)
        if self.debug:
            python_function_code = "".join(function_lines)

            print("Original Python Code:")
            print(python_function_code)

            print("\nGenerated Cython Code:")
            print(cython_function_code)
        return cython_function_code, cython_def_lines

    def hash_func(self, cython_def_lines):
        return hashlib.md5("".join(cython_def_lines).encode()).hexdigest()

    def cythonize_func_inline_method(self, func, function_lines, def_line_index, hash_only=False, *args, **kwargs):
        """
        Compiles the function if it hasn't been compile before, and stores the function for reuse
        """

        if hash_only:
            cython_def_lines = self.format_function_code_to_cython(
                func, function_lines, def_line_index, hash_only=True, *args, **kwargs
            )
            function_hash_name = self.hash_func(cython_def_lines)
            return function_hash_name
        else:
            cython_function_code, cython_def_lines = self.format_function_code_to_cython(
                func, function_lines, def_line_index, *args, **kwargs
            )
            cythonized_func = cython_inline(cython_function_code, cython_compiler_directives={
                "infer_types": self.infer_types,
            })
            cythonized_func = cythonized_func.get(func.__name__, None)

            if cythonized_func is None:
                python_func = "".join(function_lines)
                cythonized_func = cython.inline(python_func).get(func.__name__, None)
                if cythonized_func is None:
                    cythonized_func = func

            function_hash_name = self.hash_func(cython_def_lines)
            return function_hash_name, cythonized_func


def update_globals(func, required_imports):
    if callable(func):
        libs = __import__(func.__module__)
        for lib_name, lib in libs.__dict__.items():
            if lib_name not in globals():
                globals()[lib_name] = lib
    if len(required_imports) > 0:
        for lib_name, lib in required_imports.items():
            if lib_name not in globals():
                globals()[lib_name] = lib



# TODO add a return type to the inline function
# TODO Check other backends, e.g. Pythran
# TODO Track the functions that are compiled by Numba vs Cython and expose the Cythonized functions
# using http://numba.pydata.org/numba-doc/latest/extending/high-level.html#importing-cython-functions
def autocompile(*ags, **kwgs):
    """
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
        Without passing globals, Cython inline conversion will error, as it doesn't know what np (numpy) is.
    debug: True or False, type: Bool, default: False
        Shows the created function code to be used in place of the original
    force_memview: True or False, type: Bool, default: False (currently disabled)
        Forces all declared numpy arrays to be treated at cython memview. Can be unsafe, as addition of memviews
        in cython is not supported while for numpy arrays it is.
    """
    mode = "inline"
    infer_type = True
    checks_on = True
    required_imports = {}
    debug = False
    force_memview = False
    if "mode" in kwgs:
        mode = kwgs["mode"]
    if "infer_type" in kwgs:
        infer_type = kwgs["infer_type"]
    if "checks_on" in kwgs:
        checks_on = kwgs["checks_on"]
    if "required_imports" in kwgs:
        required_imports = kwgs["required_imports"]
    if "debug" in kwgs:
        debug = kwgs["debug"]
    if "force_memview" in kwgs:
        force_memview = kwgs["force_memview"]

    def _autocompile(func):
        if callable(func):
            libs = __import__(func.__module__)
            for lib_name, lib in libs.__dict__.items():
                if lib_name not in globals():
                    globals()[lib_name] = lib
        if len(required_imports) > 0:
            for lib_name, lib in required_imports.items():
                if lib_name not in globals():
                    globals()[lib_name] = lib
        # update_globals(func=func, required_imports=required_imports)

        ac = AutoCompile(
            mode=mode,
            infer_types=infer_type,
            checks_on=checks_on,
            debug=debug,
            force_memview=force_memview
        )

        numba_njit_success = False
        try:
            numba_func = numba.njit(func)
            ac.cythonized_functions[func] = numba_func
            numba_njit_success = True
        except Exception as e:
            pass

        if ac.mode == "file":
            function_lines = inspect.getsourcelines(func)[0]
            def_line_index = ["def" in line.lstrip()[:3] for line in function_lines].index(True)

            function_lines = inspect.getsourcelines(func)[0]
            cythonized_func = ac.cythonize_func_file_method(
                func=func,
                function_lines=function_lines,
                def_line_index=def_line_index,
                filepath=ac.tempfile
            )
        elif ac.mode == "inline":
            function_lines = inspect.getsourcelines(func)[0]
            def_line_index = ["def" in line.lstrip()[:3] for line in function_lines].index(True)

            function_lines = inspect.getsourcelines(func)[0]
            function_hash_name, cythonized_func = ac.cythonize_func_inline_method(
                func=func,
                function_lines=function_lines,
                def_line_index=def_line_index
            )

        ac.cythonized_functions[func] = numba_func if numba_njit_success and ac.backend == "numba" else cythonized_func

        @wraps(func)
        def run_func(*args, **kwargs):
            cythonized_func = ac.cythonized_functions.get(func, func)
            return cythonized_func(*args, **kwargs)

        return run_func

    if len(ags) > 0:
        if callable(ags[0]):
            func = ags[0]
            return _autocompile(func)
    return _autocompile


if __name__ == '_main_':

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


    print(lists_ac(50))

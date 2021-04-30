import copy
import hashlib
import inspect
from functools import wraps
from importlib import import_module
from pathlib import Path
from subprocess import check_call
from typing import get_type_hints

import cython
from Cython.Build.Inline import cython_inline

cython_file_template = """
# cython: infer_types=True
# distutils: extra_compile_args = -Ofast -fopenmp
cimport cython
import numpy as np
cimport numpy as np

"""


class AutoCompile:
    def __init__(self, mode="inline", infer_types=False):
        self.BUILD_DIR_STR = "autocompile_tmp"
        self.BUILD_DIR = Path(self.BUILD_DIR_STR)
        self.BUILD_DIR.mkdir(parents=True, exist_ok=True)

        self._type_conversion = {
            int: cython.long,
            "int": cython.long,
            float: cython.double,
            "float": cython.double,
            str: cython.char,
            "str": cython.char,
            bool: cython.bint,
            "bool": cython.bint,
        }

        self.cythonized_functions = {}

        self.cython_file_template = cython_file_template

        self.mode = mode
        if self.mode == "file":
            self.setup_tmp_file()

        self.infer_types = infer_types

    def setup_tmp_file(self):
        self.tempfile = Path(self.BUILD_DIR, "autocompile.pyx")
        with self.tempfile.open("w") as open_tempfile:
            open_tempfile.write(self.cython_file_template)

    def cythonize_func_file_method(self, filepath, function_code, func_name):
        with self.tempfile.open("a") as open_tempfile:
            open_tempfile.write(function_code)

        proc_return = check_call(f"cythonize -i3 --inplace --force {filepath.as_posix()}".split())
        if proc_return != 0:
            raise Exception("Failed to Compile Function")

        autocythonfunctions = import_module(name=f"{self.BUILD_DIR_STR}.{self.tempfile.stem}")
        cythonized_func = autocythonfunctions.__dict__.get(func_name)

        return cythonized_func

    def extract_variables_from_definition(self, func, vars):
        type_hints = get_type_hints(func)
        func_args = []

        for var in vars:
            cleaned_var = var.replace("):\n", "")
            var_name = cleaned_var.replace(" ", "").split(":")[0].split("=")[0]
            python_type = type_hints.get(var_name, None)
            cython_type = self._type_conversion.get(python_type, None)

            if python_type is None:
                cython_type = ""
            else:
                if cython_type is None:
                    cython_type = str(python_type).split("'")[1]

            if "=" in var:
                default_value = var.replace(" ", "").split("=")[1]
                if "):\n" in default_value:
                    default_value = default_value[:-3]
                default_value = eval(default_value)
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
        return func_args

    def extract_type_from_input_variables(self, func_args, args, kwargs):
        func_args_minus_kws = copy.copy(func_args)
        for kw in kwargs.keys():
            for var in func_args:
                if var["name"] == kw:
                    func_args_minus_kws.remove(var)

        for i, arg in enumerate(args):
            if func_args_minus_kws[i]["cython_type"] == "":
                func_args_minus_kws[i]["python_type"] = type(arg)
                func_args_minus_kws[i]["cython_type"] = self._type_conversion.get(
                    func_args_minus_kws[i]["python_type"], "")  # TODO make default clever
        return func_args

    def build_cython_function_definition(self, func, func_args):
        vars_string = ""
        for index, var in enumerate(func_args):
            vars_string += f"{var['cython_type']} {var['name']} {var['default_value_str']},"
        cython_def_lines = [f"def {func.__name__}({vars_string}):\n"]
        return cython_def_lines

    def extract_function_body_type_hints(self, function_lines, cython_def_lines, def_line_index):
        remove_lines = []
        for line in function_lines:
            if ":" in line and "def" not in line:
                var_string = ""
                leading_spaces = "".join([" " for i in range(len(line) - len(line.lstrip()))])
                split_line = line.replace(" ", "").replace("\n", "").split(":")
                var_name = split_line[0]
                if "[" in var_name:
                    continue
                elif len(split_line) > 1:
                    if split_line[1] == "":
                        continue
                split_line = split_line[1].split("=")
                python_type = split_line[0]
                cython_type = self._type_conversion.get(python_type, None)
                if cython_type is None:
                    cython_type = python_type
                if len(split_line) > 1:
                    default_value_str = f"= {split_line[1]}"
                else:
                    default_value_str = ""
                var_string += f"{leading_spaces}cdef {cython_type} {var_name} {default_value_str} \n"
                cython_def_lines.append(var_string)
                remove_lines.append(line)

        checking_lines = ["@cython.boundscheck(False)\n",
                          "@cython.wraparound(False)\n"]
        cython_function_lines = checking_lines + cython_def_lines + function_lines[def_line_index + 1:]
        [cython_function_lines.remove(line) for line in remove_lines]
        return cython_function_lines

    def format_function_code_to_cython(self, func, function_lines, def_line_index, args, kwargs):
        vars = function_lines[def_line_index].split("(")[1].split(",")
        func_args = self.extract_variables_from_definition(func, vars)

        func_args = self.extract_type_from_input_variables(func_args, args, kwargs)

        cython_def_lines = self.build_cython_function_definition(func, func_args)
        cython_function_lines = self.extract_function_body_type_hints(function_lines, cython_def_lines, def_line_index)

        cython_function_code = "".join(cython_function_lines)
        return cython_function_code, cython_def_lines

    def hash_func(self, cython_def_lines):
        return hashlib.md5("".join(cython_def_lines).encode()).hexdigest()

    def cythonize_func_inline_method(self, func, function_lines, def_line_index, args, kwargs, hash_only=False):

        cython_function_code, cython_def_lines = self.format_function_code_to_cython(
            func, function_lines, def_line_index, args, kwargs)

        if hash_only:
            function_hash_name = self.hash_func(cython_def_lines)
            return function_hash_name
        else:
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


# def autocompile(f, mode: str = "inline", infer_types: bool = False):
def autocompile(*ags, **kwgs):
    mode = "inline"
    infer_type = False
    if "mode" in kwgs:
        mode = kwgs["mode"]
    if "infer_type" in kwgs:
        infer_type = kwgs["infer_type"]

    def _autocompile(func):
        ac = AutoCompile(mode=mode, infer_types=infer_type)
        if callable(func):
            mod = __import__(func.__module__)
            for lib_name, lib in mod.__dict__.items():
                if lib_name not in globals():
                    globals()[lib_name] = lib

        @wraps(func)
        def run_func(*args, **kwargs):
            function_lines = inspect.getsourcelines(func)[0]
            def_line_index = ["def" in line.lstrip()[:3] for line in function_lines].index(True)

            function_lines = inspect.getsourcelines(func)[0]
            function_code = "".join(function_lines[def_line_index:])
            function_hash_name = None

            if ac.mode == "file":
                if func.__name__ in ac.cythonized_functions.keys():
                    cythonized_func = ac.cythonized_functions[func.__name__]
                else:
                    cythonized_func = ac.cythonize_func_file_method(
                        filepath=ac.tempfile,
                        function_code=function_code,
                        func_name=func.__name__)
                    function_hash_name = func.__name__
                    ac.cythonized_functions[function_hash_name] = cythonized_func
            else:
                function_hash_name = ac.cythonize_func_inline_method(
                    func=func,
                    function_lines=function_lines,
                    def_line_index=def_line_index,
                    args=args,
                    kwargs=kwargs,
                    hash_only=True)
                if function_hash_name in ac.cythonized_functions.keys():
                    cythonized_func = ac.cythonized_functions[function_hash_name]
                else:
                    function_hash_name, cythonized_func = ac.cythonize_func_inline_method(
                        func=func,
                        function_lines=function_lines,
                        def_line_index=def_line_index,
                        args=args,
                        kwargs=kwargs)
                    ac.cythonized_functions[function_hash_name] = cythonized_func
            return cythonized_func(*args, **kwargs)

        return run_func

    if len(ags) > 0:
        if callable(ags[0]):
            func = ags[0]
            return _autocompile(func)
    return _autocompile


def ac(*args, **kwargs):
    return autocompile(args, kwargs)


if __name__ == '__main__':

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

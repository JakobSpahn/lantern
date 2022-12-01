from ctypes import *
import sys
import os
from enum import IntEnum
from pathlib import Path

import numpy as np

# get the shared library
path = (Path(__file__).parent.parent.parent)
file_extension = ".so"
try:
    rel_filename = path.joinpath("libops" + file_extension)
    handle = CDLL(rel_filename)
except OSError:
    file_extension = ".dylib"
    rel_filename = path.joinpath("libops" + file_extension)
    handle = CDLL(rel_filename)

# define ctypes used
UINT = c_uint
INT = c_int
FLOAT = c_float
INTP = POINTER(INT)
FLOATP = POINTER(FLOAT)
TENSOR_DATA = [FLOATP, UINT, INTP, UINT]
OUTPUT_PARAMS = [POINTER(FLOATP), POINTER(UINT), POINTER(INTP), POINTER(UINT)]

# declarations
handle.matmul.argtypes = [*TENSOR_DATA, *TENSOR_DATA, *OUTPUT_PARAMS]
handle.matmul.restype = c_double

handle.conv2d.argtypes = [*TENSOR_DATA, *TENSOR_DATA, *TENSOR_DATA, c_bool, *OUTPUT_PARAMS]
handle.conv2d.restype = c_double

handle.max_pool2d.argtypes = [*TENSOR_DATA, INTP, UINT, *OUTPUT_PARAMS]
handle.max_pool2d.restype = c_double

def numpy_to_tensor(a : np.ndarray):
    return cast(a.ctypes.data_as(FLOATP),FLOATP), a.size, cast(a.ctypes.shape_as(INT),INTP), len(a.shape)

def helper_c_wrapper(fxn, *args):
     # declare output args
    ptr_res = {'data':FLOATP(),
               'data_n':UINT(),
               'shape':INTP(),
               'shape_n':UINT()
               }

    # call the c function
    dur = fxn(*args, byref(ptr_res["data"]), byref(ptr_res["data_n"]), byref(ptr_res["shape"]), byref(ptr_res['shape_n']))

    # init np array from output args
    ret_shape = tuple(ptr_res['shape'][:ptr_res['shape_n'].value])
    ret = np.ctypeslib.as_array(ptr_res['data'], ret_shape)

    return (ret,dur)

#(num)pythonic wrapper functions
def matmul(a : np.ndarray, b : np.ndarray):
    return helper_c_wrapper(handle.matmul, *numpy_to_tensor(a), *numpy_to_tensor(b))

def conv2d(a : np.ndarray, b : np.ndarray, c : np.ndarray = np.empty(0)):
    return helper_c_wrapper(handle.conv2d, *numpy_to_tensor(a), *numpy_to_tensor(b), *numpy_to_tensor(c), (c.size != 0))

def max_pool2d(a : np.ndarray, kernel_size : int):
    ks = np.asarray(kernel_size, dtype=np.intc)
    return helper_c_wrapper(handle.max_pool2d, *numpy_to_tensor(a), cast(ks.ctypes.data_as(INTP),INTP), ks.size)

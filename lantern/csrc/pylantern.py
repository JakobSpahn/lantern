from ctypes import *
import sys
import os
from enum import IntEnum
from pathlib import Path

import numpy as np

# define ctypes used
UINT = c_uint
INT = c_int
FLOAT = c_float
INTP = POINTER(INT)
FLOATP = POINTER(FLOAT)

# get the shared library
path = (Path(__file__).parent.parent.parent)
rel_filename = path.joinpath('libops.dylib')
handle = CDLL(rel_filename)

# declare the c function
handle.binary_op.argtypes = [FLOATP, UINT, INTP, UINT,
                                   FLOATP, UINT, INTP, UINT,
                                   FLOATP, UINT, INTP, UINT,
                                   c_bool,
                                   POINTER(FLOATP), POINTER(UINT), POINTER(INTP), POINTER(UINT),
                                   INT]
handle.binary_op.restype = c_double

def lantern_wrapper(a : np.ndarray, b : np.ndarray, type : int, c : np.ndarray = np.empty(0)):
        
    # declare output args
    ptr_res = {'data':FLOATP(),
               'data_n':UINT(),
               'shape':INTP(),
               'shape_n':UINT()
               }

    # call the c function
    dur = handle.binary_op(cast(a.ctypes.data_as(FLOATP),FLOATP), a.size, cast(a.ctypes.shape_as(INT),INTP), len(a.shape),
                           cast(b.ctypes.data_as(FLOATP),FLOATP), b.size, cast(b.ctypes.shape_as(INT),INTP), len(b.shape),
                           cast(c.ctypes.data_as(FLOATP),FLOATP), c.size, cast(c.ctypes.shape_as(INT),INTP), len(c.shape), (c.size != 0),
                           byref(ptr_res["data"]), byref(ptr_res["data_n"]), byref(ptr_res["shape"]), byref(ptr_res['shape_n']), # byref() more efficient than pointer()
                           type)

    # init np array from output args
    ret_shape = tuple(ptr_res['shape'][:ptr_res['shape_n'].value])
    ret = np.ctypeslib.as_array(ptr_res['data'], ret_shape)

    return (ret,dur)

class OpType(IntEnum):
    MATMUL = 0
    CONV2D = 1

#lantern helpers
def matmul(a : np.ndarray, b : np.ndarray):
    return lantern_wrapper(a, b, OpType.MATMUL)
def conv2d(a : np.ndarray, b : np.ndarray, c : np.ndarray = np.empty(0)):
    return lantern_wrapper(a, b, OpType.CONV2D, c)

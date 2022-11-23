import time
import unittest

import torch
import numpy as np

import lantern.csrc.pylantern as lantern

def helper_test_op(shps : list, torch_fxn, lantern_fxn, atol=1e-6, rtol=1e-3, a=-0.5, b=3):
    torch.manual_seed(0)
    np.random.seed(0)

    ts = [torch.tensor((np.random.random(size=x).astype(np.float32)+a)*b) for x in shps]
    tsl = [x.detach().numpy() for x in ts]

    st = time.monotonic()
    out = torch_fxn(*ts)
    torch_fp = time.monotonic() - st

    ret,lantern_fp = lantern_fxn(*tsl) 

    def compare(s, x, y, atol, rtol):
        assert x.shape == y.shape, f"shape missmatch {x.shape != y.shape}"
        try:
            np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)
        except Exception:
            raise SanityCheckError(f"{s} failed shape {x.shape}")
    
    compare("forward pass", ret, out.detach().numpy(), atol=atol, rtol=rtol)
    #print(f"torch time: {torch_fp}, lantern time: {lantern_fp}")

    print("\ntesting %40r   torch/lantern fp: %.2f / %.2f ms " % (shps, torch_fp*1000, lantern_fp*1000), end="")

class TestOps(unittest.TestCase):

    def test_matmul(self):
        helper_test_op([(1,65), (65,99)], lambda x,y: x.matmul(y), lantern.matmul)
    def test_simple_conv2d(self):
        helper_test_op([(1,1,9,9), (1,1,3,3)],
                        lambda x,y: torch.nn.functional.conv2d(x,y),
                        lambda x,y: lantern.conv2d(x,y), atol=1e-4)




class SanityCheckError(Exception):
    pass
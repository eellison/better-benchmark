"""
Standard prelude for canonical repros.

Import this once at the top of any repro.py to get all common globals:
    from repro_prelude import *

Sets up: torch, device, inf, nan, inductor_prims, and any other
commonly-needed symbols that repros reference.
"""
import torch
import torch._inductor.inductor_prims  # noqa: F401 (registers inductor RNG ops)
from math import inf, nan  # noqa: F401
from torch import device  # noqa: F401

# Re-export everything repros might need
__all__ = ["torch", "device", "inf", "nan"]

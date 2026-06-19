import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# These two files are one-off A/B analysis SCRIPTS (despite the ``test_``
# prefix) -- they have no pytest ``test_*`` functions and instead run a full
# GPU benchmark sweep at module import time (torch.compile + triton do_bench
# over the cat repros, monkeypatching a pytorch source tree under
# /tmp/pytorch-work). pytest executes module top-level code during collection,
# so collecting them would (a) error without that pytorch tree + a GPU and
# (b) otherwise kick off heavy GPU work. They are kept here for provenance but
# excluded from collection. Run them directly if needed:
#   python tests/test_pointwise_cat_ab.py
collect_ignore = [
    "test_pointwise_cat_ab.py",
    "test_pointwise_cat_fix.py",
]

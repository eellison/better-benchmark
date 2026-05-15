"""Debug: check consumers for all models."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
import sympy, importlib.util, os
from functools import reduce
import operator
cfg.force_disable_caches = True

original_update = GraphLowering._update_scheduler
def debug_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    for op in list(self.operations):
        if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
            pw = op.data
            ranges = pw.ranges
            ndim = len(ranges)
            var_ranges = {}
            iter_vars = []
            for j, r in enumerate(ranges):
                sym = sympy.Symbol(f"d{j}", integer=True, nonneg=True)
                var_ranges[sym] = r
                iter_vars.append(sym)
            try:
                body = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
            except Exception:
                continue
            masked = [k for k in body.subblocks if k.startswith('masked_subblock')]
            if len(masked) == 2:
                buf_name = op.get_name()
                op_numel = reduce(operator.mul, (int(x) for x in ranges), 1)
                print(f"  Candidate: {buf_name} ranges={list(ranges)} numel={op_numel}")
                has_pw_consumer = False
                for other_op in self.operations:
                    if other_op is op:
                        continue
                    if not isinstance(other_op, ir.ComputedBuffer):
                        continue
                    read_names = other_op.get_read_names()
                    if buf_name not in read_names:
                        continue
                    kind = type(other_op.data).__name__
                    other_ranges = list(other_op.data.ranges) if hasattr(other_op.data, 'ranges') else '?'
                    other_numel = reduce(operator.mul, (int(x) for x in other_op.data.ranges), 1) if hasattr(other_op.data, 'ranges') else '?'
                    print(f"    Consumer: {other_op.get_name()} {kind} ranges={other_ranges} numel={other_numel}")
                    if isinstance(other_op.data, (ir.Pointwise, ir.Reduction)):
                        has_pw_consumer = True
                if not has_pw_consumer:
                    print(f"    -> NO Pointwise/Reduction consumers -> PROFITABLE")
                else:
                    print(f"    -> Has Pointwise/Reduction consumers -> NOT profitable")
    
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

repro_dir = "repo/repros/multi_kernel_inference"
models = [
    ("Mistral-7B", "k03_vllm_mistralai_Mistral-7B-Instruct-v0.3_inference__region_011_pointwise_108892bc6e62_8f1de50a.py"),
    ("Qwen3-0.6B (r005)", "k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"),
    ("gpt-oss-20b (r013)", "k03_vllm_openai_gpt-oss-20b_inference__region_013_pointwise_1b8e92f2de8a_3af1681a.py"),
]

for name, filename in models:
    filepath = os.path.join(repro_dir, filename)
    if not os.path.exists(filepath):
        continue
    print(f"\n=== {name} ===")
    spec = importlib.util.spec_from_file_location("repro_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        continue
    inputs = mod.make_inputs()
    torch._dynamo.reset()
    GraphLowering._update_scheduler = debug_update_scheduler
    try:
        compiled = torch.compile(mod.Repro())
        with torch.no_grad():
            compiled(*inputs)
    except Exception as e:
        print(f"  ERROR: {e}")
    GraphLowering._update_scheduler = original_update

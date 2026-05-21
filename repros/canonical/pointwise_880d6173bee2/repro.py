"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train
Pattern hash: 880d6173bee2
Shape hash: e2101542
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_4: "i64[]", getitem_1: "f32[1, 128, 1, 1]", rsqrt: "f32[1, 128, 1, 1]", primals_5: "f32[128]", getitem: "f32[1, 128, 1, 1]", primals_6: "f32[128]", primals_10: "i64[]", getitem_3: "f32[1, 256, 1, 1]", rsqrt_1: "f32[1, 256, 1, 1]", primals_11: "f32[256]", getitem_2: "f32[1, 256, 1, 1]", primals_12: "f32[256]", primals_16: "i64[]", getitem_5: "f32[1, 512, 1, 1]", rsqrt_2: "f32[1, 512, 1, 1]", primals_17: "f32[512]", getitem_4: "f32[1, 512, 1, 1]", primals_18: "f32[512]", convolution_4: "f32[1024, 1, 1, 1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_4, 1)
        squeeze_dims: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_1: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[128]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000038147118175);  squeeze_dims_2 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(primals_6, 0.9)
        add_tensor_2: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_10, 1)
        squeeze_dims_3: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_6: "f32[256]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_tensor_4: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_5: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000152590218967);  squeeze_dims_5 = None
        mul_tensor_8: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[256]" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_tensor_5: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_16, 1)
        squeeze_dims_6: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_7: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1)
        mul_tensor_11: "f32[512]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_tensor_7: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_8: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.0000610388817677);  squeeze_dims_8 = None
        mul_tensor_13: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[512]" = torch.ops.aten.mul.Tensor(primals_18, 0.9)
        add_tensor_8: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        sigmoid_default: "f32[1024, 1, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_4: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_4, add_tensor);  primals_4 = add_tensor = None
        copy__default_1: "f32[128]" = torch.ops.aten.copy_.default(primals_5, add_tensor_1);  primals_5 = add_tensor_1 = None
        copy__default_2: "f32[128]" = torch.ops.aten.copy_.default(primals_6, add_tensor_2);  primals_6 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_10, add_tensor_3);  primals_10 = add_tensor_3 = None
        copy__default_4: "f32[256]" = torch.ops.aten.copy_.default(primals_11, add_tensor_4);  primals_11 = add_tensor_4 = None
        copy__default_5: "f32[256]" = torch.ops.aten.copy_.default(primals_12, add_tensor_5);  primals_12 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_16, add_tensor_6);  primals_16 = add_tensor_6 = None
        copy__default_7: "f32[512]" = torch.ops.aten.copy_.default(primals_17, add_tensor_7);  primals_17 = add_tensor_7 = None
        copy__default_8: "f32[512]" = torch.ops.aten.copy_.default(primals_18, add_tensor_8);  primals_18 = add_tensor_8 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_7, sigmoid_default, unsqueeze_default_2, unsqueeze_default_5, unsqueeze_default_8, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

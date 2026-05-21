"""
Standalone repro captured via capture_hook.
Label: torchbench_mnasnet1_0_train
Pattern hash: cc2edb9afd77
Shape hash: 27800b4f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256, 1280, 7, 7], f32), T([1280], f32), T([1280], f32), T([1280], f32), T([1280], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_51: "f32[256, 1280, 7, 7]", primals_310: "f32[1280]", primals_311: "f32[1280]", primals_312: "f32[1280]", primals_313: "f32[1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:159 in forward, code: x = self.layers(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1280, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1280, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[256, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_1);  convolution_51 = None
        mul_tensor: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.00029999999999996696);  squeeze_dims = None
        mul_tensor_2: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_310, 0.9997)
        add_tensor_1: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000797257434426);  squeeze_dims_1 = None
        mul_tensor_4: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.00029999999999996696);  mul_tensor_3 = None
        mul_tensor_5: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_311, 0.9997)
        add_tensor_2: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1);  primals_312 = None
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[256, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[256, 1280, 7, 7]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:161 in forward, code: x = x.mean([2, 3])
        mean_dim: "f32[256, 1280]" = torch.ops.aten.mean.dim(relu_default, [2, 3]);  relu_default = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:162 in forward, code: return self.classifier(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[256, 1280]" = torch.ops.prims.inductor_random.default([256, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[256, 1280]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8);  inductor_random_default = None
        convert_element_type_default: "f32[256, 1280]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None
        div_scalar: "f32[256, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.8);  convert_element_type_default = None
        mul_tensor_7: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(mean_dim, div_scalar);  mean_dim = div_scalar = None

        # No stacktrace found for following nodes
        copy__default: "f32[1280]" = torch.ops.aten.copy_.default(primals_310, add_tensor_1);  primals_310 = add_tensor_1 = None
        copy__default_1: "f32[1280]" = torch.ops.aten.copy_.default(primals_311, add_tensor_2);  primals_311 = add_tensor_2 = None
        return (copy__default, copy__default_1, mul_tensor_7)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

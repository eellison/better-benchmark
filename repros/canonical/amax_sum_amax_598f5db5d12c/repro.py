"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 598f5db5d12c
Shape hash: f831701a
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
_shapes_config = "(T([128, 2], f32), T([16384, 20005], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 128], f32), T([1536, 128, 64], f32), T([1536, 128, 64], f32), T([1536, 64, 128], f32), S([128, 128, 20005]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_72: "f32[128, 2]", addmm_73: "f32[16384, 20005]", view_254: "f32[1536, 128, 128]", view_255: "f32[1536, 128, 64]", view_251: "f32[1536, 128, 64]", view_252: "f32[1536, 64, 128]", view_232: "f32[1536, 128, 128]", view_233: "f32[1536, 128, 64]", view_229: "f32[1536, 128, 64]", view_230: "f32[1536, 64, 128]", view_210: "f32[1536, 128, 128]", view_211: "f32[1536, 128, 64]", view_207: "f32[1536, 128, 64]", view_208: "f32[1536, 64, 128]", view_188: "f32[1536, 128, 128]", view_189: "f32[1536, 128, 64]", view_185: "f32[1536, 128, 64]", view_186: "f32[1536, 64, 128]", view_166: "f32[1536, 128, 128]", view_167: "f32[1536, 128, 64]", view_163: "f32[1536, 128, 64]", view_164: "f32[1536, 64, 128]", view_144: "f32[1536, 128, 128]", view_145: "f32[1536, 128, 64]", view_141: "f32[1536, 128, 64]", view_142: "f32[1536, 64, 128]", view_122: "f32[1536, 128, 128]", view_123: "f32[1536, 128, 64]", view_119: "f32[1536, 128, 64]", view_120: "f32[1536, 64, 128]", view_100: "f32[1536, 128, 128]", view_101: "f32[1536, 128, 64]", view_97: "f32[1536, 128, 64]", view_98: "f32[1536, 64, 128]", view_78: "f32[1536, 128, 128]", view_79: "f32[1536, 128, 64]", view_75: "f32[1536, 128, 64]", view_76: "f32[1536, 64, 128]", view_56: "f32[1536, 128, 128]", view_57: "f32[1536, 128, 64]", view_53: "f32[1536, 128, 64]", view_54: "f32[1536, 64, 128]", view_34: "f32[1536, 128, 128]", view_35: "f32[1536, 128, 64]", view_31: "f32[1536, 128, 64]", view_32: "f32[1536, 64, 128]", view_12: "f32[1536, 128, 128]", view_13: "f32[1536, 128, 64]", view_9: "f32[1536, 128, 64]", view_10: "f32[1536, 64, 128]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        amax_default: "f32[128, 1]" = torch.ops.aten.amax.default(addmm_72, [-1], True)
        sub_tensor: "f32[128, 2]" = torch.ops.aten.sub.Tensor(addmm_72, amax_default);  addmm_72 = amax_default = None
        exp_default: "f32[128, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[128, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[128, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        reshape_default: "f32[128, 128, 20005]" = torch.ops.aten.reshape.default(addmm_73, _shape_param_0);  addmm_73 = _shape_param_0 = None
        amax_default_1: "f32[128, 128, 1]" = torch.ops.aten.amax.default(reshape_default, [-1], True)
        sub_tensor_2: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(reshape_default, amax_default_1);  reshape_default = amax_default_1 = None
        exp_default_1: "f32[128, 128, 20005]" = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True);  exp_default_1 = None
        log_default_1: "f32[128, 128, 1]" = torch.ops.aten.log.default(sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default_1);  sub_tensor_2 = log_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_default_1: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_2: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_default_3: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_4: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_default_5: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_6: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_default_7: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_8: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_default_9: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_10: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_default_11: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_12: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_default_13: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_14: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_default_15: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_16: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_default_17: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_18: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_default_19: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_20: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_default_21: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_22: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_default_23: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_24: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_default_25: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_26: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_default_27: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_28: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_default_29: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_30: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_default_31: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_32: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_default_33: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_34: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_default_35: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_36: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_default_37: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_38: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_default_39: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_40: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_default_41: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_42: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_default_43: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_default_44: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_default_45: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_46: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_47: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (sub_tensor_1, sub_tensor_3, permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, permute_default_22, permute_default_23, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47)



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

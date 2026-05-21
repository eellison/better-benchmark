"""
Standalone repro captured via capture_hook.
Label: torchbench_pyhpc_equation_of_state_infer
Pattern hash: a2382a85ee99
Shape hash: 7708fa07
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
_shapes_config = "(T([1, 1, 26], f64), T([204, 204, 26], f64), T([204, 204, 26], f64))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f64[1, 1, 26]", arg0_1: "f64[204, 204, 26]", arg1_1: "f64[204, 204, 26]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        mul_tensor: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 10000.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:70 in gsw_dHdT, code: t1 = v45 * ct
        mul_tensor_1: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.8648264253656e-14)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:71 in gsw_dHdT, code: t2 = 0.2e1 * t1
        mul_tensor_2: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_tensor: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_2, -1.941660213148725e-11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:72 in gsw_dHdT, code: t3 = v46 * sa
        mul_tensor_3: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 1.119522344879478e-14)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_tensor_1: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_3);  add_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:74 in gsw_dHdT, code: t5 = v14 * ct
        mul_tensor_4: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 3.726050720345733e-06)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:75 in gsw_dHdT, code: t7 = ct * (v13 + t5)
        add_tensor_2: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_4, -0.000343609007985188)
        mul_tensor_5: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_2);  add_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:76 in gsw_dHdT, code: t8 = 0.5 * t7
        mul_tensor_6: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_5, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:79 in gsw_dHdT, code: t13 = t4 + t8 + t12
        add_tensor_3: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_6, -0.011166348136762635)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:77 in gsw_dHdT, code: t11 = sa * (v15 + v16 * ct)
        mul_tensor_7: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.876837219536232e-07)
        add_tensor_4: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_7, -0.0001806789763745328);  mul_tensor_7 = None
        mul_tensor_8: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_tensor_4);  add_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:78 in gsw_dHdT, code: t12 = 0.5 * t11
        mul_tensor_9: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_8, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:79 in gsw_dHdT, code: t13 = t4 + t8 + t12
        add_tensor_5: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_9);  add_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:190 in gsw_dHdT, code: - 2.0 * v48 * t13 * t20
        mul_tensor_10: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_5, 1.2115804975093732e-16)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:80 in gsw_dHdT, code: t15 = v19 * ct
        mul_tensor_11: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.061519070296458e-11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:81 in gsw_dHdT, code: t19 = v17 + ct * (v18 + t15) + v20 * sa
        add_tensor_6: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_11, -1.988366587925593e-08)
        mul_tensor_12: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_6);  add_tensor_6 = None
        add_tensor_7: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_12, -3.087032500374211e-07);  mul_tensor_12 = None
        mul_tensor_13: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 1.55093272922008e-10)
        add_tensor_8: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_13);  add_tensor_7 = mul_tensor_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:82 in gsw_dHdT, code: t20 = 1.0 / t19
        reciprocal_default: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_tensor_8)
        mul_tensor_14: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1.0);  reciprocal_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:190 in gsw_dHdT, code: - 2.0 * v48 * t13 * t20
        mul_tensor_15: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_14);  mul_tensor_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        sub_tensor: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_1, mul_tensor_15);  add_tensor_1 = mul_tensor_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:83 in gsw_dHdT, code: t24 = v47 + v48 * ct
        mul_tensor_16: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.057902487546866e-17)
        add_tensor_9: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_16, -1.200507748551599e-15);  mul_tensor_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:191 in gsw_dHdT, code: - 2.0 * t24 * t29 * t20
        mul_tensor_17: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_9, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:85 in gsw_dHdT, code: t26 = 1.0 * t5
        mul_tensor_18: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:88 in gsw_dHdT, code: t29 = t25 + t26 + t28
        add_tensor_10: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_18, -0.000171804503992594)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:86 in gsw_dHdT, code: t27 = sa * v16
        mul_tensor_19: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.876837219536232e-07)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:87 in gsw_dHdT, code: t28 = 0.5 * t27
        mul_tensor_20: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_19, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:88 in gsw_dHdT, code: t29 = t25 + t26 + t28
        add_tensor_11: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_10, mul_tensor_20);  add_tensor_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:191 in gsw_dHdT, code: - 2.0 * t24 * t29 * t20
        mul_tensor_21: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_17, add_tensor_11);  mul_tensor_17 = None
        mul_tensor_22: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_21, mul_tensor_14);  mul_tensor_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        sub_tensor_1: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_22);  sub_tensor = mul_tensor_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:89 in gsw_dHdT, code: t33 = t24 * t13
        mul_tensor_23: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_9, add_tensor_5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:192 in gsw_dHdT, code: + 2.0 * t33 * t38
        mul_tensor_24: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_23, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:90 in gsw_dHdT, code: t34 = t19**2
        pow_tensor_scalar: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_8, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:91 in gsw_dHdT, code: t35 = 1.0 / t34
        reciprocal_default_1: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(pow_tensor_scalar);  pow_tensor_scalar = None
        mul_tensor_25: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1.0);  reciprocal_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:92 in gsw_dHdT, code: t37 = v18 + 2.0 * t15
        mul_tensor_26: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 2.0);  mul_tensor_11 = None
        add_tensor_12: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_26, -1.988366587925593e-08);  mul_tensor_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:93 in gsw_dHdT, code: t38 = t35 * t37
        mul_tensor_27: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_25, add_tensor_12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:192 in gsw_dHdT, code: + 2.0 * t33 * t38
        mul_tensor_28: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_24, mul_tensor_27);  mul_tensor_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_tensor_13: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_tensor_1, mul_tensor_28);  sub_tensor_1 = mul_tensor_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:193 in gsw_dHdT, code: + 0.5 * v48 * p
        mul_tensor_29: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 3.028951243773433e-17)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_tensor_14: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_13, mul_tensor_29);  add_tensor_13 = mul_tensor_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        mul_tensor_30: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_14);  mul_tensor = add_tensor_14 = None
        mul_tensor_31: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_30, mul_tensor_14);  mul_tensor_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:196 in gsw_dHdT, code: - 0.1e5 * p * (v43 + t48 - 2.0 * t33 * t20 + 0.5 * t24 * p) * t38
        mul_tensor_32: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 10000.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:94 in gsw_dHdT, code: t48 = ct * (v44 + t1 + t3)
        add_tensor_15: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_1, -1.941660213148725e-11);  mul_tensor_1 = None
        add_tensor_16: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_15, mul_tensor_3);  add_tensor_15 = None
        mul_tensor_33: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_16);  add_tensor_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:196 in gsw_dHdT, code: - 0.1e5 * p * (v43 + t48 - 2.0 * t33 * t20 + 0.5 * t24 * p) * t38
        add_tensor_17: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_33, -1.11901159287511e-10)
        mul_tensor_34: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_23, 2.0)
        mul_tensor_35: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_34, mul_tensor_14);  mul_tensor_34 = None
        sub_tensor_2: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_17, mul_tensor_35);  add_tensor_17 = mul_tensor_35 = None
        mul_tensor_36: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_9, 0.5)
        mul_tensor_37: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_36, arg2_1);  mul_tensor_36 = None
        add_tensor_18: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_tensor_2, mul_tensor_37);  sub_tensor_2 = mul_tensor_37 = None
        mul_tensor_38: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_32, add_tensor_18);  mul_tensor_32 = add_tensor_18 = None
        mul_tensor_39: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_38, mul_tensor_27);  mul_tensor_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_tensor_3: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_31, mul_tensor_39);  mul_tensor_31 = mul_tensor_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:95 in gsw_dHdT, code: t57 = v40 * ct
        mul_tensor_40: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.931012931541776e-12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:96 in gsw_dHdT, code: t59 = ct * (v39 + t57)
        add_tensor_19: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_40, 3.191413910561627e-09)
        mul_tensor_41: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_19);  add_tensor_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_tensor_20: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_41, -3.212746477974189e-07)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:126 in gsw_dHdT, code: + ct * (v39 + 2.0 * t57)
        mul_tensor_42: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_40, 2.0);  mul_tensor_40 = None
        add_tensor_21: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_42, 3.191413910561627e-09);  mul_tensor_42 = None
        mul_tensor_43: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_21);  add_tensor_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_tensor_22: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_20, mul_tensor_43);  add_tensor_20 = mul_tensor_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:127 in gsw_dHdT, code: + sa * v42
        mul_tensor_44: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.211426728363857e-10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_tensor_23: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_22, mul_tensor_44);  add_tensor_22 = mul_tensor_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:97 in gsw_dHdT, code: t64 = t13**2
        pow_tensor_scalar_1: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_5, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        mul_tensor_45: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar_1, 2.4231609950187464e-16)
        mul_tensor_46: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_45, mul_tensor_14);  mul_tensor_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:130 in gsw_dHdT, code: + 8.0 * t33 * t68
        mul_tensor_47: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_23, 8.0);  mul_tensor_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:98 in gsw_dHdT, code: t68 = t20 * t29
        mul_tensor_48: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_14, add_tensor_11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:130 in gsw_dHdT, code: + 8.0 * t33 * t68
        mul_tensor_49: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_47, mul_tensor_48);  mul_tensor_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        add_tensor_24: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_46, mul_tensor_49);  mul_tensor_46 = mul_tensor_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:99 in gsw_dHdT, code: t71 = t24 * t64
        mul_tensor_50: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_9, pow_tensor_scalar_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:131 in gsw_dHdT, code: - 4.0 * t71 * t38
        mul_tensor_51: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_50, 4.0)
        mul_tensor_52: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_51, mul_tensor_27);  mul_tensor_51 = mul_tensor_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_tensor_4: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_24, mul_tensor_52);  add_tensor_24 = mul_tensor_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:100 in gsw_dHdT, code: t74 = v04 * ct
        mul_tensor_53: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 0.001181805545074306)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:101 in gsw_dHdT, code: t76 = ct * (v03 + t74)
        add_tensor_25: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_53, -0.03147759265588511)
        mul_tensor_54: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_25);  add_tensor_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:108 in gsw_dHdT, code: + ct * (v02 + t76)
        add_tensor_26: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_54, 2.839940833161907)
        mul_tensor_55: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_26);  add_tensor_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:107 in gsw_dHdT, code: v01
        add_tensor_27: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_55, 999.8420897506056);  mul_tensor_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:102 in gsw_dHdT, code: t79 = v07 * ct
        mul_tensor_56: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 0.0002327859407479162)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:109 in gsw_dHdT, code: + sa * (v05 + ct * (v06 + t79) + t82 * (v08 + ct * (v09 + t85)))
        add_tensor_28: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_56, -0.02986498947203215)
        mul_tensor_57: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_28);  add_tensor_28 = None
        add_tensor_29: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_57, -6.698001071123802);  mul_tensor_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:103 in gsw_dHdT, code: t82 = torch.sqrt(sa)
        sqrt_default: "f64[204, 204, 26]" = torch.ops.aten.sqrt.default(arg1_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:104 in gsw_dHdT, code: t83 = v11 * ct
        mul_tensor_58: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 1.645039373682922e-07)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:105 in gsw_dHdT, code: t85 = ct * (v10 + t83)
        add_tensor_30: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_58, -1.426984671633621e-05)
        mul_tensor_59: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_30);  add_tensor_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:109 in gsw_dHdT, code: + sa * (v05 + ct * (v06 + t79) + t82 * (v08 + ct * (v09 + t85)))
        add_tensor_31: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_59, 0.00050954225738805)
        mul_tensor_60: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_31);  add_tensor_31 = None
        add_tensor_32: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_60, -0.0398882237896849);  mul_tensor_60 = None
        mul_tensor_61: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt_default, add_tensor_32);  add_tensor_32 = None
        add_tensor_33: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_29, mul_tensor_61);  add_tensor_29 = mul_tensor_61 = None
        mul_tensor_62: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_tensor_33);  add_tensor_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:107 in gsw_dHdT, code: v01
        add_tensor_34: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_27, mul_tensor_62);  add_tensor_27 = mul_tensor_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:111 in gsw_dHdT, code: t93 = v48 * t92
        mul_tensor_63: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_34, 6.057902487546866e-17)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_tensor_5: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_63);  sub_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:113 in gsw_dHdT, code: v02
        add_tensor_35: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_54, 2.839940833161907);  mul_tensor_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:115 in gsw_dHdT, code: + ct * (v03 + 2.0 * t74)
        mul_tensor_64: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_53, 2.0);  mul_tensor_53 = None
        add_tensor_36: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_64, -0.03147759265588511);  mul_tensor_64 = None
        mul_tensor_65: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_36);  add_tensor_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:113 in gsw_dHdT, code: v02
        add_tensor_37: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_35, mul_tensor_65);  add_tensor_35 = mul_tensor_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:116 in gsw_dHdT, code: + sa * (v06 + 2.0 * t79 + t82 * (v09 + t85 + ct * (v10 + 2.0 * t83)))
        mul_tensor_66: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_56, 2.0);  mul_tensor_56 = None
        add_tensor_38: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_66, -0.02986498947203215);  mul_tensor_66 = None
        add_tensor_39: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_59, 0.00050954225738805);  mul_tensor_59 = None
        mul_tensor_67: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_58, 2.0);  mul_tensor_58 = None
        add_tensor_40: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_67, -1.426984671633621e-05);  mul_tensor_67 = None
        mul_tensor_68: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_40);  add_tensor_40 = None
        add_tensor_41: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_39, mul_tensor_68);  add_tensor_39 = mul_tensor_68 = None
        mul_tensor_69: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt_default, add_tensor_41);  add_tensor_41 = None
        add_tensor_42: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_38, mul_tensor_69);  add_tensor_38 = mul_tensor_69 = None
        mul_tensor_70: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_tensor_42);  add_tensor_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:113 in gsw_dHdT, code: v02
        add_tensor_43: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_37, mul_tensor_70);  add_tensor_37 = mul_tensor_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:118 in gsw_dHdT, code: t106 = t24 * t105
        mul_tensor_71: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_9, add_tensor_43)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_tensor_6: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_5, mul_tensor_71);  sub_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:119 in gsw_dHdT, code: t107 = v44 + t2 + t3
        add_tensor_44: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_2, -1.941660213148725e-11);  mul_tensor_2 = None
        add_tensor_45: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_44, mul_tensor_3);  add_tensor_44 = mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:134 in gsw_dHdT, code: - 2.0 * t107 * t13
        mul_tensor_72: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_45, 2.0)
        mul_tensor_73: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_72, add_tensor_5);  mul_tensor_72 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_tensor_7: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_6, mul_tensor_73);  sub_tensor_6 = mul_tensor_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:120 in gsw_dHdT, code: t110 = v43 + t48
        add_tensor_46: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_33, -1.11901159287511e-10);  mul_tensor_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:135 in gsw_dHdT, code: - 2.0 * t110 * t29
        mul_tensor_74: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_46, 2.0)
        mul_tensor_75: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_74, add_tensor_11);  mul_tensor_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_tensor_8: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_7, mul_tensor_75);  sub_tensor_7 = mul_tensor_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:128 in gsw_dHdT, code: + (
        mul_tensor_76: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_8, mul_tensor_14);  sub_tensor_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_tensor_47: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_23, mul_tensor_76);  add_tensor_23 = mul_tensor_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:122 in gsw_dHdT, code: t120 = 4.0 * t71 * t20 - t117 - 2.0 * t110 * t13
        mul_tensor_77: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_50, 4.0);  mul_tensor_50 = None
        mul_tensor_78: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_77, mul_tensor_14);  mul_tensor_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:121 in gsw_dHdT, code: t117 = t24 * t92
        mul_tensor_79: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_9, add_tensor_34);  add_tensor_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:122 in gsw_dHdT, code: t120 = 4.0 * t71 * t20 - t117 - 2.0 * t110 * t13
        sub_tensor_9: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = None
        mul_tensor_80: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_46, 2.0)
        mul_tensor_81: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_80, add_tensor_5);  mul_tensor_80 = None
        sub_tensor_10: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_9, mul_tensor_81);  sub_tensor_9 = mul_tensor_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:138 in gsw_dHdT, code: - t120 * t35 * t37
        mul_tensor_82: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_10, mul_tensor_25)
        mul_tensor_83: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_82, add_tensor_12);  mul_tensor_82 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        sub_tensor_11: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_47, mul_tensor_83);  add_tensor_47 = mul_tensor_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:197 in gsw_dHdT, code: + 0.5e4 * t123 * t20 * t134
        mul_tensor_84: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_11, 5000.0)
        mul_tensor_85: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_84, mul_tensor_14);  mul_tensor_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:141 in gsw_dHdT, code: t130 = p * (1.0 * v12 + 1.0 * t7 + 1.0 * t11 + t128)
        mul_tensor_86: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_5, 1.0);  mul_tensor_5 = None
        add_tensor_48: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_86, -0.02233269627352527);  mul_tensor_86 = None
        mul_tensor_87: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_8, 1.0);  mul_tensor_8 = None
        add_tensor_49: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_48, mul_tensor_87);  add_tensor_48 = mul_tensor_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:140 in gsw_dHdT, code: t128 = t19 * p
        mul_tensor_88: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_8, arg2_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:141 in gsw_dHdT, code: t130 = p * (1.0 * v12 + 1.0 * t7 + 1.0 * t11 + t128)
        add_tensor_50: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_49, mul_tensor_88);  add_tensor_49 = None
        mul_tensor_89: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_tensor_50);  add_tensor_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:142 in gsw_dHdT, code: t131 = 1.0 / t92
        reciprocal_default_2: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_tensor_34)
        mul_tensor_90: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1.0);  reciprocal_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:143 in gsw_dHdT, code: t133 = 1.0 + t130 * t131
        mul_tensor_91: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_89, mul_tensor_90)
        add_tensor_51: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_91, 1.0);  mul_tensor_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:144 in gsw_dHdT, code: t134 = torch.log(t133)
        log_default: "f64[204, 204, 26]" = torch.ops.aten.log.default(add_tensor_51)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:197 in gsw_dHdT, code: + 0.5e4 * t123 * t20 * t134
        mul_tensor_92: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_85, log_default);  mul_tensor_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_tensor_52: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_tensor_3, mul_tensor_92);  sub_tensor_3 = mul_tensor_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:145 in gsw_dHdT, code: t143 = v37 + ct * (v38 + t59) + sa * (v41 + v42 * ct) + t120 * t20
        add_tensor_53: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_41, -3.212746477974189e-07);  mul_tensor_41 = None
        mul_tensor_93: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_53);  add_tensor_53 = None
        add_tensor_54: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_93, -2.742185394906099e-05);  mul_tensor_93 = None
        mul_tensor_94: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.211426728363857e-10)
        add_tensor_55: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_94, -1.105097577149576e-07);  mul_tensor_94 = None
        mul_tensor_95: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_tensor_55);  add_tensor_55 = None
        add_tensor_56: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_54, mul_tensor_95);  add_tensor_54 = mul_tensor_95 = None
        mul_tensor_96: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_10, mul_tensor_14);  sub_tensor_10 = None
        add_tensor_57: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_56, mul_tensor_96);  add_tensor_56 = mul_tensor_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:198 in gsw_dHdT, code: - 0.5e4 * t143 * t35 * t134 * t37
        mul_tensor_97: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_57, 5000.0)
        mul_tensor_98: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_97, mul_tensor_25);  mul_tensor_97 = None
        mul_tensor_99: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_98, log_default);  mul_tensor_98 = log_default = None
        mul_tensor_100: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_99, add_tensor_12);  mul_tensor_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_tensor_12: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_52, mul_tensor_100);  add_tensor_52 = mul_tensor_100 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:199 in gsw_dHdT, code: + 0.5e4
        mul_tensor_101: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_57, 5000.0)
        mul_tensor_102: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_101, mul_tensor_14);  mul_tensor_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:202 in gsw_dHdT, code: * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131 - t130 / t156 * t105)
        mul_tensor_103: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 2.0);  mul_tensor_4 = None
        add_tensor_58: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_103, -0.000343609007985188);  mul_tensor_103 = None
        mul_tensor_104: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_19, 1.0);  mul_tensor_19 = None
        add_tensor_59: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_58, mul_tensor_104);  add_tensor_58 = mul_tensor_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:146 in gsw_dHdT, code: t152 = t37 * p
        mul_tensor_105: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_12, arg2_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:202 in gsw_dHdT, code: * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131 - t130 / t156 * t105)
        add_tensor_60: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_59, mul_tensor_105);  add_tensor_59 = None
        mul_tensor_106: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_tensor_60);  arg2_1 = add_tensor_60 = None
        mul_tensor_107: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_106, mul_tensor_90);  mul_tensor_106 = mul_tensor_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:147 in gsw_dHdT, code: t156 = t92**2
        pow_tensor_scalar_2: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_34, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:202 in gsw_dHdT, code: * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131 - t130 / t156 * t105)
        div_tensor: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_89, pow_tensor_scalar_2);  mul_tensor_89 = pow_tensor_scalar_2 = None
        mul_tensor_108: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_tensor, add_tensor_43);  div_tensor = None
        sub_tensor_13: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_107, mul_tensor_108);  mul_tensor_107 = mul_tensor_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:199 in gsw_dHdT, code: + 0.5e4
        mul_tensor_109: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_102, sub_tensor_13);  mul_tensor_102 = sub_tensor_13 = None
        div_tensor_1: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_109, add_tensor_51);  mul_tensor_109 = add_tensor_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_tensor_61: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_tensor_12, div_tensor_1);  sub_tensor_12 = div_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:148 in gsw_dHdT, code: t165 = v25 * ct
        mul_tensor_110: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.743689325042773e-10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:149 in gsw_dHdT, code: t167 = ct * (v24 + t165)
        add_tensor_62: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_110, 1.119513357486743e-06)
        mul_tensor_111: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_62);  add_tensor_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:150 in gsw_dHdT, code: t169 = ct * (v23 + t167)
        add_tensor_63: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_111, -2.349607444135925e-05)
        mul_tensor_112: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_63);  add_tensor_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_tensor_64: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_112, 0.002775927747785646)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:209 in gsw_dHdT, code: + ct * (v23 + t167 + ct * (v24 + 2.0 * t165))
        add_tensor_65: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_111, -2.349607444135925e-05);  mul_tensor_111 = None
        mul_tensor_113: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_110, 2.0);  mul_tensor_110 = None
        add_tensor_66: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_113, 1.119513357486743e-06);  mul_tensor_113 = None
        mul_tensor_114: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_66);  add_tensor_66 = None
        add_tensor_67: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_65, mul_tensor_114);  add_tensor_65 = mul_tensor_114 = None
        mul_tensor_115: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_67);  add_tensor_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_tensor_68: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_64, mul_tensor_115);  add_tensor_64 = mul_tensor_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:151 in gsw_dHdT, code: t175 = v30 * ct
        mul_tensor_116: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.811147201949891e-11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:152 in gsw_dHdT, code: t177 = ct * (v29 + t175)
        add_tensor_69: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_116, 9.527875081696435e-10)
        mul_tensor_117: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_69);  add_tensor_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:153 in gsw_dHdT, code: t179 = ct * (v28 + t177)
        add_tensor_70: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_117, 1.262937315098546e-07)
        mul_tensor_118: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_70);  add_tensor_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:212 in gsw_dHdT, code: v27
        add_tensor_71: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_118, -2.764306979894411e-05)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:214 in gsw_dHdT, code: + ct * (v28 + t177 + ct * (v29 + 2.0 * t175))
        add_tensor_72: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_117, 1.262937315098546e-07);  mul_tensor_117 = None
        mul_tensor_119: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_116, 2.0);  mul_tensor_116 = None
        add_tensor_73: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_119, 9.527875081696435e-10);  mul_tensor_119 = None
        mul_tensor_120: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_73);  add_tensor_73 = None
        add_tensor_74: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_72, mul_tensor_120);  add_tensor_72 = mul_tensor_120 = None
        mul_tensor_121: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_74);  add_tensor_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:212 in gsw_dHdT, code: v27
        add_tensor_75: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_71, mul_tensor_121);  add_tensor_71 = mul_tensor_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:154 in gsw_dHdT, code: t185 = v35 * ct
        mul_tensor_122: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 2.681097235569143e-12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:155 in gsw_dHdT, code: t187 = ct * (v34 + t185)
        add_tensor_76: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_122, -4.634182341116144e-11)
        mul_tensor_123: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_76);  add_tensor_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:156 in gsw_dHdT, code: t189 = ct * (v33 + t187)
        add_tensor_77: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_123, -7.672876869259043e-09)
        mul_tensor_124: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_77);  add_tensor_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:215 in gsw_dHdT, code: + t82 * (v32 + t189 + ct * (v33 + t187 + ct * (v34 + 2.0 * t185)))
        add_tensor_78: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_124, 3.801564588876298e-07)
        add_tensor_79: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_123, -7.672876869259043e-09);  mul_tensor_123 = None
        mul_tensor_125: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 2.0);  mul_tensor_122 = None
        add_tensor_80: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_125, -4.634182341116144e-11);  mul_tensor_125 = None
        mul_tensor_126: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_80);  add_tensor_80 = None
        add_tensor_81: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_79, mul_tensor_126);  add_tensor_79 = mul_tensor_126 = None
        mul_tensor_127: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_81);  add_tensor_81 = None
        add_tensor_82: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_78, mul_tensor_127);  add_tensor_78 = mul_tensor_127 = None
        mul_tensor_128: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt_default, add_tensor_82);  add_tensor_82 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:212 in gsw_dHdT, code: v27
        add_tensor_83: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_75, mul_tensor_128);  add_tensor_75 = mul_tensor_128 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:210 in gsw_dHdT, code: + sa
        mul_tensor_129: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_tensor_83);  add_tensor_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_tensor_84: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_68, mul_tensor_129);  add_tensor_68 = mul_tensor_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        mul_tensor_130: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_63, 2.0);  mul_tensor_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:157 in gsw_dHdT, code: t199 = t13 * t20
        mul_tensor_131: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_5, mul_tensor_14)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        mul_tensor_132: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:219 in gsw_dHdT, code: + 2.0 * t106 * t199
        mul_tensor_133: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_71, 2.0);  mul_tensor_71 = None
        mul_tensor_134: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_133, mul_tensor_131);  mul_tensor_133 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        add_tensor_85: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_132, mul_tensor_134);  mul_tensor_132 = mul_tensor_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:220 in gsw_dHdT, code: + 2.0 * t117 * t68
        mul_tensor_135: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_79, 2.0)
        mul_tensor_136: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_135, mul_tensor_48);  mul_tensor_135 = mul_tensor_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        add_tensor_86: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_85, mul_tensor_136);  add_tensor_85 = mul_tensor_136 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:221 in gsw_dHdT, code: - 2.0 * t117 * t13 * t35 * t37
        mul_tensor_137: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_79, 2.0)
        mul_tensor_138: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_137, add_tensor_5);  mul_tensor_137 = None
        mul_tensor_139: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_138, mul_tensor_25);  mul_tensor_138 = None
        mul_tensor_140: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_139, add_tensor_12);  mul_tensor_139 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        sub_tensor_14: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_86, mul_tensor_140);  add_tensor_86 = mul_tensor_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:222 in gsw_dHdT, code: - t107 * t92
        mul_tensor_141: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_45, add_tensor_34);  add_tensor_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        sub_tensor_15: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_14, mul_tensor_141);  sub_tensor_14 = mul_tensor_141 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:223 in gsw_dHdT, code: - t110 * t105
        mul_tensor_142: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_46, add_tensor_43)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        sub_tensor_16: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_15, mul_tensor_142);  sub_tensor_15 = mul_tensor_142 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:217 in gsw_dHdT, code: + (
        mul_tensor_143: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_16, mul_tensor_14);  sub_tensor_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_tensor_87: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_84, mul_tensor_143);  add_tensor_84 = mul_tensor_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:158 in gsw_dHdT, code: t217 = 2.0 * t117 * t199 - t110 * t92
        mul_tensor_144: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_79, 2.0);  mul_tensor_79 = None
        mul_tensor_145: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_144, mul_tensor_131);  mul_tensor_144 = mul_tensor_131 = None
        mul_tensor_146: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_46, add_tensor_34);  add_tensor_46 = None
        sub_tensor_17: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:226 in gsw_dHdT, code: - t217 * t35 * t37
        mul_tensor_147: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_17, mul_tensor_25)
        mul_tensor_148: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_147, add_tensor_12);  mul_tensor_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        sub_tensor_18: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_87, mul_tensor_148);  add_tensor_87 = mul_tensor_148 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        mul_tensor_149: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_18, add_tensor_8);  sub_tensor_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:161 in gsw_dHdT, code: + ct * (v22 + t169)
        add_tensor_88: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_112, 0.002775927747785646);  mul_tensor_112 = None
        mul_tensor_150: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_88);  add_tensor_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:160 in gsw_dHdT, code: v21
        add_tensor_89: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_150, 1.0);  mul_tensor_150 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:162 in gsw_dHdT, code: + sa * (v26 + ct * (v27 + t179) + v36 * sa + t82 * (v31 + ct * (v32 + t189)))
        add_tensor_90: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_118, -2.764306979894411e-05);  mul_tensor_118 = None
        mul_tensor_151: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_90);  add_tensor_90 = None
        add_tensor_91: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_151, -0.007521448093615448);  mul_tensor_151 = None
        mul_tensor_152: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 5.41932655114874e-06)
        add_tensor_92: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_91, mul_tensor_152);  add_tensor_91 = mul_tensor_152 = None
        add_tensor_93: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_124, 3.801564588876298e-07);  mul_tensor_124 = None
        mul_tensor_153: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_tensor_93);  arg0_1 = add_tensor_93 = None
        add_tensor_94: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_153, -3.303308871386421e-05);  mul_tensor_153 = None
        mul_tensor_154: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt_default, add_tensor_94);  sqrt_default = add_tensor_94 = None
        add_tensor_95: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_92, mul_tensor_154);  add_tensor_92 = mul_tensor_154 = None
        mul_tensor_155: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_tensor_95);  arg1_1 = add_tensor_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:160 in gsw_dHdT, code: v21
        add_tensor_96: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_89, mul_tensor_155);  add_tensor_89 = mul_tensor_155 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:163 in gsw_dHdT, code: + t217 * t20
        mul_tensor_156: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_17, mul_tensor_14);  sub_tensor_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:160 in gsw_dHdT, code: v21
        add_tensor_97: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_96, mul_tensor_156);  add_tensor_96 = mul_tensor_156 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:229 in gsw_dHdT, code: + t234 * t37
        mul_tensor_157: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_97, add_tensor_12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        add_tensor_98: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_149, mul_tensor_157);  mul_tensor_149 = mul_tensor_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:230 in gsw_dHdT, code: - t123 * t13
        mul_tensor_158: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_11, add_tensor_5);  sub_tensor_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        sub_tensor_19: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_98, mul_tensor_158);  add_tensor_98 = mul_tensor_158 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:231 in gsw_dHdT, code: - t143 * t29
        mul_tensor_159: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_57, add_tensor_11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        sub_tensor_20: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_19, mul_tensor_159);  sub_tensor_19 = mul_tensor_159 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:204 in gsw_dHdT, code: + 0.5e4
        mul_tensor_160: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_20, 5000.0);  sub_tensor_20 = None
        mul_tensor_161: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_160, mul_tensor_14);  mul_tensor_160 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:165 in gsw_dHdT, code: t241 = t64 - t92 * t19
        mul_tensor_162: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_34, add_tensor_8)
        sub_tensor_21: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(pow_tensor_scalar_1, mul_tensor_162);  pow_tensor_scalar_1 = mul_tensor_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:166 in gsw_dHdT, code: t242 = torch.sqrt(t241)
        sqrt_default_1: "f64[204, 204, 26]" = torch.ops.aten.sqrt.default(sub_tensor_21)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:167 in gsw_dHdT, code: t243 = 1.0 / t242
        reciprocal_default_3: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(sqrt_default_1)
        mul_tensor_163: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_default_3, 1.0);  reciprocal_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:173 in gsw_dHdT, code: t252 = 1.0 + 2.0 * t128 * t249
        mul_tensor_164: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_88, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:168 in gsw_dHdT, code: t244 = t4 + t8 + t12 - t242
        add_tensor_99: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_6, -0.011166348136762635)
        add_tensor_100: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_99, mul_tensor_9);  add_tensor_99 = None
        sub_tensor_22: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_100, sqrt_default_1);  add_tensor_100 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:169 in gsw_dHdT, code: t245 = 1.0 / t244
        reciprocal_default_4: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(sub_tensor_22)
        mul_tensor_165: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_default_4, 1.0);  reciprocal_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:172 in gsw_dHdT, code: t249 = t242 * t245 * t248
        mul_tensor_166: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt_default_1, mul_tensor_165)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:170 in gsw_dHdT, code: t247 = t4 + t8 + t12 + t242 + t128
        add_tensor_101: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_6, -0.011166348136762635);  mul_tensor_6 = None
        add_tensor_102: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_101, mul_tensor_9);  add_tensor_101 = mul_tensor_9 = None
        add_tensor_103: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_102, sqrt_default_1);  add_tensor_102 = None
        add_tensor_104: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_103, mul_tensor_88);  add_tensor_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:171 in gsw_dHdT, code: t248 = 1.0 / t247
        reciprocal_default_5: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_tensor_104)
        mul_tensor_167: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_default_5, 1.0);  reciprocal_default_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:172 in gsw_dHdT, code: t249 = t242 * t245 * t248
        mul_tensor_168: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_166, mul_tensor_167);  mul_tensor_166 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:173 in gsw_dHdT, code: t252 = 1.0 + 2.0 * t128 * t249
        mul_tensor_169: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_164, mul_tensor_168);  mul_tensor_164 = None
        add_tensor_105: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_169, 1.0);  mul_tensor_169 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:174 in gsw_dHdT, code: t253 = torch.log(t252)
        log_default_1: "f64[204, 204, 26]" = torch.ops.aten.log.default(add_tensor_105)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:175 in gsw_dHdT, code: t254 = t243 * t253
        mul_tensor_170: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_163, log_default_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:204 in gsw_dHdT, code: + 0.5e4
        mul_tensor_171: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_161, mul_tensor_170);  mul_tensor_161 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_tensor_106: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_61, mul_tensor_171);  add_tensor_61 = mul_tensor_171 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:176 in gsw_dHdT, code: t259 = t234 * t19 - t143 * t13
        mul_tensor_172: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_97, add_tensor_8);  add_tensor_97 = None
        mul_tensor_173: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_57, add_tensor_5);  add_tensor_57 = None
        sub_tensor_23: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_172, mul_tensor_173);  mul_tensor_172 = mul_tensor_173 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:235 in gsw_dHdT, code: - 0.5e4 * t259 * t35 * t254 * t37
        mul_tensor_174: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_23, 5000.0)
        mul_tensor_175: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_174, mul_tensor_25);  mul_tensor_174 = mul_tensor_25 = None
        mul_tensor_176: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_175, mul_tensor_170);  mul_tensor_175 = mul_tensor_170 = None
        mul_tensor_177: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_176, add_tensor_12);  mul_tensor_176 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_tensor_24: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_106, mul_tensor_177);  add_tensor_106 = mul_tensor_177 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:177 in gsw_dHdT, code: t264 = t259 * t20
        mul_tensor_178: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_23, mul_tensor_14);  sub_tensor_23 = mul_tensor_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:236 in gsw_dHdT, code: - 0.25e4 * t264 / t242 / t241 * t253 * t272
        mul_tensor_179: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_178, 2500.0)
        div_tensor_2: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_179, sqrt_default_1);  mul_tensor_179 = None
        div_tensor_3: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(div_tensor_2, sub_tensor_21);  div_tensor_2 = sub_tensor_21 = None
        mul_tensor_180: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_tensor_3, log_default_1);  div_tensor_3 = log_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:178 in gsw_dHdT, code: t272 = 2.0 * t13 * t29 - t105 * t19 - t92 * t37
        mul_tensor_181: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_5, 2.0);  add_tensor_5 = None
        mul_tensor_182: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_181, add_tensor_11);  mul_tensor_181 = add_tensor_11 = None
        mul_tensor_183: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_43, add_tensor_8);  add_tensor_43 = add_tensor_8 = None
        sub_tensor_25: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_182, mul_tensor_183);  mul_tensor_182 = mul_tensor_183 = None
        mul_tensor_184: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_tensor_34, add_tensor_12);  add_tensor_34 = add_tensor_12 = None
        sub_tensor_26: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_25, mul_tensor_184);  sub_tensor_25 = mul_tensor_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:236 in gsw_dHdT, code: - 0.25e4 * t264 / t242 / t241 * t253 * t272
        mul_tensor_185: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_180, sub_tensor_26);  mul_tensor_180 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_tensor_27: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_24, mul_tensor_185);  sub_tensor_24 = mul_tensor_185 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:237 in gsw_dHdT, code: + 0.5e4
        mul_tensor_186: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_178, 5000.0);  mul_tensor_178 = None
        mul_tensor_187: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_186, mul_tensor_163);  mul_tensor_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        mul_tensor_188: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_105, 2.0)
        mul_tensor_189: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_188, mul_tensor_168);  mul_tensor_188 = mul_tensor_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:242 in gsw_dHdT, code: + t128 * t243 * t245 * t248 * t272
        mul_tensor_190: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_88, mul_tensor_163)
        mul_tensor_191: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_190, mul_tensor_165);  mul_tensor_190 = None
        mul_tensor_192: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_191, mul_tensor_167);  mul_tensor_191 = None
        mul_tensor_193: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_192, sub_tensor_26);  mul_tensor_192 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        add_tensor_107: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_189, mul_tensor_193);  mul_tensor_189 = mul_tensor_193 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:179 in gsw_dHdT, code: t282 = t128 * t242
        mul_tensor_194: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_88, sqrt_default_1);  mul_tensor_88 = sqrt_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:243 in gsw_dHdT, code: - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
        mul_tensor_195: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_194, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:180 in gsw_dHdT, code: t283 = t244**2
        pow_tensor_scalar_3: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(sub_tensor_22, 2);  sub_tensor_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:243 in gsw_dHdT, code: - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
        div_tensor_4: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_195, pow_tensor_scalar_3);  mul_tensor_195 = pow_tensor_scalar_3 = None
        mul_tensor_196: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_tensor_4, mul_tensor_167);  div_tensor_4 = mul_tensor_167 = None
        add_tensor_108: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_18, -0.000171804503992594)
        add_tensor_109: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_108, mul_tensor_20);  add_tensor_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:181 in gsw_dHdT, code: t287 = t243 * t272 / 2.0
        mul_tensor_197: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_163, sub_tensor_26);  mul_tensor_163 = sub_tensor_26 = None
        div_tensor_5: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_197, 2.0);  mul_tensor_197 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:243 in gsw_dHdT, code: - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
        sub_tensor_28: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_109, div_tensor_5);  add_tensor_109 = None
        mul_tensor_198: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_196, sub_tensor_28);  mul_tensor_196 = sub_tensor_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        sub_tensor_29: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_tensor_107, mul_tensor_198);  add_tensor_107 = mul_tensor_198 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:244 in gsw_dHdT, code: - 2.0 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)
        mul_tensor_199: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_194, 2.0);  mul_tensor_194 = None
        mul_tensor_200: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_199, mul_tensor_165);  mul_tensor_199 = mul_tensor_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:182 in gsw_dHdT, code: t292 = t247**2
        pow_tensor_scalar_4: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_104, 2);  add_tensor_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:244 in gsw_dHdT, code: - 2.0 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)
        div_tensor_6: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_200, pow_tensor_scalar_4);  mul_tensor_200 = pow_tensor_scalar_4 = None
        add_tensor_110: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_18, -0.000171804503992594);  mul_tensor_18 = None
        add_tensor_111: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_110, mul_tensor_20);  add_tensor_110 = mul_tensor_20 = None
        add_tensor_112: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_111, div_tensor_5);  add_tensor_111 = div_tensor_5 = None
        add_tensor_113: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_tensor_112, mul_tensor_105);  add_tensor_112 = mul_tensor_105 = None
        mul_tensor_201: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_tensor_6, add_tensor_113);  div_tensor_6 = add_tensor_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        sub_tensor_30: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_tensor_29, mul_tensor_201);  sub_tensor_29 = mul_tensor_201 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:237 in gsw_dHdT, code: + 0.5e4
        mul_tensor_202: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_187, sub_tensor_30);  mul_tensor_187 = sub_tensor_30 = None
        div_tensor_7: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_202, add_tensor_105);  mul_tensor_202 = add_tensor_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_tensor_114: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_tensor_27, div_tensor_7);  sub_tensor_27 = div_tensor_7 = None
        return add_tensor_114



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

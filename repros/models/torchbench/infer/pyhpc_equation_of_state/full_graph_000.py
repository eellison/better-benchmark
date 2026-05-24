import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f64[204, 204, 26]", arg1_1: "f64[204, 204, 26]", arg2_1: "f64[1, 1, 26]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        mul_115: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 10000.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:70 in gsw_dHdT, code: t1 = v45 * ct
        mul: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.8648264253656e-14)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:71 in gsw_dHdT, code: t2 = 0.2e1 * t1
        mul_1: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_74: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_1, -1.941660213148725e-11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:72 in gsw_dHdT, code: t3 = v46 * sa
        mul_2: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 1.119522344879478e-14)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_75: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_74, mul_2);  add_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:74 in gsw_dHdT, code: t5 = v14 * ct
        mul_3: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 3.726050720345733e-06)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:75 in gsw_dHdT, code: t7 = ct * (v13 + t5)
        add: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_3, -0.000343609007985188)
        mul_4: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add);  add = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:76 in gsw_dHdT, code: t8 = 0.5 * t7
        mul_5: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_4, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:79 in gsw_dHdT, code: t13 = t4 + t8 + t12
        add_2: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_5, -0.011166348136762635)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:77 in gsw_dHdT, code: t11 = sa * (v15 + v16 * ct)
        mul_6: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.876837219536232e-07)
        add_1: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_6, -0.0001806789763745328);  mul_6 = None
        mul_7: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_1);  add_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:78 in gsw_dHdT, code: t12 = 0.5 * t11
        mul_8: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_7, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:79 in gsw_dHdT, code: t13 = t4 + t8 + t12
        add_3: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_2, mul_8);  add_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:190 in gsw_dHdT, code: - 2.0 * v48 * t13 * t20
        mul_116: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_3, 1.2115804975093732e-16)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:80 in gsw_dHdT, code: t15 = v19 * ct
        mul_9: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.061519070296458e-11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:81 in gsw_dHdT, code: t19 = v17 + ct * (v18 + t15) + v20 * sa
        add_4: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_9, -1.988366587925593e-08)
        mul_10: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_4);  add_4 = None
        add_5: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_10, -3.087032500374211e-07);  mul_10 = None
        mul_11: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 1.55093272922008e-10)
        add_6: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_5, mul_11);  add_5 = mul_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:82 in gsw_dHdT, code: t20 = 1.0 / t19
        reciprocal: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_6)
        mul_12: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal, 1.0);  reciprocal = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:190 in gsw_dHdT, code: - 2.0 * v48 * t13 * t20
        mul_117: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_116, mul_12);  mul_116 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        sub_14: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_75, mul_117);  add_75 = mul_117 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:83 in gsw_dHdT, code: t24 = v47 + v48 * ct
        mul_13: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.057902487546866e-17)
        add_7: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_13, -1.200507748551599e-15);  mul_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:191 in gsw_dHdT, code: - 2.0 * t24 * t29 * t20
        mul_118: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_7, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:85 in gsw_dHdT, code: t26 = 1.0 * t5
        mul_14: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_3, 1.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:88 in gsw_dHdT, code: t29 = t25 + t26 + t28
        add_8: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_14, -0.000171804503992594)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:86 in gsw_dHdT, code: t27 = sa * v16
        mul_15: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.876837219536232e-07)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:87 in gsw_dHdT, code: t28 = 0.5 * t27
        mul_16: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_15, 0.5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:88 in gsw_dHdT, code: t29 = t25 + t26 + t28
        add_9: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_8, mul_16);  add_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:191 in gsw_dHdT, code: - 2.0 * t24 * t29 * t20
        mul_119: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_118, add_9);  mul_118 = None
        mul_120: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_119, mul_12);  mul_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        sub_15: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_14, mul_120);  sub_14 = mul_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:89 in gsw_dHdT, code: t33 = t24 * t13
        mul_17: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_7, add_3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:192 in gsw_dHdT, code: + 2.0 * t33 * t38
        mul_121: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_17, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:90 in gsw_dHdT, code: t34 = t19**2
        pow_1: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:91 in gsw_dHdT, code: t35 = 1.0 / t34
        reciprocal_1: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul_18: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1.0);  reciprocal_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:92 in gsw_dHdT, code: t37 = v18 + 2.0 * t15
        mul_19: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_9, 2.0);  mul_9 = None
        add_10: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_19, -1.988366587925593e-08);  mul_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:93 in gsw_dHdT, code: t38 = t35 * t37
        mul_20: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_18, add_10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:192 in gsw_dHdT, code: + 2.0 * t33 * t38
        mul_122: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_121, mul_20);  mul_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_76: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_15, mul_122);  sub_15 = mul_122 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:193 in gsw_dHdT, code: + 0.5 * v48 * p
        mul_123: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 3.028951243773433e-17)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:187 in gsw_dHdT, code: v44
        add_77: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_76, mul_123);  add_76 = mul_123 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        mul_124: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_115, add_77);  mul_115 = add_77 = None
        mul_125: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_124, mul_12);  mul_124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:196 in gsw_dHdT, code: - 0.1e5 * p * (v43 + t48 - 2.0 * t33 * t20 + 0.5 * t24 * p) * t38
        mul_126: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 10000.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:94 in gsw_dHdT, code: t48 = ct * (v44 + t1 + t3)
        add_11: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul, -1.941660213148725e-11);  mul = None
        add_12: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_11, mul_2);  add_11 = None
        mul_21: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_12);  add_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:196 in gsw_dHdT, code: - 0.1e5 * p * (v43 + t48 - 2.0 * t33 * t20 + 0.5 * t24 * p) * t38
        add_78: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_21, -1.11901159287511e-10)
        mul_127: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_17, 2.0)
        mul_128: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_127, mul_12);  mul_127 = None
        sub_16: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_78, mul_128);  add_78 = mul_128 = None
        mul_129: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_7, 0.5)
        mul_130: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_129, arg2_1);  mul_129 = None
        add_79: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_16, mul_130);  sub_16 = mul_130 = None
        mul_131: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_126, add_79);  mul_126 = add_79 = None
        mul_132: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_131, mul_20);  mul_131 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_17: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_125, mul_132);  mul_125 = mul_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:95 in gsw_dHdT, code: t57 = v40 * ct
        mul_22: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.931012931541776e-12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:96 in gsw_dHdT, code: t59 = ct * (v39 + t57)
        add_13: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_22, 3.191413910561627e-09)
        mul_23: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_13);  add_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_36: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_23, -3.212746477974189e-07)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:126 in gsw_dHdT, code: + ct * (v39 + 2.0 * t57)
        mul_50: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_22, 2.0);  mul_22 = None
        add_37: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_50, 3.191413910561627e-09);  mul_50 = None
        mul_51: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_37);  add_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_38: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_36, mul_51);  add_36 = mul_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:127 in gsw_dHdT, code: + sa * v42
        mul_52: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.211426728363857e-10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_39: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_38, mul_52);  add_38 = mul_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:97 in gsw_dHdT, code: t64 = t13**2
        pow_2: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_3, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        mul_53: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(pow_2, 2.4231609950187464e-16)
        mul_54: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_53, mul_12);  mul_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:130 in gsw_dHdT, code: + 8.0 * t33 * t68
        mul_55: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_17, 8.0);  mul_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:98 in gsw_dHdT, code: t68 = t20 * t29
        mul_24: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_12, add_9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:130 in gsw_dHdT, code: + 8.0 * t33 * t68
        mul_56: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_55, mul_24);  mul_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        add_40: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_54, mul_56);  mul_54 = mul_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:99 in gsw_dHdT, code: t71 = t24 * t64
        mul_25: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_7, pow_2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:131 in gsw_dHdT, code: - 4.0 * t71 * t38
        mul_57: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_25, 4.0)
        mul_58: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_57, mul_20);  mul_57 = mul_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_2: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_40, mul_58);  add_40 = mul_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:100 in gsw_dHdT, code: t74 = v04 * ct
        mul_26: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 0.001181805545074306)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:101 in gsw_dHdT, code: t76 = ct * (v03 + t74)
        add_14: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_26, -0.03147759265588511)
        mul_27: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_14);  add_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:108 in gsw_dHdT, code: + ct * (v02 + t76)
        add_16: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_27, 2.839940833161907)
        mul_31: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_16);  add_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:107 in gsw_dHdT, code: v01
        add_17: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_31, 999.8420897506056);  mul_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:102 in gsw_dHdT, code: t79 = v07 * ct
        mul_28: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 0.0002327859407479162)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:109 in gsw_dHdT, code: + sa * (v05 + ct * (v06 + t79) + t82 * (v08 + ct * (v09 + t85)))
        add_18: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_28, -0.02986498947203215)
        mul_32: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_18);  add_18 = None
        add_19: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_32, -6.698001071123802);  mul_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:103 in gsw_dHdT, code: t82 = torch.sqrt(sa)
        sqrt: "f64[204, 204, 26]" = torch.ops.aten.sqrt.default(arg1_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:104 in gsw_dHdT, code: t83 = v11 * ct
        mul_29: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 1.645039373682922e-07)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:105 in gsw_dHdT, code: t85 = ct * (v10 + t83)
        add_15: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_29, -1.426984671633621e-05)
        mul_30: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_15);  add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:109 in gsw_dHdT, code: + sa * (v05 + ct * (v06 + t79) + t82 * (v08 + ct * (v09 + t85)))
        add_20: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_30, 0.00050954225738805)
        mul_33: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_20);  add_20 = None
        add_21: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_33, -0.0398882237896849);  mul_33 = None
        mul_34: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_21);  add_21 = None
        add_22: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_19, mul_34);  add_19 = mul_34 = None
        mul_35: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_22);  add_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:107 in gsw_dHdT, code: v01
        add_23: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_17, mul_35);  add_17 = mul_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:111 in gsw_dHdT, code: t93 = v48 * t92
        mul_36: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_23, 6.057902487546866e-17)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_3: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_2, mul_36);  sub_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:113 in gsw_dHdT, code: v02
        add_24: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_27, 2.839940833161907);  mul_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:115 in gsw_dHdT, code: + ct * (v03 + 2.0 * t74)
        mul_37: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_26, 2.0);  mul_26 = None
        add_25: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_37, -0.03147759265588511);  mul_37 = None
        mul_38: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_25);  add_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:113 in gsw_dHdT, code: v02
        add_26: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_24, mul_38);  add_24 = mul_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:116 in gsw_dHdT, code: + sa * (v06 + 2.0 * t79 + t82 * (v09 + t85 + ct * (v10 + 2.0 * t83)))
        mul_39: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_28, 2.0);  mul_28 = None
        add_27: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_39, -0.02986498947203215);  mul_39 = None
        add_28: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_30, 0.00050954225738805);  mul_30 = None
        mul_40: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_29, 2.0);  mul_29 = None
        add_29: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_40, -1.426984671633621e-05);  mul_40 = None
        mul_41: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_29);  add_29 = None
        add_30: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_28, mul_41);  add_28 = mul_41 = None
        mul_42: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_30);  add_30 = None
        add_31: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_27, mul_42);  add_27 = mul_42 = None
        mul_43: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_31);  add_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:113 in gsw_dHdT, code: v02
        add_32: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_26, mul_43);  add_26 = mul_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:118 in gsw_dHdT, code: t106 = t24 * t105
        mul_44: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_7, add_32)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_4: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_3, mul_44);  sub_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:119 in gsw_dHdT, code: t107 = v44 + t2 + t3
        add_33: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_1, -1.941660213148725e-11);  mul_1 = None
        add_34: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_33, mul_2);  add_33 = mul_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:134 in gsw_dHdT, code: - 2.0 * t107 * t13
        mul_59: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_34, 2.0)
        mul_60: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_59, add_3);  mul_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_5: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_4, mul_60);  sub_4 = mul_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:120 in gsw_dHdT, code: t110 = v43 + t48
        add_35: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_21, -1.11901159287511e-10);  mul_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:135 in gsw_dHdT, code: - 2.0 * t110 * t29
        mul_61: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_35, 2.0)
        mul_62: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_61, add_9);  mul_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:129 in gsw_dHdT, code: 4.0 * v48 * t64 * t20
        sub_6: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_5, mul_62);  sub_5 = mul_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:128 in gsw_dHdT, code: + (
        mul_63: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_6, mul_12);  sub_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        add_41: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_39, mul_63);  add_39 = mul_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:122 in gsw_dHdT, code: t120 = 4.0 * t71 * t20 - t117 - 2.0 * t110 * t13
        mul_46: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_25, 4.0);  mul_25 = None
        mul_47: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_46, mul_12);  mul_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:121 in gsw_dHdT, code: t117 = t24 * t92
        mul_45: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_7, add_23);  add_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:122 in gsw_dHdT, code: t120 = 4.0 * t71 * t20 - t117 - 2.0 * t110 * t13
        sub: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_47, mul_45);  mul_47 = None
        mul_48: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_35, 2.0)
        mul_49: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_48, add_3);  mul_48 = None
        sub_1: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub, mul_49);  sub = mul_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:138 in gsw_dHdT, code: - t120 * t35 * t37
        mul_64: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_1, mul_18)
        mul_65: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_64, add_10);  mul_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:124 in gsw_dHdT, code: v38
        sub_7: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_41, mul_65);  add_41 = mul_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:197 in gsw_dHdT, code: + 0.5e4 * t123 * t20 * t134
        mul_133: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_7, 5000.0)
        mul_134: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_133, mul_12);  mul_133 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:141 in gsw_dHdT, code: t130 = p * (1.0 * v12 + 1.0 * t7 + 1.0 * t11 + t128)
        mul_67: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_4, 1.0);  mul_4 = None
        add_42: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_67, -0.02233269627352527);  mul_67 = None
        mul_68: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_7, 1.0);  mul_7 = None
        add_43: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_42, mul_68);  add_42 = mul_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:140 in gsw_dHdT, code: t128 = t19 * p
        mul_66: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_6, arg2_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:141 in gsw_dHdT, code: t130 = p * (1.0 * v12 + 1.0 * t7 + 1.0 * t11 + t128)
        add_44: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_43, mul_66);  add_43 = None
        mul_69: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_44);  add_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:142 in gsw_dHdT, code: t131 = 1.0 / t92
        reciprocal_2: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_23)
        mul_70: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1.0);  reciprocal_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:143 in gsw_dHdT, code: t133 = 1.0 + t130 * t131
        mul_71: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_69, mul_70)
        add_45: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_71, 1.0);  mul_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:144 in gsw_dHdT, code: t134 = torch.log(t133)
        log: "f64[204, 204, 26]" = torch.ops.aten.log.default(add_45)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:197 in gsw_dHdT, code: + 0.5e4 * t123 * t20 * t134
        mul_135: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_134, log);  mul_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_80: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_17, mul_135);  sub_17 = mul_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:145 in gsw_dHdT, code: t143 = v37 + ct * (v38 + t59) + sa * (v41 + v42 * ct) + t120 * t20
        add_46: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_23, -3.212746477974189e-07);  mul_23 = None
        mul_72: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_46);  add_46 = None
        add_47: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_72, -2.742185394906099e-05);  mul_72 = None
        mul_73: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.211426728363857e-10)
        add_48: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_73, -1.105097577149576e-07);  mul_73 = None
        mul_74: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_48);  add_48 = None
        add_49: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_47, mul_74);  add_47 = mul_74 = None
        mul_75: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_1, mul_12);  sub_1 = None
        add_50: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_49, mul_75);  add_49 = mul_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:198 in gsw_dHdT, code: - 0.5e4 * t143 * t35 * t134 * t37
        mul_136: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_50, 5000.0)
        mul_137: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_136, mul_18);  mul_136 = None
        mul_138: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_137, log);  mul_137 = log = None
        mul_139: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_138, add_10);  mul_138 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_18: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_80, mul_139);  add_80 = mul_139 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:199 in gsw_dHdT, code: + 0.5e4
        mul_140: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_50, 5000.0)
        mul_141: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_140, mul_12);  mul_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:202 in gsw_dHdT, code: * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131 - t130 / t156 * t105)
        mul_142: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_3, 2.0);  mul_3 = None
        add_81: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_142, -0.000343609007985188);  mul_142 = None
        mul_143: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_15, 1.0);  mul_15 = None
        add_82: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_81, mul_143);  add_81 = mul_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:146 in gsw_dHdT, code: t152 = t37 * p
        mul_76: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_10, arg2_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:202 in gsw_dHdT, code: * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131 - t130 / t156 * t105)
        add_83: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_82, mul_76);  add_82 = None
        mul_144: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_83);  arg2_1 = add_83 = None
        mul_145: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_144, mul_70);  mul_144 = mul_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:147 in gsw_dHdT, code: t156 = t92**2
        pow_3: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:202 in gsw_dHdT, code: * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131 - t130 / t156 * t105)
        div_1: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_69, pow_3);  mul_69 = pow_3 = None
        mul_146: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_1, add_32);  div_1 = None
        sub_19: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:199 in gsw_dHdT, code: + 0.5e4
        mul_147: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_141, sub_19);  mul_141 = sub_19 = None
        div_2: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_147, add_45);  mul_147 = add_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_84: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_18, div_2);  sub_18 = div_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:148 in gsw_dHdT, code: t165 = v25 * ct
        mul_77: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 6.743689325042773e-10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:149 in gsw_dHdT, code: t167 = ct * (v24 + t165)
        add_51: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_77, 1.119513357486743e-06)
        mul_78: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_51);  add_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:150 in gsw_dHdT, code: t169 = ct * (v23 + t167)
        add_52: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_78, -2.349607444135925e-05)
        mul_79: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_52);  add_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_85: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_79, 0.002775927747785646)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:209 in gsw_dHdT, code: + ct * (v23 + t167 + ct * (v24 + 2.0 * t165))
        add_86: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_78, -2.349607444135925e-05);  mul_78 = None
        mul_148: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_77, 2.0);  mul_77 = None
        add_87: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_148, 1.119513357486743e-06);  mul_148 = None
        mul_149: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_87);  add_87 = None
        add_88: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_86, mul_149);  add_86 = mul_149 = None
        mul_150: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_88);  add_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_89: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_85, mul_150);  add_85 = mul_150 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:151 in gsw_dHdT, code: t175 = v30 * ct
        mul_80: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, -1.811147201949891e-11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:152 in gsw_dHdT, code: t177 = ct * (v29 + t175)
        add_53: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_80, 9.527875081696435e-10)
        mul_81: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_53);  add_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:153 in gsw_dHdT, code: t179 = ct * (v28 + t177)
        add_54: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_81, 1.262937315098546e-07)
        mul_82: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_54);  add_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:212 in gsw_dHdT, code: v27
        add_90: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_82, -2.764306979894411e-05)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:214 in gsw_dHdT, code: + ct * (v28 + t177 + ct * (v29 + 2.0 * t175))
        add_91: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_81, 1.262937315098546e-07);  mul_81 = None
        mul_151: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_80, 2.0);  mul_80 = None
        add_92: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_151, 9.527875081696435e-10);  mul_151 = None
        mul_152: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_92);  add_92 = None
        add_93: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_91, mul_152);  add_91 = mul_152 = None
        mul_153: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_93);  add_93 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:212 in gsw_dHdT, code: v27
        add_94: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_90, mul_153);  add_90 = mul_153 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:154 in gsw_dHdT, code: t185 = v35 * ct
        mul_83: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 2.681097235569143e-12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:155 in gsw_dHdT, code: t187 = ct * (v34 + t185)
        add_55: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_83, -4.634182341116144e-11)
        mul_84: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_55);  add_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:156 in gsw_dHdT, code: t189 = ct * (v33 + t187)
        add_56: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_84, -7.672876869259043e-09)
        mul_85: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_56);  add_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:215 in gsw_dHdT, code: + t82 * (v32 + t189 + ct * (v33 + t187 + ct * (v34 + 2.0 * t185)))
        add_95: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_85, 3.801564588876298e-07)
        add_96: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_84, -7.672876869259043e-09);  mul_84 = None
        mul_154: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_83, 2.0);  mul_83 = None
        add_97: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_154, -4.634182341116144e-11);  mul_154 = None
        mul_155: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_97);  add_97 = None
        add_98: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_96, mul_155);  add_96 = mul_155 = None
        mul_156: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_98);  add_98 = None
        add_99: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_95, mul_156);  add_95 = mul_156 = None
        mul_157: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_99);  add_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:212 in gsw_dHdT, code: v27
        add_100: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_94, mul_157);  add_94 = mul_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:210 in gsw_dHdT, code: + sa
        mul_158: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_100);  add_100 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_101: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_89, mul_158);  add_89 = mul_158 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        mul_159: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_36, 2.0);  mul_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:157 in gsw_dHdT, code: t199 = t13 * t20
        mul_86: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_3, mul_12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        mul_160: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_159, mul_86);  mul_159 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:219 in gsw_dHdT, code: + 2.0 * t106 * t199
        mul_161: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_44, 2.0);  mul_44 = None
        mul_162: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_161, mul_86);  mul_161 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        add_102: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_160, mul_162);  mul_160 = mul_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:220 in gsw_dHdT, code: + 2.0 * t117 * t68
        mul_163: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_45, 2.0)
        mul_164: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_163, mul_24);  mul_163 = mul_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        add_103: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_102, mul_164);  add_102 = mul_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:221 in gsw_dHdT, code: - 2.0 * t117 * t13 * t35 * t37
        mul_165: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_45, 2.0)
        mul_166: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_165, add_3);  mul_165 = None
        mul_167: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_166, mul_18);  mul_166 = None
        mul_168: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_167, add_10);  mul_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        sub_20: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_103, mul_168);  add_103 = mul_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:222 in gsw_dHdT, code: - t107 * t92
        mul_169: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_34, add_23);  add_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        sub_21: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_20, mul_169);  sub_20 = mul_169 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:223 in gsw_dHdT, code: - t110 * t105
        mul_170: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_35, add_32)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:218 in gsw_dHdT, code: 2.0 * t93 * t199
        sub_22: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_21, mul_170);  sub_21 = mul_170 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:217 in gsw_dHdT, code: + (
        mul_171: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_22, mul_12);  sub_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        add_104: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_101, mul_171);  add_101 = mul_171 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:158 in gsw_dHdT, code: t217 = 2.0 * t117 * t199 - t110 * t92
        mul_87: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_45, 2.0);  mul_45 = None
        mul_88: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_87, mul_86);  mul_87 = mul_86 = None
        mul_89: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_35, add_23);  add_35 = None
        sub_8: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:226 in gsw_dHdT, code: - t217 * t35 * t37
        mul_172: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_8, mul_18)
        mul_173: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_172, add_10);  mul_172 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:207 in gsw_dHdT, code: v22
        sub_23: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_104, mul_173);  add_104 = mul_173 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        mul_174: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_23, add_6);  sub_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:161 in gsw_dHdT, code: + ct * (v22 + t169)
        add_57: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_79, 0.002775927747785646);  mul_79 = None
        mul_90: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_57);  add_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:160 in gsw_dHdT, code: v21
        add_58: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_90, 1.0);  mul_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:162 in gsw_dHdT, code: + sa * (v26 + ct * (v27 + t179) + v36 * sa + t82 * (v31 + ct * (v32 + t189)))
        add_59: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_82, -2.764306979894411e-05);  mul_82 = None
        mul_91: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_59);  add_59 = None
        add_60: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_91, -0.007521448093615448);  mul_91 = None
        mul_92: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 5.41932655114874e-06)
        add_61: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_60, mul_92);  add_60 = mul_92 = None
        add_62: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_85, 3.801564588876298e-07);  mul_85 = None
        mul_93: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_62);  arg0_1 = add_62 = None
        add_63: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_93, -3.303308871386421e-05);  mul_93 = None
        mul_94: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_63);  sqrt = add_63 = None
        add_64: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_61, mul_94);  add_61 = mul_94 = None
        mul_95: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_64);  arg1_1 = add_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:160 in gsw_dHdT, code: v21
        add_65: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_58, mul_95);  add_58 = mul_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:163 in gsw_dHdT, code: + t217 * t20
        mul_96: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_8, mul_12);  sub_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:160 in gsw_dHdT, code: v21
        add_66: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_65, mul_96);  add_65 = mul_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:229 in gsw_dHdT, code: + t234 * t37
        mul_175: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_66, add_10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        add_105: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:230 in gsw_dHdT, code: - t123 * t13
        mul_176: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_7, add_3);  sub_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        sub_24: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_105, mul_176);  add_105 = mul_176 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:231 in gsw_dHdT, code: - t143 * t29
        mul_177: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_50, add_9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:206 in gsw_dHdT, code: (
        sub_25: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_24, mul_177);  sub_24 = mul_177 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:204 in gsw_dHdT, code: + 0.5e4
        mul_178: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_25, 5000.0);  sub_25 = None
        mul_179: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_178, mul_12);  mul_178 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:165 in gsw_dHdT, code: t241 = t64 - t92 * t19
        mul_97: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_23, add_6)
        sub_9: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(pow_2, mul_97);  pow_2 = mul_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:166 in gsw_dHdT, code: t242 = torch.sqrt(t241)
        sqrt_1: "f64[204, 204, 26]" = torch.ops.aten.sqrt.default(sub_9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:167 in gsw_dHdT, code: t243 = 1.0 / t242
        reciprocal_3: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(sqrt_1)
        mul_98: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1.0);  reciprocal_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:173 in gsw_dHdT, code: t252 = 1.0 + 2.0 * t128 * t249
        mul_103: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_66, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:168 in gsw_dHdT, code: t244 = t4 + t8 + t12 - t242
        add_67: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_5, -0.011166348136762635)
        add_68: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_67, mul_8);  add_67 = None
        sub_10: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_68, sqrt_1);  add_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:169 in gsw_dHdT, code: t245 = 1.0 / t244
        reciprocal_4: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(sub_10)
        mul_99: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1.0);  reciprocal_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:172 in gsw_dHdT, code: t249 = t242 * t245 * t248
        mul_101: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt_1, mul_99)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:170 in gsw_dHdT, code: t247 = t4 + t8 + t12 + t242 + t128
        add_69: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_5, -0.011166348136762635);  mul_5 = None
        add_70: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_69, mul_8);  add_69 = mul_8 = None
        add_71: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_70, sqrt_1);  add_70 = None
        add_72: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_71, mul_66);  add_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:171 in gsw_dHdT, code: t248 = 1.0 / t247
        reciprocal_5: "f64[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_72)
        mul_100: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1.0);  reciprocal_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:172 in gsw_dHdT, code: t249 = t242 * t245 * t248
        mul_102: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_101, mul_100);  mul_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:173 in gsw_dHdT, code: t252 = 1.0 + 2.0 * t128 * t249
        mul_104: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_103, mul_102);  mul_103 = None
        add_73: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_104, 1.0);  mul_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:174 in gsw_dHdT, code: t253 = torch.log(t252)
        log_1: "f64[204, 204, 26]" = torch.ops.aten.log.default(add_73)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:175 in gsw_dHdT, code: t254 = t243 * t253
        mul_105: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_98, log_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:204 in gsw_dHdT, code: + 0.5e4
        mul_180: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_179, mul_105);  mul_179 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_106: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_84, mul_180);  add_84 = mul_180 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:176 in gsw_dHdT, code: t259 = t234 * t19 - t143 * t13
        mul_106: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_66, add_6);  add_66 = None
        mul_107: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_50, add_3);  add_50 = None
        sub_11: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:235 in gsw_dHdT, code: - 0.5e4 * t259 * t35 * t254 * t37
        mul_181: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_11, 5000.0)
        mul_182: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_181, mul_18);  mul_181 = mul_18 = None
        mul_183: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_182, mul_105);  mul_182 = mul_105 = None
        mul_184: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_183, add_10);  mul_183 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_26: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_106, mul_184);  add_106 = mul_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:177 in gsw_dHdT, code: t264 = t259 * t20
        mul_108: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_11, mul_12);  sub_11 = mul_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:236 in gsw_dHdT, code: - 0.25e4 * t264 / t242 / t241 * t253 * t272
        mul_185: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_108, 2500.0)
        div_3: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_185, sqrt_1);  mul_185 = None
        div_4: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(div_3, sub_9);  div_3 = sub_9 = None
        mul_186: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_4, log_1);  div_4 = log_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:178 in gsw_dHdT, code: t272 = 2.0 * t13 * t29 - t105 * t19 - t92 * t37
        mul_109: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_3, 2.0);  add_3 = None
        mul_110: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_109, add_9);  mul_109 = add_9 = None
        mul_111: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_32, add_6);  add_32 = add_6 = None
        sub_12: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        mul_112: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_23, add_10);  add_23 = add_10 = None
        sub_13: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_12, mul_112);  sub_12 = mul_112 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:236 in gsw_dHdT, code: - 0.25e4 * t264 / t242 / t241 * t253 * t272
        mul_187: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_186, sub_13);  mul_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        sub_27: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_26, mul_187);  sub_26 = mul_187 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:237 in gsw_dHdT, code: + 0.5e4
        mul_188: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_108, 5000.0);  mul_108 = None
        mul_189: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_188, mul_98);  mul_188 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        mul_190: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_76, 2.0)
        mul_191: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_190, mul_102);  mul_190 = mul_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:242 in gsw_dHdT, code: + t128 * t243 * t245 * t248 * t272
        mul_192: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_66, mul_98)
        mul_193: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_192, mul_99);  mul_192 = None
        mul_194: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_193, mul_100);  mul_193 = None
        mul_195: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_194, sub_13);  mul_194 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        add_107: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_191, mul_195);  mul_191 = mul_195 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:179 in gsw_dHdT, code: t282 = t128 * t242
        mul_113: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_66, sqrt_1);  mul_66 = sqrt_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:243 in gsw_dHdT, code: - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
        mul_196: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_113, 2.0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:180 in gsw_dHdT, code: t283 = t244**2
        pow_4: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(sub_10, 2);  sub_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:243 in gsw_dHdT, code: - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
        div_5: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_196, pow_4);  mul_196 = pow_4 = None
        mul_197: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_5, mul_100);  div_5 = mul_100 = None
        add_108: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_14, -0.000171804503992594)
        add_109: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_108, mul_16);  add_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:181 in gsw_dHdT, code: t287 = t243 * t272 / 2.0
        mul_114: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_98, sub_13);  mul_98 = sub_13 = None
        div: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_114, 2.0);  mul_114 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:243 in gsw_dHdT, code: - 2.0 * t282 / t283 * t248 * (t25 + t26 + t28 - t287)
        sub_28: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_109, div);  add_109 = None
        mul_198: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_197, sub_28);  mul_197 = sub_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        sub_29: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_107, mul_198);  add_107 = mul_198 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:244 in gsw_dHdT, code: - 2.0 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)
        mul_199: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_113, 2.0);  mul_113 = None
        mul_200: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_199, mul_99);  mul_199 = mul_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:182 in gsw_dHdT, code: t292 = t247**2
        pow_5: "f64[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_72, 2);  add_72 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:244 in gsw_dHdT, code: - 2.0 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)
        div_6: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_200, pow_5);  mul_200 = pow_5 = None
        add_110: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_14, -0.000171804503992594);  mul_14 = None
        add_111: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_110, mul_16);  add_110 = mul_16 = None
        add_112: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_111, div);  add_111 = div = None
        add_113: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(add_112, mul_76);  add_112 = mul_76 = None
        mul_201: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_6, add_113);  div_6 = add_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:241 in gsw_dHdT, code: 2.0 * t152 * t249
        sub_30: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_29, mul_201);  sub_29 = mul_201 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:237 in gsw_dHdT, code: + 0.5e4
        mul_202: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_189, sub_30);  mul_189 = sub_30 = None
        div_7: "f64[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_202, add_73);  mul_202 = add_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_equation_of_state/eos_pytorch.py:184 in gsw_dHdT, code: 0.1e5
        add_114: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_27, div_7);  sub_27 = div_7 = None
        return (add_114,)

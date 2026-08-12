"""
Standalone repro captured via capture_hook.
Label: torchbench_pyhpc_equation_of_state_infer
Pattern hash: 238e96847398
Shape hash: 2149da1a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 1, 26]", arg1_1: "bf16[204, 204, 26]", arg2_1: "bf16[204, 204, 26]"):
        # No stacktrace found for following nodes
        mul: "bf16[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 10000.0)
        mul_1: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, -1.8648264253656e-14)
        mul_2: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_1, 2.0)
        add: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_2, -1.941660213148725e-11)
        mul_3: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 1.119522344879478e-14)
        add_1: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add, mul_3);  add = None
        mul_4: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 3.726050720345733e-06)
        add_2: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_4, -0.000343609007985188)
        mul_5: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_2);  add_2 = None
        mul_6: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_5, 0.5)
        add_3: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_6, -0.011166348136762635)
        mul_7: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.876837219536232e-07)
        add_4: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_7, -0.0001806789763745328);  mul_7 = None
        mul_8: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_4);  add_4 = None
        mul_9: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_8, 0.5)
        add_5: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_3, mul_9);  add_3 = None
        mul_10: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_5, 1.2115804975093732e-16)
        mul_11: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, -1.061519070296458e-11)
        add_6: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_11, -1.988366587925593e-08)
        mul_12: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_6);  add_6 = None
        add_7: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_12, -3.087032500374211e-07);  mul_12 = None
        mul_13: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 1.55093272922008e-10)
        add_8: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_7, mul_13);  add_7 = mul_13 = None
        reciprocal: "bf16[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_8)
        mul_14: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal, 1.0);  reciprocal = None
        mul_15: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_10, mul_14);  mul_10 = None
        sub: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_1, mul_15);  add_1 = mul_15 = None
        mul_16: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.057902487546866e-17)
        add_9: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_16, -1.200507748551599e-15);  mul_16 = None
        mul_17: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_9, 2.0)
        mul_18: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_4, 1.0)
        add_10: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_18, -0.000171804503992594)
        mul_19: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 6.876837219536232e-07)
        mul_20: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_19, 0.5)
        add_11: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_10, mul_20);  add_10 = None
        mul_21: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_17, add_11);  mul_17 = None
        mul_22: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_21, mul_14);  mul_21 = None
        sub_1: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub, mul_22);  sub = mul_22 = None
        mul_23: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_9, add_5)
        mul_24: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_23, 2.0)
        pow_1: "bf16[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 2)
        reciprocal_1: "bf16[204, 204, 26]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul_25: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1.0);  reciprocal_1 = None
        mul_26: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_11, 2.0);  mul_11 = None
        add_12: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_26, -1.988366587925593e-08);  mul_26 = None
        mul_27: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_25, add_12)
        mul_28: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_24, mul_27);  mul_24 = None
        add_13: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_1, mul_28);  sub_1 = mul_28 = None
        mul_29: "bf16[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 3.028951243773433e-17)
        add_14: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_13, mul_29);  add_13 = mul_29 = None
        mul_30: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul, add_14);  mul = add_14 = None
        mul_31: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_30, mul_14);  mul_30 = None
        mul_32: "bf16[1, 1, 26]" = torch.ops.aten.mul.Tensor(arg0_1, 10000.0)
        add_15: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_1, -1.941660213148725e-11);  mul_1 = None
        add_16: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_15, mul_3);  add_15 = None
        mul_33: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_16);  add_16 = None
        add_17: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_33, -1.11901159287511e-10)
        mul_34: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_23, 2.0)
        mul_35: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_34, mul_14);  mul_34 = None
        sub_2: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_17, mul_35);  add_17 = mul_35 = None
        mul_36: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_9, 0.5)
        mul_37: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_36, arg0_1);  mul_36 = None
        add_18: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_2, mul_37);  sub_2 = mul_37 = None
        mul_38: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_32, add_18);  mul_32 = add_18 = None
        mul_39: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_38, mul_27);  mul_38 = None
        sub_3: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_31, mul_39);  mul_31 = mul_39 = None
        mul_40: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, -1.931012931541776e-12)
        add_19: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_40, 3.191413910561627e-09)
        mul_41: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_19);  add_19 = None
        add_20: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_41, -3.212746477974189e-07)
        mul_42: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_40, 2.0);  mul_40 = None
        add_21: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_42, 3.191413910561627e-09);  mul_42 = None
        mul_43: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_21);  add_21 = None
        add_22: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_20, mul_43);  add_20 = mul_43 = None
        mul_44: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 6.211426728363857e-10)
        add_23: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_22, mul_44);  add_22 = mul_44 = None
        pow_2: "bf16[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_5, 2)
        mul_45: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(pow_2, 2.4231609950187464e-16)
        mul_46: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_45, mul_14);  mul_45 = None
        mul_47: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_23, 8.0);  mul_23 = None
        mul_48: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_14, add_11)
        mul_49: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_47, mul_48);  mul_47 = None
        add_24: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_46, mul_49);  mul_46 = mul_49 = None
        mul_50: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_9, pow_2)
        mul_51: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_50, 4.0)
        mul_52: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_51, mul_27);  mul_51 = mul_27 = None
        sub_4: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_24, mul_52);  add_24 = mul_52 = None
        mul_53: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 0.001181805545074306)
        add_25: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_53, -0.03147759265588511)
        mul_54: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_25);  add_25 = None
        add_26: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_54, 2.839940833161907)
        mul_55: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_26);  add_26 = None
        add_27: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_55, 999.8420897506056);  mul_55 = None
        mul_56: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 0.0002327859407479162)
        add_28: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_56, -0.02986498947203215)
        mul_57: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_28);  add_28 = None
        add_29: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_57, -6.698001071123802);  mul_57 = None
        sqrt: "bf16[204, 204, 26]" = torch.ops.aten.sqrt.default(arg2_1)
        mul_58: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 1.645039373682922e-07)
        add_30: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_58, -1.426984671633621e-05)
        mul_59: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_30);  add_30 = None
        add_31: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_59, 0.00050954225738805)
        mul_60: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_31);  add_31 = None
        add_32: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_60, -0.0398882237896849);  mul_60 = None
        mul_61: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_32);  add_32 = None
        add_33: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_29, mul_61);  add_29 = mul_61 = None
        mul_62: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_33);  add_33 = None
        add_34: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_27, mul_62);  add_27 = mul_62 = None
        mul_63: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_34, 6.057902487546866e-17)
        sub_5: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_4, mul_63);  sub_4 = None
        add_35: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_54, 2.839940833161907);  mul_54 = None
        mul_64: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_53, 2.0);  mul_53 = None
        add_36: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_64, -0.03147759265588511);  mul_64 = None
        mul_65: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_36);  add_36 = None
        add_37: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_35, mul_65);  add_35 = mul_65 = None
        mul_66: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_56, 2.0);  mul_56 = None
        add_38: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_66, -0.02986498947203215);  mul_66 = None
        add_39: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_59, 0.00050954225738805);  mul_59 = None
        mul_67: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_58, 2.0);  mul_58 = None
        add_40: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_67, -1.426984671633621e-05);  mul_67 = None
        mul_68: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_40);  add_40 = None
        add_41: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_39, mul_68);  add_39 = mul_68 = None
        mul_69: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_41);  add_41 = None
        add_42: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_38, mul_69);  add_38 = mul_69 = None
        mul_70: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_42);  add_42 = None
        add_43: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_37, mul_70);  add_37 = mul_70 = None
        mul_71: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_9, add_43)
        sub_6: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_5, mul_71);  sub_5 = None
        add_44: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_2, -1.941660213148725e-11);  mul_2 = None
        add_45: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_44, mul_3);  add_44 = mul_3 = None
        mul_72: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_45, 2.0)
        mul_73: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_72, add_5);  mul_72 = None
        sub_7: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_6, mul_73);  sub_6 = mul_73 = None
        add_46: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_33, -1.11901159287511e-10);  mul_33 = None
        mul_74: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_46, 2.0)
        mul_75: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_74, add_11);  mul_74 = None
        sub_8: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_7, mul_75);  sub_7 = mul_75 = None
        mul_76: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_8, mul_14);  sub_8 = None
        add_47: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_23, mul_76);  add_23 = mul_76 = None
        mul_77: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_50, 4.0);  mul_50 = None
        mul_78: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_77, mul_14);  mul_77 = None
        mul_79: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_9, add_34);  add_9 = None
        sub_9: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_78, mul_79);  mul_78 = None
        mul_80: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_46, 2.0)
        mul_81: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_80, add_5);  mul_80 = None
        sub_10: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_9, mul_81);  sub_9 = mul_81 = None
        mul_82: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_10, mul_25)
        mul_83: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_82, add_12);  mul_82 = None
        sub_11: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_47, mul_83);  add_47 = mul_83 = None
        mul_84: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_11, 5000.0)
        mul_85: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_84, mul_14);  mul_84 = None
        mul_86: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_5, 1.0);  mul_5 = None
        add_48: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_86, -0.02233269627352527);  mul_86 = None
        mul_87: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_8, 1.0);  mul_8 = None
        add_49: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_48, mul_87);  add_48 = mul_87 = None
        mul_88: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_8, arg0_1)
        add_50: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_49, mul_88);  add_49 = None
        mul_89: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_50);  add_50 = None
        reciprocal_2: "bf16[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_34)
        mul_90: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1.0);  reciprocal_2 = None
        mul_91: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_89, mul_90)
        add_51: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_91, 1.0);  mul_91 = None
        log: "bf16[204, 204, 26]" = torch.ops.aten.log.default(add_51)
        mul_92: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_85, log);  mul_85 = None
        add_52: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_3, mul_92);  sub_3 = mul_92 = None
        add_53: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_41, -3.212746477974189e-07);  mul_41 = None
        mul_93: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_53);  add_53 = None
        add_54: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_93, -2.742185394906099e-05);  mul_93 = None
        mul_94: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.211426728363857e-10)
        add_55: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_94, -1.105097577149576e-07);  mul_94 = None
        mul_95: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_55);  add_55 = None
        add_56: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_54, mul_95);  add_54 = mul_95 = None
        mul_96: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_10, mul_14);  sub_10 = None
        add_57: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_56, mul_96);  add_56 = mul_96 = None
        mul_97: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_57, 5000.0)
        mul_98: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_97, mul_25);  mul_97 = None
        mul_99: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_98, log);  mul_98 = log = None
        mul_100: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_99, add_12);  mul_99 = None
        sub_12: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_52, mul_100);  add_52 = mul_100 = None
        mul_101: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_57, 5000.0)
        mul_102: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_101, mul_14);  mul_101 = None
        mul_103: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_4, 2.0);  mul_4 = None
        add_58: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_103, -0.000343609007985188);  mul_103 = None
        mul_104: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_19, 1.0);  mul_19 = None
        add_59: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_58, mul_104);  add_58 = mul_104 = None
        mul_105: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_12, arg0_1)
        add_60: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_59, mul_105);  add_59 = None
        mul_106: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg0_1, add_60);  arg0_1 = add_60 = None
        mul_107: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_106, mul_90);  mul_106 = mul_90 = None
        pow_3: "bf16[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        div: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_89, pow_3);  mul_89 = pow_3 = None
        mul_108: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(div, add_43);  div = None
        sub_13: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None
        mul_109: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_102, sub_13);  mul_102 = sub_13 = None
        div_1: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_109, add_51);  mul_109 = add_51 = None
        add_61: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_12, div_1);  sub_12 = div_1 = None
        mul_110: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 6.743689325042773e-10)
        add_62: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_110, 1.119513357486743e-06)
        mul_111: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_62);  add_62 = None
        add_63: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_111, -2.349607444135925e-05)
        mul_112: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_63);  add_63 = None
        add_64: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_112, 0.002775927747785646)
        add_65: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_111, -2.349607444135925e-05);  mul_111 = None
        mul_113: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_110, 2.0);  mul_110 = None
        add_66: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_113, 1.119513357486743e-06);  mul_113 = None
        mul_114: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_66);  add_66 = None
        add_67: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_65, mul_114);  add_65 = mul_114 = None
        mul_115: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_67);  add_67 = None
        add_68: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_64, mul_115);  add_64 = mul_115 = None
        mul_116: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, -1.811147201949891e-11)
        add_69: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_116, 9.527875081696435e-10)
        mul_117: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_69);  add_69 = None
        add_70: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_117, 1.262937315098546e-07)
        mul_118: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_70);  add_70 = None
        add_71: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_118, -2.764306979894411e-05)
        add_72: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_117, 1.262937315098546e-07);  mul_117 = None
        mul_119: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_116, 2.0);  mul_116 = None
        add_73: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_119, 9.527875081696435e-10);  mul_119 = None
        mul_120: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_73);  add_73 = None
        add_74: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_72, mul_120);  add_72 = mul_120 = None
        mul_121: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_74);  add_74 = None
        add_75: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_71, mul_121);  add_71 = mul_121 = None
        mul_122: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, 2.681097235569143e-12)
        add_76: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_122, -4.634182341116144e-11)
        mul_123: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_76);  add_76 = None
        add_77: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_123, -7.672876869259043e-09)
        mul_124: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_77);  add_77 = None
        add_78: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_124, 3.801564588876298e-07)
        add_79: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_123, -7.672876869259043e-09);  mul_123 = None
        mul_125: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_122, 2.0);  mul_122 = None
        add_80: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_125, -4.634182341116144e-11);  mul_125 = None
        mul_126: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_80);  add_80 = None
        add_81: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_79, mul_126);  add_79 = mul_126 = None
        mul_127: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_81);  add_81 = None
        add_82: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_78, mul_127);  add_78 = mul_127 = None
        mul_128: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_82);  add_82 = None
        add_83: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_75, mul_128);  add_75 = mul_128 = None
        mul_129: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_83);  add_83 = None
        add_84: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_68, mul_129);  add_68 = mul_129 = None
        mul_130: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_63, 2.0);  mul_63 = None
        mul_131: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_5, mul_14)
        mul_132: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_130, mul_131);  mul_130 = None
        mul_133: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_71, 2.0);  mul_71 = None
        mul_134: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_133, mul_131);  mul_133 = None
        add_85: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_132, mul_134);  mul_132 = mul_134 = None
        mul_135: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_79, 2.0)
        mul_136: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_135, mul_48);  mul_135 = mul_48 = None
        add_86: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_85, mul_136);  add_85 = mul_136 = None
        mul_137: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_79, 2.0)
        mul_138: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_137, add_5);  mul_137 = None
        mul_139: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_138, mul_25);  mul_138 = None
        mul_140: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_139, add_12);  mul_139 = None
        sub_14: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_86, mul_140);  add_86 = mul_140 = None
        mul_141: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_45, add_34);  add_45 = None
        sub_15: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_14, mul_141);  sub_14 = mul_141 = None
        mul_142: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_46, add_43)
        sub_16: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_15, mul_142);  sub_15 = mul_142 = None
        mul_143: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_16, mul_14);  sub_16 = None
        add_87: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_84, mul_143);  add_84 = mul_143 = None
        mul_144: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_79, 2.0);  mul_79 = None
        mul_145: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_144, mul_131);  mul_144 = mul_131 = None
        mul_146: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_46, add_34);  add_46 = None
        sub_17: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
        mul_147: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_17, mul_25)
        mul_148: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_147, add_12);  mul_147 = None
        sub_18: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_87, mul_148);  add_87 = mul_148 = None
        mul_149: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_18, add_8);  sub_18 = None
        add_88: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_112, 0.002775927747785646);  mul_112 = None
        mul_150: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_88);  add_88 = None
        add_89: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_150, 1.0);  mul_150 = None
        add_90: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_118, -2.764306979894411e-05);  mul_118 = None
        mul_151: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_90);  add_90 = None
        add_91: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_151, -0.007521448093615448);  mul_151 = None
        mul_152: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, 5.41932655114874e-06)
        add_92: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_91, mul_152);  add_91 = mul_152 = None
        add_93: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_124, 3.801564588876298e-07);  mul_124 = None
        mul_153: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg1_1, add_93);  arg1_1 = add_93 = None
        add_94: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_153, -3.303308871386421e-05);  mul_153 = None
        mul_154: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt, add_94);  sqrt = add_94 = None
        add_95: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_92, mul_154);  add_92 = mul_154 = None
        mul_155: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg2_1, add_95);  arg2_1 = add_95 = None
        add_96: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_89, mul_155);  add_89 = mul_155 = None
        mul_156: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_17, mul_14);  sub_17 = None
        add_97: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_96, mul_156);  add_96 = mul_156 = None
        mul_157: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_97, add_12)
        add_98: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_149, mul_157);  mul_149 = mul_157 = None
        mul_158: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_11, add_5);  sub_11 = None
        sub_19: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_98, mul_158);  add_98 = mul_158 = None
        mul_159: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_57, add_11)
        sub_20: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_19, mul_159);  sub_19 = mul_159 = None
        mul_160: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_20, 5000.0);  sub_20 = None
        mul_161: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_160, mul_14);  mul_160 = None
        mul_162: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_34, add_8)
        sub_21: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(pow_2, mul_162);  pow_2 = mul_162 = None
        sqrt_1: "bf16[204, 204, 26]" = torch.ops.aten.sqrt.default(sub_21)
        reciprocal_3: "bf16[204, 204, 26]" = torch.ops.aten.reciprocal.default(sqrt_1)
        mul_163: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1.0);  reciprocal_3 = None
        mul_164: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_88, 2.0)
        add_99: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_6, -0.011166348136762635)
        add_100: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_99, mul_9);  add_99 = None
        sub_22: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_100, sqrt_1);  add_100 = None
        reciprocal_4: "bf16[204, 204, 26]" = torch.ops.aten.reciprocal.default(sub_22)
        mul_165: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1.0);  reciprocal_4 = None
        mul_166: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sqrt_1, mul_165)
        add_101: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_6, -0.011166348136762635);  mul_6 = None
        add_102: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_101, mul_9);  add_101 = mul_9 = None
        add_103: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_102, sqrt_1);  add_102 = None
        add_104: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_103, mul_88);  add_103 = None
        reciprocal_5: "bf16[204, 204, 26]" = torch.ops.aten.reciprocal.default(add_104)
        mul_167: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1.0);  reciprocal_5 = None
        mul_168: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_166, mul_167);  mul_166 = None
        mul_169: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_164, mul_168);  mul_164 = None
        add_105: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_169, 1.0);  mul_169 = None
        log_1: "bf16[204, 204, 26]" = torch.ops.aten.log.default(add_105)
        mul_170: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_163, log_1)
        mul_171: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_161, mul_170);  mul_161 = None
        add_106: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_61, mul_171);  add_61 = mul_171 = None
        mul_172: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_97, add_8);  add_97 = None
        mul_173: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_57, add_5);  add_57 = None
        sub_23: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        mul_174: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_23, 5000.0)
        mul_175: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_174, mul_25);  mul_174 = mul_25 = None
        mul_176: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_175, mul_170);  mul_175 = mul_170 = None
        mul_177: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_176, add_12);  mul_176 = None
        sub_24: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_106, mul_177);  add_106 = mul_177 = None
        mul_178: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_23, mul_14);  sub_23 = mul_14 = None
        mul_179: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_178, 2500.0)
        div_2: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_179, sqrt_1);  mul_179 = None
        div_3: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(div_2, sub_21);  div_2 = sub_21 = None
        mul_180: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_3, log_1);  div_3 = log_1 = None
        mul_181: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_5, 2.0);  add_5 = None
        mul_182: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_181, add_11);  mul_181 = add_11 = None
        mul_183: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_43, add_8);  add_43 = add_8 = None
        sub_25: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        mul_184: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(add_34, add_12);  add_34 = add_12 = None
        sub_26: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_25, mul_184);  sub_25 = mul_184 = None
        mul_185: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_180, sub_26);  mul_180 = None
        sub_27: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_24, mul_185);  sub_24 = mul_185 = None
        mul_186: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_178, 5000.0);  mul_178 = None
        mul_187: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_186, mul_163);  mul_186 = None
        mul_188: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_105, 2.0)
        mul_189: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_188, mul_168);  mul_188 = mul_168 = None
        mul_190: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_88, mul_163)
        mul_191: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_190, mul_165);  mul_190 = None
        mul_192: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_191, mul_167);  mul_191 = None
        mul_193: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_192, sub_26);  mul_192 = None
        add_107: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_189, mul_193);  mul_189 = mul_193 = None
        mul_194: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_88, sqrt_1);  mul_88 = sqrt_1 = None
        mul_195: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_194, 2.0)
        pow_4: "bf16[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(sub_22, 2);  sub_22 = None
        div_4: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_195, pow_4);  mul_195 = pow_4 = None
        mul_196: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_4, mul_167);  div_4 = mul_167 = None
        add_108: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_18, -0.000171804503992594)
        add_109: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_108, mul_20);  add_108 = None
        mul_197: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_163, sub_26);  mul_163 = sub_26 = None
        div_5: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_197, 2.0);  mul_197 = None
        sub_28: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_109, div_5);  add_109 = None
        mul_198: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_196, sub_28);  mul_196 = sub_28 = None
        sub_29: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(add_107, mul_198);  add_107 = mul_198 = None
        mul_199: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_194, 2.0);  mul_194 = None
        mul_200: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_199, mul_165);  mul_199 = mul_165 = None
        pow_5: "bf16[204, 204, 26]" = torch.ops.aten.pow.Tensor_Scalar(add_104, 2);  add_104 = None
        div_6: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_200, pow_5);  mul_200 = pow_5 = None
        add_110: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_18, -0.000171804503992594);  mul_18 = None
        add_111: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_110, mul_20);  add_110 = mul_20 = None
        add_112: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_111, div_5);  add_111 = div_5 = None
        add_113: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(add_112, mul_105);  add_112 = mul_105 = None
        mul_201: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(div_6, add_113);  div_6 = add_113 = None
        sub_30: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(sub_29, mul_201);  sub_29 = mul_201 = None
        mul_202: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(mul_187, sub_30);  mul_187 = sub_30 = None
        div_7: "bf16[204, 204, 26]" = torch.ops.aten.div.Tensor(mul_202, add_105);  mul_202 = add_105 = None
        add_114: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(sub_27, div_7);  sub_27 = div_7 = None
        return add_114



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""
Oracle for pointwise_a2382a85ee99.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle emits the complete
372-op f64 equation-of-state pointwise graph as a hand-authored flat Triton
kernel with a small tile chosen for the register pressure of the whole
expression, whereas Inductor's generic pointwise lowering already produces a
slightly faster kernel under the requested tuning configs; Inductor cannot be
improved by a missing fusion-pattern change here because the measured floor is
set by the same fp64 load/store and transcendental-throughput work that both
kernels must perform; the fix is BANDWIDTH_BOUND: no new scheduler pattern is
needed for this repro unless a future codegen pass can reduce the required f64
math itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ARG2_SHAPE = (1, 1, 26)
OUT_SHAPE = (204, 204, 26)
OUT_STRIDE = (5304, 26, 1)
N_ELEMS = 204 * 204 * 26
K = 26
BLOCK_N = 32

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _equation_pointwise_kernel(
        arg2_ptr,
        arg0_ptr,
        arg1_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        arg2_offsets = offsets % 26
        arg2_1 = tl.load(arg2_ptr + arg2_offsets, mask=mask, other=0.0).to(tl.float64)
        arg0_1 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float64)
        arg1_1 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float64)

        mul_tensor = (arg2_1 * 10000.0)
        mul_tensor_1 = (arg0_1 * -1.8648264253656e-14)
        mul_tensor_2 = (mul_tensor_1 * 2.0)
        add_tensor = (mul_tensor_2 + -1.941660213148725e-11)
        mul_tensor_3 = (arg1_1 * 1.119522344879478e-14)
        add_tensor_1 = (add_tensor + mul_tensor_3)
        mul_tensor_4 = (arg0_1 * 3.726050720345733e-06)
        add_tensor_2 = (mul_tensor_4 + -0.000343609007985188)
        mul_tensor_5 = (arg0_1 * add_tensor_2)
        mul_tensor_6 = (mul_tensor_5 * 0.5)
        add_tensor_3 = (mul_tensor_6 + -0.011166348136762635)
        mul_tensor_7 = (arg0_1 * 6.876837219536232e-07)
        add_tensor_4 = (mul_tensor_7 + -0.0001806789763745328)
        mul_tensor_8 = (arg1_1 * add_tensor_4)
        mul_tensor_9 = (mul_tensor_8 * 0.5)
        add_tensor_5 = (add_tensor_3 + mul_tensor_9)
        mul_tensor_10 = (add_tensor_5 * 1.2115804975093732e-16)
        mul_tensor_11 = (arg0_1 * -1.061519070296458e-11)
        add_tensor_6 = (mul_tensor_11 + -1.988366587925593e-08)
        mul_tensor_12 = (arg0_1 * add_tensor_6)
        add_tensor_7 = (mul_tensor_12 + -3.087032500374211e-07)
        mul_tensor_13 = (arg1_1 * 1.55093272922008e-10)
        add_tensor_8 = (add_tensor_7 + mul_tensor_13)
        reciprocal_default = (1.0 / add_tensor_8)
        mul_tensor_14 = (reciprocal_default * 1.0)
        mul_tensor_15 = (mul_tensor_10 * mul_tensor_14)
        sub_tensor = (add_tensor_1 - mul_tensor_15)
        mul_tensor_16 = (arg0_1 * 6.057902487546866e-17)
        add_tensor_9 = (mul_tensor_16 + -1.200507748551599e-15)
        mul_tensor_17 = (add_tensor_9 * 2.0)
        mul_tensor_18 = (mul_tensor_4 * 1.0)
        add_tensor_10 = (mul_tensor_18 + -0.000171804503992594)
        mul_tensor_19 = (arg1_1 * 6.876837219536232e-07)
        mul_tensor_20 = (mul_tensor_19 * 0.5)
        add_tensor_11 = (add_tensor_10 + mul_tensor_20)
        mul_tensor_21 = (mul_tensor_17 * add_tensor_11)
        mul_tensor_22 = (mul_tensor_21 * mul_tensor_14)
        sub_tensor_1 = (sub_tensor - mul_tensor_22)
        mul_tensor_23 = (add_tensor_9 * add_tensor_5)
        mul_tensor_24 = (mul_tensor_23 * 2.0)
        pow_tensor_scalar = (add_tensor_8 * add_tensor_8)
        reciprocal_default_1 = (1.0 / pow_tensor_scalar)
        mul_tensor_25 = (reciprocal_default_1 * 1.0)
        mul_tensor_26 = (mul_tensor_11 * 2.0)
        add_tensor_12 = (mul_tensor_26 + -1.988366587925593e-08)
        mul_tensor_27 = (mul_tensor_25 * add_tensor_12)
        mul_tensor_28 = (mul_tensor_24 * mul_tensor_27)
        add_tensor_13 = (sub_tensor_1 + mul_tensor_28)
        mul_tensor_29 = (arg2_1 * 3.028951243773433e-17)
        add_tensor_14 = (add_tensor_13 + mul_tensor_29)
        mul_tensor_30 = (mul_tensor * add_tensor_14)
        mul_tensor_31 = (mul_tensor_30 * mul_tensor_14)
        mul_tensor_32 = (arg2_1 * 10000.0)
        add_tensor_15 = (mul_tensor_1 + -1.941660213148725e-11)
        add_tensor_16 = (add_tensor_15 + mul_tensor_3)
        mul_tensor_33 = (arg0_1 * add_tensor_16)
        add_tensor_17 = (mul_tensor_33 + -1.11901159287511e-10)
        mul_tensor_34 = (mul_tensor_23 * 2.0)
        mul_tensor_35 = (mul_tensor_34 * mul_tensor_14)
        sub_tensor_2 = (add_tensor_17 - mul_tensor_35)
        mul_tensor_36 = (add_tensor_9 * 0.5)
        mul_tensor_37 = (mul_tensor_36 * arg2_1)
        add_tensor_18 = (sub_tensor_2 + mul_tensor_37)
        mul_tensor_38 = (mul_tensor_32 * add_tensor_18)
        mul_tensor_39 = (mul_tensor_38 * mul_tensor_27)
        sub_tensor_3 = (mul_tensor_31 - mul_tensor_39)
        mul_tensor_40 = (arg0_1 * -1.931012931541776e-12)
        add_tensor_19 = (mul_tensor_40 + 3.191413910561627e-09)
        mul_tensor_41 = (arg0_1 * add_tensor_19)
        add_tensor_20 = (mul_tensor_41 + -3.212746477974189e-07)
        mul_tensor_42 = (mul_tensor_40 * 2.0)
        add_tensor_21 = (mul_tensor_42 + 3.191413910561627e-09)
        mul_tensor_43 = (arg0_1 * add_tensor_21)
        add_tensor_22 = (add_tensor_20 + mul_tensor_43)
        mul_tensor_44 = (arg1_1 * 6.211426728363857e-10)
        add_tensor_23 = (add_tensor_22 + mul_tensor_44)
        pow_tensor_scalar_1 = (add_tensor_5 * add_tensor_5)
        mul_tensor_45 = (pow_tensor_scalar_1 * 2.4231609950187464e-16)
        mul_tensor_46 = (mul_tensor_45 * mul_tensor_14)
        mul_tensor_47 = (mul_tensor_23 * 8.0)
        mul_tensor_48 = (mul_tensor_14 * add_tensor_11)
        mul_tensor_49 = (mul_tensor_47 * mul_tensor_48)
        add_tensor_24 = (mul_tensor_46 + mul_tensor_49)
        mul_tensor_50 = (add_tensor_9 * pow_tensor_scalar_1)
        mul_tensor_51 = (mul_tensor_50 * 4.0)
        mul_tensor_52 = (mul_tensor_51 * mul_tensor_27)
        sub_tensor_4 = (add_tensor_24 - mul_tensor_52)
        mul_tensor_53 = (arg0_1 * 0.001181805545074306)
        add_tensor_25 = (mul_tensor_53 + -0.03147759265588511)
        mul_tensor_54 = (arg0_1 * add_tensor_25)
        add_tensor_26 = (mul_tensor_54 + 2.839940833161907)
        mul_tensor_55 = (arg0_1 * add_tensor_26)
        add_tensor_27 = (mul_tensor_55 + 999.8420897506056)
        mul_tensor_56 = (arg0_1 * 0.0002327859407479162)
        add_tensor_28 = (mul_tensor_56 + -0.02986498947203215)
        mul_tensor_57 = (arg0_1 * add_tensor_28)
        add_tensor_29 = (mul_tensor_57 + -6.698001071123802)
        sqrt_default = tl.sqrt(arg1_1)
        mul_tensor_58 = (arg0_1 * 1.645039373682922e-07)
        add_tensor_30 = (mul_tensor_58 + -1.426984671633621e-05)
        mul_tensor_59 = (arg0_1 * add_tensor_30)
        add_tensor_31 = (mul_tensor_59 + 0.00050954225738805)
        mul_tensor_60 = (arg0_1 * add_tensor_31)
        add_tensor_32 = (mul_tensor_60 + -0.0398882237896849)
        mul_tensor_61 = (sqrt_default * add_tensor_32)
        add_tensor_33 = (add_tensor_29 + mul_tensor_61)
        mul_tensor_62 = (arg1_1 * add_tensor_33)
        add_tensor_34 = (add_tensor_27 + mul_tensor_62)
        mul_tensor_63 = (add_tensor_34 * 6.057902487546866e-17)
        sub_tensor_5 = (sub_tensor_4 - mul_tensor_63)
        add_tensor_35 = (mul_tensor_54 + 2.839940833161907)
        mul_tensor_64 = (mul_tensor_53 * 2.0)
        add_tensor_36 = (mul_tensor_64 + -0.03147759265588511)
        mul_tensor_65 = (arg0_1 * add_tensor_36)
        add_tensor_37 = (add_tensor_35 + mul_tensor_65)
        mul_tensor_66 = (mul_tensor_56 * 2.0)
        add_tensor_38 = (mul_tensor_66 + -0.02986498947203215)
        add_tensor_39 = (mul_tensor_59 + 0.00050954225738805)
        mul_tensor_67 = (mul_tensor_58 * 2.0)
        add_tensor_40 = (mul_tensor_67 + -1.426984671633621e-05)
        mul_tensor_68 = (arg0_1 * add_tensor_40)
        add_tensor_41 = (add_tensor_39 + mul_tensor_68)
        mul_tensor_69 = (sqrt_default * add_tensor_41)
        add_tensor_42 = (add_tensor_38 + mul_tensor_69)
        mul_tensor_70 = (arg1_1 * add_tensor_42)
        add_tensor_43 = (add_tensor_37 + mul_tensor_70)
        mul_tensor_71 = (add_tensor_9 * add_tensor_43)
        sub_tensor_6 = (sub_tensor_5 - mul_tensor_71)
        add_tensor_44 = (mul_tensor_2 + -1.941660213148725e-11)
        add_tensor_45 = (add_tensor_44 + mul_tensor_3)
        mul_tensor_72 = (add_tensor_45 * 2.0)
        mul_tensor_73 = (mul_tensor_72 * add_tensor_5)
        sub_tensor_7 = (sub_tensor_6 - mul_tensor_73)
        add_tensor_46 = (mul_tensor_33 + -1.11901159287511e-10)
        mul_tensor_74 = (add_tensor_46 * 2.0)
        mul_tensor_75 = (mul_tensor_74 * add_tensor_11)
        sub_tensor_8 = (sub_tensor_7 - mul_tensor_75)
        mul_tensor_76 = (sub_tensor_8 * mul_tensor_14)
        add_tensor_47 = (add_tensor_23 + mul_tensor_76)
        mul_tensor_77 = (mul_tensor_50 * 4.0)
        mul_tensor_78 = (mul_tensor_77 * mul_tensor_14)
        mul_tensor_79 = (add_tensor_9 * add_tensor_34)
        sub_tensor_9 = (mul_tensor_78 - mul_tensor_79)
        mul_tensor_80 = (add_tensor_46 * 2.0)
        mul_tensor_81 = (mul_tensor_80 * add_tensor_5)
        sub_tensor_10 = (sub_tensor_9 - mul_tensor_81)
        mul_tensor_82 = (sub_tensor_10 * mul_tensor_25)
        mul_tensor_83 = (mul_tensor_82 * add_tensor_12)
        sub_tensor_11 = (add_tensor_47 - mul_tensor_83)
        mul_tensor_84 = (sub_tensor_11 * 5000.0)
        mul_tensor_85 = (mul_tensor_84 * mul_tensor_14)
        mul_tensor_86 = (mul_tensor_5 * 1.0)
        add_tensor_48 = (mul_tensor_86 + -0.02233269627352527)
        mul_tensor_87 = (mul_tensor_8 * 1.0)
        add_tensor_49 = (add_tensor_48 + mul_tensor_87)
        mul_tensor_88 = (add_tensor_8 * arg2_1)
        add_tensor_50 = (add_tensor_49 + mul_tensor_88)
        mul_tensor_89 = (arg2_1 * add_tensor_50)
        reciprocal_default_2 = (1.0 / add_tensor_34)
        mul_tensor_90 = (reciprocal_default_2 * 1.0)
        mul_tensor_91 = (mul_tensor_89 * mul_tensor_90)
        add_tensor_51 = (mul_tensor_91 + 1.0)
        log_default = tl.log(add_tensor_51)
        mul_tensor_92 = (mul_tensor_85 * log_default)
        add_tensor_52 = (sub_tensor_3 + mul_tensor_92)
        add_tensor_53 = (mul_tensor_41 + -3.212746477974189e-07)
        mul_tensor_93 = (arg0_1 * add_tensor_53)
        add_tensor_54 = (mul_tensor_93 + -2.742185394906099e-05)
        mul_tensor_94 = (arg0_1 * 6.211426728363857e-10)
        add_tensor_55 = (mul_tensor_94 + -1.105097577149576e-07)
        mul_tensor_95 = (arg1_1 * add_tensor_55)
        add_tensor_56 = (add_tensor_54 + mul_tensor_95)
        mul_tensor_96 = (sub_tensor_10 * mul_tensor_14)
        add_tensor_57 = (add_tensor_56 + mul_tensor_96)
        mul_tensor_97 = (add_tensor_57 * 5000.0)
        mul_tensor_98 = (mul_tensor_97 * mul_tensor_25)
        mul_tensor_99 = (mul_tensor_98 * log_default)
        mul_tensor_100 = (mul_tensor_99 * add_tensor_12)
        sub_tensor_12 = (add_tensor_52 - mul_tensor_100)
        mul_tensor_101 = (add_tensor_57 * 5000.0)
        mul_tensor_102 = (mul_tensor_101 * mul_tensor_14)
        mul_tensor_103 = (mul_tensor_4 * 2.0)
        add_tensor_58 = (mul_tensor_103 + -0.000343609007985188)
        mul_tensor_104 = (mul_tensor_19 * 1.0)
        add_tensor_59 = (add_tensor_58 + mul_tensor_104)
        mul_tensor_105 = (add_tensor_12 * arg2_1)
        add_tensor_60 = (add_tensor_59 + mul_tensor_105)
        mul_tensor_106 = (arg2_1 * add_tensor_60)
        mul_tensor_107 = (mul_tensor_106 * mul_tensor_90)
        pow_tensor_scalar_2 = (add_tensor_34 * add_tensor_34)
        div_tensor = (mul_tensor_89 / pow_tensor_scalar_2)
        mul_tensor_108 = (div_tensor * add_tensor_43)
        sub_tensor_13 = (mul_tensor_107 - mul_tensor_108)
        mul_tensor_109 = (mul_tensor_102 * sub_tensor_13)
        div_tensor_1 = (mul_tensor_109 / add_tensor_51)
        add_tensor_61 = (sub_tensor_12 + div_tensor_1)
        mul_tensor_110 = (arg0_1 * 6.743689325042773e-10)
        add_tensor_62 = (mul_tensor_110 + 1.119513357486743e-06)
        mul_tensor_111 = (arg0_1 * add_tensor_62)
        add_tensor_63 = (mul_tensor_111 + -2.349607444135925e-05)
        mul_tensor_112 = (arg0_1 * add_tensor_63)
        add_tensor_64 = (mul_tensor_112 + 0.002775927747785646)
        add_tensor_65 = (mul_tensor_111 + -2.349607444135925e-05)
        mul_tensor_113 = (mul_tensor_110 * 2.0)
        add_tensor_66 = (mul_tensor_113 + 1.119513357486743e-06)
        mul_tensor_114 = (arg0_1 * add_tensor_66)
        add_tensor_67 = (add_tensor_65 + mul_tensor_114)
        mul_tensor_115 = (arg0_1 * add_tensor_67)
        add_tensor_68 = (add_tensor_64 + mul_tensor_115)
        mul_tensor_116 = (arg0_1 * -1.811147201949891e-11)
        add_tensor_69 = (mul_tensor_116 + 9.527875081696435e-10)
        mul_tensor_117 = (arg0_1 * add_tensor_69)
        add_tensor_70 = (mul_tensor_117 + 1.262937315098546e-07)
        mul_tensor_118 = (arg0_1 * add_tensor_70)
        add_tensor_71 = (mul_tensor_118 + -2.764306979894411e-05)
        add_tensor_72 = (mul_tensor_117 + 1.262937315098546e-07)
        mul_tensor_119 = (mul_tensor_116 * 2.0)
        add_tensor_73 = (mul_tensor_119 + 9.527875081696435e-10)
        mul_tensor_120 = (arg0_1 * add_tensor_73)
        add_tensor_74 = (add_tensor_72 + mul_tensor_120)
        mul_tensor_121 = (arg0_1 * add_tensor_74)
        add_tensor_75 = (add_tensor_71 + mul_tensor_121)
        mul_tensor_122 = (arg0_1 * 2.681097235569143e-12)
        add_tensor_76 = (mul_tensor_122 + -4.634182341116144e-11)
        mul_tensor_123 = (arg0_1 * add_tensor_76)
        add_tensor_77 = (mul_tensor_123 + -7.672876869259043e-09)
        mul_tensor_124 = (arg0_1 * add_tensor_77)
        add_tensor_78 = (mul_tensor_124 + 3.801564588876298e-07)
        add_tensor_79 = (mul_tensor_123 + -7.672876869259043e-09)
        mul_tensor_125 = (mul_tensor_122 * 2.0)
        add_tensor_80 = (mul_tensor_125 + -4.634182341116144e-11)
        mul_tensor_126 = (arg0_1 * add_tensor_80)
        add_tensor_81 = (add_tensor_79 + mul_tensor_126)
        mul_tensor_127 = (arg0_1 * add_tensor_81)
        add_tensor_82 = (add_tensor_78 + mul_tensor_127)
        mul_tensor_128 = (sqrt_default * add_tensor_82)
        add_tensor_83 = (add_tensor_75 + mul_tensor_128)
        mul_tensor_129 = (arg1_1 * add_tensor_83)
        add_tensor_84 = (add_tensor_68 + mul_tensor_129)
        mul_tensor_130 = (mul_tensor_63 * 2.0)
        mul_tensor_131 = (add_tensor_5 * mul_tensor_14)
        mul_tensor_132 = (mul_tensor_130 * mul_tensor_131)
        mul_tensor_133 = (mul_tensor_71 * 2.0)
        mul_tensor_134 = (mul_tensor_133 * mul_tensor_131)
        add_tensor_85 = (mul_tensor_132 + mul_tensor_134)
        mul_tensor_135 = (mul_tensor_79 * 2.0)
        mul_tensor_136 = (mul_tensor_135 * mul_tensor_48)
        add_tensor_86 = (add_tensor_85 + mul_tensor_136)
        mul_tensor_137 = (mul_tensor_79 * 2.0)
        mul_tensor_138 = (mul_tensor_137 * add_tensor_5)
        mul_tensor_139 = (mul_tensor_138 * mul_tensor_25)
        mul_tensor_140 = (mul_tensor_139 * add_tensor_12)
        sub_tensor_14 = (add_tensor_86 - mul_tensor_140)
        mul_tensor_141 = (add_tensor_45 * add_tensor_34)
        sub_tensor_15 = (sub_tensor_14 - mul_tensor_141)
        mul_tensor_142 = (add_tensor_46 * add_tensor_43)
        sub_tensor_16 = (sub_tensor_15 - mul_tensor_142)
        mul_tensor_143 = (sub_tensor_16 * mul_tensor_14)
        add_tensor_87 = (add_tensor_84 + mul_tensor_143)
        mul_tensor_144 = (mul_tensor_79 * 2.0)
        mul_tensor_145 = (mul_tensor_144 * mul_tensor_131)
        mul_tensor_146 = (add_tensor_46 * add_tensor_34)
        sub_tensor_17 = (mul_tensor_145 - mul_tensor_146)
        mul_tensor_147 = (sub_tensor_17 * mul_tensor_25)
        mul_tensor_148 = (mul_tensor_147 * add_tensor_12)
        sub_tensor_18 = (add_tensor_87 - mul_tensor_148)
        mul_tensor_149 = (sub_tensor_18 * add_tensor_8)
        add_tensor_88 = (mul_tensor_112 + 0.002775927747785646)
        mul_tensor_150 = (arg0_1 * add_tensor_88)
        add_tensor_89 = (mul_tensor_150 + 1.0)
        add_tensor_90 = (mul_tensor_118 + -2.764306979894411e-05)
        mul_tensor_151 = (arg0_1 * add_tensor_90)
        add_tensor_91 = (mul_tensor_151 + -0.007521448093615448)
        mul_tensor_152 = (arg1_1 * 5.41932655114874e-06)
        add_tensor_92 = (add_tensor_91 + mul_tensor_152)
        add_tensor_93 = (mul_tensor_124 + 3.801564588876298e-07)
        mul_tensor_153 = (arg0_1 * add_tensor_93)
        add_tensor_94 = (mul_tensor_153 + -3.303308871386421e-05)
        mul_tensor_154 = (sqrt_default * add_tensor_94)
        add_tensor_95 = (add_tensor_92 + mul_tensor_154)
        mul_tensor_155 = (arg1_1 * add_tensor_95)
        add_tensor_96 = (add_tensor_89 + mul_tensor_155)
        mul_tensor_156 = (sub_tensor_17 * mul_tensor_14)
        add_tensor_97 = (add_tensor_96 + mul_tensor_156)
        mul_tensor_157 = (add_tensor_97 * add_tensor_12)
        add_tensor_98 = (mul_tensor_149 + mul_tensor_157)
        mul_tensor_158 = (sub_tensor_11 * add_tensor_5)
        sub_tensor_19 = (add_tensor_98 - mul_tensor_158)
        mul_tensor_159 = (add_tensor_57 * add_tensor_11)
        sub_tensor_20 = (sub_tensor_19 - mul_tensor_159)
        mul_tensor_160 = (sub_tensor_20 * 5000.0)
        mul_tensor_161 = (mul_tensor_160 * mul_tensor_14)
        mul_tensor_162 = (add_tensor_34 * add_tensor_8)
        sub_tensor_21 = (pow_tensor_scalar_1 - mul_tensor_162)
        sqrt_default_1 = tl.sqrt(sub_tensor_21)
        reciprocal_default_3 = (1.0 / sqrt_default_1)
        mul_tensor_163 = (reciprocal_default_3 * 1.0)
        mul_tensor_164 = (mul_tensor_88 * 2.0)
        add_tensor_99 = (mul_tensor_6 + -0.011166348136762635)
        add_tensor_100 = (add_tensor_99 + mul_tensor_9)
        sub_tensor_22 = (add_tensor_100 - sqrt_default_1)
        reciprocal_default_4 = (1.0 / sub_tensor_22)
        mul_tensor_165 = (reciprocal_default_4 * 1.0)
        mul_tensor_166 = (sqrt_default_1 * mul_tensor_165)
        add_tensor_101 = (mul_tensor_6 + -0.011166348136762635)
        add_tensor_102 = (add_tensor_101 + mul_tensor_9)
        add_tensor_103 = (add_tensor_102 + sqrt_default_1)
        add_tensor_104 = (add_tensor_103 + mul_tensor_88)
        reciprocal_default_5 = (1.0 / add_tensor_104)
        mul_tensor_167 = (reciprocal_default_5 * 1.0)
        mul_tensor_168 = (mul_tensor_166 * mul_tensor_167)
        mul_tensor_169 = (mul_tensor_164 * mul_tensor_168)
        add_tensor_105 = (mul_tensor_169 + 1.0)
        log_default_1 = tl.log(add_tensor_105)
        mul_tensor_170 = (mul_tensor_163 * log_default_1)
        mul_tensor_171 = (mul_tensor_161 * mul_tensor_170)
        add_tensor_106 = (add_tensor_61 + mul_tensor_171)
        mul_tensor_172 = (add_tensor_97 * add_tensor_8)
        mul_tensor_173 = (add_tensor_57 * add_tensor_5)
        sub_tensor_23 = (mul_tensor_172 - mul_tensor_173)
        mul_tensor_174 = (sub_tensor_23 * 5000.0)
        mul_tensor_175 = (mul_tensor_174 * mul_tensor_25)
        mul_tensor_176 = (mul_tensor_175 * mul_tensor_170)
        mul_tensor_177 = (mul_tensor_176 * add_tensor_12)
        sub_tensor_24 = (add_tensor_106 - mul_tensor_177)
        mul_tensor_178 = (sub_tensor_23 * mul_tensor_14)
        mul_tensor_179 = (mul_tensor_178 * 2500.0)
        div_tensor_2 = (mul_tensor_179 / sqrt_default_1)
        div_tensor_3 = (div_tensor_2 / sub_tensor_21)
        mul_tensor_180 = (div_tensor_3 * log_default_1)
        mul_tensor_181 = (add_tensor_5 * 2.0)
        mul_tensor_182 = (mul_tensor_181 * add_tensor_11)
        mul_tensor_183 = (add_tensor_43 * add_tensor_8)
        sub_tensor_25 = (mul_tensor_182 - mul_tensor_183)
        mul_tensor_184 = (add_tensor_34 * add_tensor_12)
        sub_tensor_26 = (sub_tensor_25 - mul_tensor_184)
        mul_tensor_185 = (mul_tensor_180 * sub_tensor_26)
        sub_tensor_27 = (sub_tensor_24 - mul_tensor_185)
        mul_tensor_186 = (mul_tensor_178 * 5000.0)
        mul_tensor_187 = (mul_tensor_186 * mul_tensor_163)
        mul_tensor_188 = (mul_tensor_105 * 2.0)
        mul_tensor_189 = (mul_tensor_188 * mul_tensor_168)
        mul_tensor_190 = (mul_tensor_88 * mul_tensor_163)
        mul_tensor_191 = (mul_tensor_190 * mul_tensor_165)
        mul_tensor_192 = (mul_tensor_191 * mul_tensor_167)
        mul_tensor_193 = (mul_tensor_192 * sub_tensor_26)
        add_tensor_107 = (mul_tensor_189 + mul_tensor_193)
        mul_tensor_194 = (mul_tensor_88 * sqrt_default_1)
        mul_tensor_195 = (mul_tensor_194 * 2.0)
        pow_tensor_scalar_3 = (sub_tensor_22 * sub_tensor_22)
        div_tensor_4 = (mul_tensor_195 / pow_tensor_scalar_3)
        mul_tensor_196 = (div_tensor_4 * mul_tensor_167)
        add_tensor_108 = (mul_tensor_18 + -0.000171804503992594)
        add_tensor_109 = (add_tensor_108 + mul_tensor_20)
        mul_tensor_197 = (mul_tensor_163 * sub_tensor_26)
        div_tensor_5 = (mul_tensor_197 / 2.0)
        sub_tensor_28 = (add_tensor_109 - div_tensor_5)
        mul_tensor_198 = (mul_tensor_196 * sub_tensor_28)
        sub_tensor_29 = (add_tensor_107 - mul_tensor_198)
        mul_tensor_199 = (mul_tensor_194 * 2.0)
        mul_tensor_200 = (mul_tensor_199 * mul_tensor_165)
        pow_tensor_scalar_4 = (add_tensor_104 * add_tensor_104)
        div_tensor_6 = (mul_tensor_200 / pow_tensor_scalar_4)
        add_tensor_110 = (mul_tensor_18 + -0.000171804503992594)
        add_tensor_111 = (add_tensor_110 + mul_tensor_20)
        add_tensor_112 = (add_tensor_111 + div_tensor_5)
        add_tensor_113 = (add_tensor_112 + mul_tensor_105)
        mul_tensor_201 = (div_tensor_6 * add_tensor_113)
        sub_tensor_30 = (sub_tensor_29 - mul_tensor_201)
        mul_tensor_202 = (mul_tensor_187 * sub_tensor_30)
        div_tensor_7 = (mul_tensor_202 / add_tensor_105)
        add_tensor_114 = (sub_tensor_27 + div_tensor_7)

        tl.store(out_ptr + offsets, add_tensor_114, mask=mask)


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    _require_triton()
    if len(inputs) != 3:
        raise ValueError(f"expected 3 Repro inputs, got {len(inputs)}")

    arg2_1, arg0_1, arg1_1 = inputs
    expected = {
        "arg2_1": (arg2_1, ARG2_SHAPE, torch.float64),
        "arg0_1": (arg0_1, OUT_SHAPE, torch.float64),
        "arg1_1": (arg1_1, OUT_SHAPE, torch.float64),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    """Run the full Repro.forward computation for the canonical shape."""
    inputs = tuple(inputs)
    _validate_inputs(inputs)
    arg2_1, arg0_1, arg1_1 = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.float64,
    )
    _equation_pointwise_kernel[(triton.cdiv(N_ELEMS, BLOCK_N),)](
        arg2_1,
        arg0_1,
        arg1_1,
        out,
        N=N_ELEMS,
        BLOCK=BLOCK_N,
        num_warps=1,
    )
    return out


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_nan_aware(
    instance: torch.nn.Module,
    inputs: tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Correctness check for this repro's expected NaN-producing domains."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(eager_list) != len(oracle_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    ok_all = True
    for idx, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = actual.shape == expected.shape
        dtype_ok = actual.dtype == expected.dtype
        stride_ok = actual.stride() == expected.stride()
        if not (shape_ok and dtype_ok and stride_ok):
            print(
                f"  output {idx}: SCOPE_MISMATCH "
                f"shape={shape_ok} dtype={dtype_ok} stride={stride_ok}"
            )
            ok_all = False
            continue

        if not expected.is_floating_point():
            ok = torch.equal(actual, expected)
            print(f"  output {idx}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
            ok_all = ok_all and ok
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_match = torch.equal(expected_nan, actual_nan)
        finite = ~(expected_nan | actual_nan)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        else:
            max_diff = 0.0
        values_ok = torch.allclose(
            expected_f32,
            actual_f32,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        ok = nan_match and values_ok
        print(
            f"  output {idx}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={tuple(expected.stride())} max_finite_diff={max_diff:.2e} "
            f"nan_count={expected_nan.sum().item()})"
        )
        ok_all = ok_all and ok
    return ok_all


def _compile_with_config(
    instance: torch.nn.Module,
    inputs: tuple[Any, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(instance)
        with torch.no_grad():
            for _ in range(5):
                compiled(*inputs)
            torch.cuda.synchronize()
    return compiled


def _do_cuda_bench(fn, warmup: int, rep: int) -> float:
    _require_triton()
    with torch.no_grad():
        return triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0


def bench_required_configs(inputs: tuple[Any, ...], warmup: int, rep: int) -> dict[str, object]:
    """Benchmark the full-scope oracle and the requested torch.compile configs."""
    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _do_cuda_bench(lambda: oracle_forward(inputs), warmup, rep)
    print(f"oracle_triton full-scope f64 pointwise equation: {oracle_us:.3f} us")

    compile_timings: dict[str, float] = {}
    for label, config in COMPILE_CONFIGS:
        instance = get_repro_instance()
        compiled = _compile_with_config(instance, inputs, config)
        with torch.no_grad():
            compile_us = _do_cuda_bench(lambda: compiled(*inputs), warmup, rep)
        compile_timings[label] = compile_us
        print(f"torch.compile {label}: {compile_us:.3f} us")

    best_compile_us = min(compile_timings.values())
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_timings["coordinate_descent_tuning=True"], 3),
        "combo_compile_us": round(
            compile_timings[
                "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
                "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
                "triton.multi_kernel=3"
            ],
            3,
        ),
        "best_required_compile_us": round(best_compile_us, 3),
        "ratio": round(best_compile_us / oracle_us, 3) if oracle_us > 0.0 else 0.0,
        "status": "GOOD" if oracle_us < best_compile_us else "BAD_ORACLE",
        "true_floor": oracle_us < best_compile_us,
        "classification": "BANDWIDTH_BOUND",
    }
    print(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Accepted for template compatibility; this repro has no stochastic outputs.",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_oracle_nan_aware(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        "WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            bench_required_configs(tuple(inputs), warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()

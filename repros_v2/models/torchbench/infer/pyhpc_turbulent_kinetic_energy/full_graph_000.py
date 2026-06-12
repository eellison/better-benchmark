class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0", arg1_1: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0", arg2_1: "bf16[204, 204, 26][5304, 26, 1]cuda:0", arg3_1: "i64[204, 204][204, 1]cuda:0", arg4_1: "bf16[26][1]cuda:0", arg5_1: "bf16[204, 204, 26][5304, 26, 1]cuda:0", arg6_1: "bf16[26][1]cuda:0", arg7_1: "bf16[204, 204, 26][5304, 26, 1]cuda:0", arg8_1: "bf16[204, 204, 26][5304, 26, 1]cuda:0", arg9_1: "bf16[204, 204][204, 1]cuda:0", arg10_1: "bf16[204][1]cuda:0", arg11_1: "bf16[204][1]cuda:0", arg12_1: "bf16[204][1]cuda:0", arg13_1: "bf16[204, 204, 26][5304, 26, 1]cuda:0", arg14_1: "bf16[204][1]cuda:0", arg15_1: "bf16[204, 204, 26][5304, 26, 1]cuda:0", arg16_1: "bf16[204][1]cuda:0", arg17_1: "bf16[204][1]cuda:0", arg18_1: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0", arg19_1: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0", arg20_1: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:198 in integrate_tke, code: torch.maximum(torch.tensor([0.0], device=tke.device), tke[:, :, :, tau])
        _tensor_constant0: "f32[1][1]cuda:0" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:214 in integrate_tke, code: d_tri = torch.zeros_like(maskU[2:-2, 2:-2])
        full_default_2: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:246 in integrate_tke, code: d_tri[...] = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :]
        slice_53: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_54: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_53, 1, 2, -2);  slice_53 = None
        select_15: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_54, 3, 0);  slice_54 = None
        slice_55: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2);  arg8_1 = None
        slice_56: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_55, 1, 2, -2);  slice_55 = None
        mul_11: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_56, 1);  slice_56 = None
        add_8: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(select_15, mul_11);  select_15 = mul_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:247 in integrate_tke, code: d_tri[:, :, -1] += dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])
        select_18: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(add_8, 2, -1)
        slice_57: "bf16[200, 204][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2);  arg9_1 = None
        slice_58: "bf16[200, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_57, 1, 2, -2);  slice_57 = None
        mul_12: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_58, 1);  slice_58 = None
        select_17: "bf16[][]cuda:0" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_13: "bf16[][]cuda:0" = torch.ops.aten.mul.Tensor(select_17, 0.5);  select_17 = None
        div_7: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        add_9: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_18, div_7);  select_18 = div_7 = None
        select_scatter_2: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_8, add_9, 2, -1);  add_8 = add_9 = None
        select_21: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_2, 2, -1);  select_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:209 in integrate_tke, code: ks = kbot[2:-2, 2:-2] - 1
        slice_1: "i64[200, 204][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg3_1, 0, 2, -2);  arg3_1 = None
        slice_2: "i64[200, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1, 1, 2, -2);  slice_1 = None
        sub: "i64[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_2, 1);  slice_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:27 in solve_implicit, code: land_mask = (ks >= 0)[:, :, None]
        ge: "b8[200, 200][200, 1]cuda:0" = torch.ops.aten.ge.Scalar(sub, 0)
        unsqueeze_10: "b8[200, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ge, 2);  ge = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:29 in solve_implicit, code: torch.arange(a.shape[2], device=ks.device)[None, None, :] == ks[:, :, None]
        iota: "i64[26][1]cuda:0" = torch.ops.prims.iota.default(26, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_11: "i64[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        unsqueeze_12: "i64[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 1);  unsqueeze_11 = None
        unsqueeze_13: "i64[200, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(sub, 2)
        eq: "b8[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.eq.Tensor(unsqueeze_12, unsqueeze_13);  unsqueeze_12 = unsqueeze_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:28 in solve_implicit, code: edge_mask = land_mask & (
        bitwise_and: "b8[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(unsqueeze_10, eq);  eq = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:215 in integrate_tke, code: delta = torch.zeros_like(maskU[2:-2, 2:-2])
        full_7: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:217 in integrate_tke, code: delta[:, :, :-1] = (
        slice_20: "bf16[200, 200, 25][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_7, 2, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:219 in integrate_tke, code: / dzt[None, None, 1:]
        unsqueeze: "bf16[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        unsqueeze_1: "bf16[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        slice_13: "bf16[1, 1, 25][26, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_1, 2, 1, 9223372036854775807);  unsqueeze_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:218 in integrate_tke, code: dt_tke
        reciprocal: "bf16[1, 1, 25][25, 25, 1]cuda:0" = torch.ops.aten.reciprocal.default(slice_13);  slice_13 = None
        mul: "bf16[1, 1, 25][25, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_1: "bf16[1, 1, 25][25, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.0);  mul = None
        mul_2: "bf16[1, 1, 25][25, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, 0.5);  mul_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:222 in integrate_tke, code: * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:])
        slice_14: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_15: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_14, 1, 2, -2);  slice_14 = None
        slice_16: "bf16[200, 200, 25][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_15, 2, 0, -1);  slice_15 = None
        slice_17: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2);  arg5_1 = None
        slice_18: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_17, 1, 2, -2);  slice_17 = None
        slice_19: "bf16[200, 200, 25][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_18, 2, 1, 9223372036854775807);  slice_18 = None
        add: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_16, slice_19);  slice_16 = slice_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:218 in integrate_tke, code: dt_tke
        mul_3: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, add);  mul_2 = add = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:217 in integrate_tke, code: delta[:, :, :-1] = (
        copy: "bf16[200, 200, 25][5200, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_20, mul_3);  slice_20 = mul_3 = None
        slice_scatter: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_7, copy, 2, 0, -1);  full_7 = copy = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:240 in integrate_tke, code: + delta / dzw[None, None, :]
        unsqueeze_6: "bf16[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_7: "bf16[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        div_5: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(slice_scatter, unsqueeze_7);  unsqueeze_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:239 in integrate_tke, code: 1
        add_6: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(div_5, 1);  div_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:241 in integrate_tke, code: + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]
        slice_44: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2)
        slice_45: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_44, 1, 2, -2);  slice_44 = None
        reciprocal_2: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.reciprocal.default(slice_45);  slice_45 = None
        mul_9: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 0.7);  reciprocal_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:198 in integrate_tke, code: torch.maximum(torch.tensor([0.0], device=tke.device), tke[:, :, :, tau])
        full_default_1: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(arg0_1, 3, 0)
        maximum: "f32[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.maximum.default(full_default_1, select);  full_default_1 = select = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:197 in integrate_tke, code: sqrttke = torch.sqrt(
        sqrt: "f32[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.sqrt.default(maximum);  maximum = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:241 in integrate_tke, code: + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]
        slice_46: "f32[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(sqrt, 0, 2, -2)
        slice_47: "f32[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_46, 1, 2, -2);  slice_46 = None
        mul_10: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, slice_47);  mul_9 = slice_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:239 in integrate_tke, code: 1
        add_7: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, mul_10);  add_6 = mul_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:32 in solve_implicit, code: torch.arange(a.shape[2], device=ks.device)[None, None, :] >= ks[:, :, None]
        iota_1: "i64[26][1]cuda:0" = torch.ops.prims.iota.default(26, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_14: "i64[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, 0);  iota_1 = None
        unsqueeze_15: "i64[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
        unsqueeze_16: "i64[200, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(sub, 2);  sub = None
        ge_1: "b8[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.ge.Tensor(unsqueeze_15, unsqueeze_16);  unsqueeze_15 = unsqueeze_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:31 in solve_implicit, code: water_mask = land_mask & (
        bitwise_and_1: "b8[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(unsqueeze_10, ge_1);  unsqueeze_10 = ge_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:212 in integrate_tke, code: b_tri = torch.zeros_like(maskU[2:-2, 2:-2])
        full_4: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:228 in integrate_tke, code: b_tri[:, :, 1:-1] = (
        slice_38: "bf16[200, 200, 24][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_4, 2, 1, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:230 in integrate_tke, code: + (delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[None, None, 1:-1]
        slice_29: "bf16[200, 200, 24][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 1, -1)
        slice_30: "bf16[200, 200, 24][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 0, -2)
        add_1: "bf16[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_29, slice_30);  slice_29 = slice_30 = None
        unsqueeze_4: "bf16[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_5: "bf16[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        slice_31: "bf16[1, 1, 24][26, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_5, 2, 1, -1);  unsqueeze_5 = None
        div_2: "bf16[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.div.Tensor(add_1, slice_31);  add_1 = slice_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:229 in integrate_tke, code: 1
        add_2: "bf16[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.add.Tensor(div_2, 1);  div_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:231 in integrate_tke, code: + dt_tke * c_eps * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1]
        slice_32: "f32[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(sqrt, 0, 2, -2)
        slice_33: "f32[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_32, 1, 2, -2);  slice_32 = None
        slice_34: "f32[200, 200, 24][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_33, 2, 1, -1);  slice_33 = None
        mul_5: "f32[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_34, 0.7);  slice_34 = None
        slice_35: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2)
        slice_36: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_35, 1, 2, -2);  slice_35 = None
        slice_37: "bf16[200, 200, 24][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_36, 2, 1, -1);  slice_36 = None
        div_3: "f32[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_5, slice_37);  mul_5 = slice_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:229 in integrate_tke, code: 1
        add_3: "f32[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, div_3);  add_2 = div_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:228 in integrate_tke, code: b_tri[:, :, 1:-1] = (
        copy_3: "bf16[200, 200, 24][5200, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_38, add_3);  slice_38 = add_3 = None
        slice_scatter_2: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_4, copy_3, 2, 1, -1);  full_4 = copy_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:233 in integrate_tke, code: b_tri[:, :, -1] = (
        select_13: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(slice_scatter_2, 2, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:235 in integrate_tke, code: + delta[:, :, -2] / (0.5 * dzw[-1])
        select_9: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(slice_scatter, 2, -2)
        select_8: "bf16[][]cuda:0" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_6: "bf16[][]cuda:0" = torch.ops.aten.mul.Tensor(select_8, 0.5);  select_8 = None
        div_4: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_9, mul_6);  select_9 = mul_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:234 in integrate_tke, code: 1
        add_4: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(div_4, 1);  div_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:236 in integrate_tke, code: + dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1]
        slice_40: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2);  arg7_1 = None
        slice_41: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_40, 1, 2, -2);  slice_40 = None
        select_10: "bf16[200, 200][5304, 26]cuda:0" = torch.ops.aten.select.int(slice_41, 2, -1);  slice_41 = None
        reciprocal_1: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.reciprocal.default(select_10);  select_10 = None
        mul_7: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 0.7);  reciprocal_1 = None
        slice_42: "f32[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(sqrt, 0, 2, -2);  sqrt = None
        slice_43: "f32[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_42, 1, 2, -2);  slice_42 = None
        select_11: "f32[200, 200][5304, 26]cuda:0" = torch.ops.aten.select.int(slice_43, 2, -1);  slice_43 = None
        mul_8: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, select_11);  mul_7 = select_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:234 in integrate_tke, code: 1
        add_5: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4, mul_8);  add_4 = mul_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:233 in integrate_tke, code: b_tri[:, :, -1] = (
        copy_4: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_13, add_5);  select_13 = add_5 = None
        select_scatter_1: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_2, copy_4, 2, -1);  slice_scatter_2 = copy_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:36 in solve_implicit, code: b_tri = torch.where(water_mask, b, 1.0)
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 1.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.where.self(bitwise_and_1, select_scatter_1, full_default_3);  select_scatter_1 = full_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:37 in solve_implicit, code: b_tri = torch.where(edge_mask, b_edge, b_tri)
        where_1: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.where.self(bitwise_and, add_7, where);  add_7 = where = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_25: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(where_1, 2, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:211 in integrate_tke, code: a_tri = torch.zeros_like(maskU[2:-2, 2:-2])
        full_3: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:225 in integrate_tke, code: a_tri[:, :, 1:-1] = -delta[:, :, :-2] / dzw[None, None, 1:-1]
        slice_25: "bf16[200, 200, 24][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_3, 2, 1, -1)
        slice_23: "bf16[200, 200, 24][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 0, -2)
        neg: "bf16[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.neg.default(slice_23);  slice_23 = None
        unsqueeze_2: "bf16[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_3: "bf16[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        slice_24: "bf16[1, 1, 24][26, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 1, -1);  unsqueeze_3 = None
        div: "bf16[200, 200, 24][4800, 24, 1]cuda:0" = torch.ops.aten.div.Tensor(neg, slice_24);  neg = slice_24 = None
        copy_1: "bf16[200, 200, 24][5200, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_25, div);  slice_25 = div = None
        slice_scatter_1: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_3, copy_1, 2, 1, -1);  full_3 = copy_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:226 in integrate_tke, code: a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * dzw[-1])
        select_5: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(slice_scatter_1, 2, -1)
        select_2: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(slice_scatter, 2, -2)
        neg_1: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(select_2);  select_2 = None
        select_3: "bf16[][]cuda:0" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_4: "bf16[][]cuda:0" = torch.ops.aten.mul.Tensor(select_3, 0.5);  select_3 = None
        div_1: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(neg_1, mul_4);  neg_1 = mul_4 = None
        copy_2: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_5, div_1);  select_5 = div_1 = None
        select_scatter: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_1, copy_2, 2, -1);  slice_scatter_1 = copy_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:35 in solve_implicit, code: a_tri = water_mask * a * torch.logical_not(edge_mask)
        mul_14: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(bitwise_and_1, select_scatter);  select_scatter = None
        logical_not: "b8[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.logical_not.default(bitwise_and);  bitwise_and = None
        mul_15: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, logical_not);  mul_14 = logical_not = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_23: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 1)
        select_24: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(where_1, 2, 0)
        div_8: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_23, select_24);  select_23 = select_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_3: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_8)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:213 in integrate_tke, code: c_tri = torch.zeros_like(maskU[2:-2, 2:-2])
        full_5: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:244 in integrate_tke, code: c_tri[:, :, :-1] = -delta[:, :, :-1] / dzw[None, None, :-1]
        slice_51: "bf16[200, 200, 25][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_5, 2, 0, -1)
        slice_49: "bf16[200, 200, 25][5200, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 0, -1);  slice_scatter = None
        neg_2: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.neg.default(slice_49);  slice_49 = None
        unsqueeze_8: "bf16[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_9: "bf16[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
        slice_50: "bf16[1, 1, 25][26, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, -1);  unsqueeze_9 = None
        div_6: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.div.Tensor(neg_2, slice_50);  neg_2 = slice_50 = None
        copy_5: "bf16[200, 200, 25][5200, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_51, div_6);  slice_51 = div_6 = None
        slice_scatter_3: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_5, copy_5, 2, 0, -1);  full_5 = copy_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:38 in solve_implicit, code: c_tri = water_mask * c
        mul_16: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(bitwise_and_1, slice_scatter_3);  slice_scatter_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_26: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 0)
        mul_18: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_3, select_26);  neg_3 = select_26 = None
        add_10: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_25, mul_18);  select_25 = mul_18 = None
        select_scatter_4: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(where_1, add_10, 2, 1);  where_1 = add_10 = None
        select_29: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_4, 2, 1);  select_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:247 in integrate_tke, code: d_tri[:, :, -1] += dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])
        select_19: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_2, 2, -1)
        select_scatter_3: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_2, select_19, 2, -1);  select_scatter_2 = select_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:39 in solve_implicit, code: d_tri = water_mask * d
        mul_17: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(bitwise_and_1, select_scatter_3);  select_scatter_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_31: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_17, 2, 1)
        neg_4: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_8);  div_8 = None
        select_32: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_17, 2, 0)
        mul_19: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_4, select_32);  neg_4 = select_32 = None
        add_11: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_31, mul_19);  select_31 = mul_19 = None
        convert_element_type: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        select_scatter_6: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(mul_17, convert_element_type, 2, 1);  mul_17 = convert_element_type = None
        select_35: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_6, 2, 1);  select_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_27: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_4, 2, 1)
        select_scatter_5: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_4, select_27, 2, 1);  select_scatter_4 = select_27 = None
        select_42: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_5, 2, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_37: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 2)
        select_39: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_5, 2, 1)
        div_9: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_37, select_39);  select_37 = select_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_5: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_9)
        select_41: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 1)
        mul_20: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_5, select_41);  neg_5 = select_41 = None
        add_12: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_42, mul_20);  select_42 = mul_20 = None
        select_scatter_8: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_5, add_12, 2, 2);  select_scatter_5 = add_12 = None
        select_45: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_8, 2, 2);  select_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_33: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_6, 2, 1)
        select_scatter_7: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_6, select_33, 2, 1);  select_scatter_6 = select_33 = None
        select_50: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_7, 2, 2)
        neg_6: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        select_49: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_7, 2, 1)
        mul_21: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_6, select_49);  neg_6 = select_49 = None
        add_13: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_50, mul_21);  select_50 = mul_21 = None
        convert_element_type_1: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        select_scatter_10: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_7, convert_element_type_1, 2, 2);  select_scatter_7 = convert_element_type_1 = None
        select_53: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_10, 2, 2);  select_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_43: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_8, 2, 2)
        select_scatter_9: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_8, select_43, 2, 2);  select_scatter_8 = select_43 = None
        select_60: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_9, 2, 3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_55: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 3)
        select_57: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_9, 2, 2)
        div_10: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_55, select_57);  select_55 = select_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_7: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_10)
        select_59: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 2)
        mul_22: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_7, select_59);  neg_7 = select_59 = None
        add_14: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_60, mul_22);  select_60 = mul_22 = None
        select_scatter_12: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_9, add_14, 2, 3);  select_scatter_9 = add_14 = None
        select_63: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_12, 2, 3);  select_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_51: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_10, 2, 2)
        select_scatter_11: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_10, select_51, 2, 2);  select_scatter_10 = select_51 = None
        select_68: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_11, 2, 3)
        neg_8: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        select_67: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_11, 2, 2)
        mul_23: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_8, select_67);  neg_8 = select_67 = None
        add_15: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_68, mul_23);  select_68 = mul_23 = None
        convert_element_type_2: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        select_scatter_14: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_11, convert_element_type_2, 2, 3);  select_scatter_11 = convert_element_type_2 = None
        select_71: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_14, 2, 3);  select_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_61: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_12, 2, 3)
        select_scatter_13: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_12, select_61, 2, 3);  select_scatter_12 = select_61 = None
        select_78: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_13, 2, 4)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_73: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 4)
        select_75: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_13, 2, 3)
        div_11: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_73, select_75);  select_73 = select_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_9: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_11)
        select_77: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 3)
        mul_24: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_9, select_77);  neg_9 = select_77 = None
        add_16: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_78, mul_24);  select_78 = mul_24 = None
        select_scatter_16: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_13, add_16, 2, 4);  select_scatter_13 = add_16 = None
        select_81: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_16, 2, 4);  select_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_69: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_14, 2, 3)
        select_scatter_15: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_14, select_69, 2, 3);  select_scatter_14 = select_69 = None
        select_86: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_15, 2, 4)
        neg_10: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        select_85: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_15, 2, 3)
        mul_25: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_10, select_85);  neg_10 = select_85 = None
        add_17: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_86, mul_25);  select_86 = mul_25 = None
        convert_element_type_3: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        select_scatter_18: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_15, convert_element_type_3, 2, 4);  select_scatter_15 = convert_element_type_3 = None
        select_89: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_18, 2, 4);  select_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_79: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_16, 2, 4)
        select_scatter_17: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_16, select_79, 2, 4);  select_scatter_16 = select_79 = None
        select_96: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_17, 2, 5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_91: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 5)
        select_93: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_17, 2, 4)
        div_12: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_91, select_93);  select_91 = select_93 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_11: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_12)
        select_95: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 4)
        mul_26: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_11, select_95);  neg_11 = select_95 = None
        add_18: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_96, mul_26);  select_96 = mul_26 = None
        select_scatter_20: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_17, add_18, 2, 5);  select_scatter_17 = add_18 = None
        select_99: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_20, 2, 5);  select_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_87: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_18, 2, 4)
        select_scatter_19: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_18, select_87, 2, 4);  select_scatter_18 = select_87 = None
        select_104: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_19, 2, 5)
        neg_12: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_12);  div_12 = None
        select_103: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_19, 2, 4)
        mul_27: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_12, select_103);  neg_12 = select_103 = None
        add_19: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_104, mul_27);  select_104 = mul_27 = None
        convert_element_type_4: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        select_scatter_22: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_19, convert_element_type_4, 2, 5);  select_scatter_19 = convert_element_type_4 = None
        select_107: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_22, 2, 5);  select_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_97: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_20, 2, 5)
        select_scatter_21: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_20, select_97, 2, 5);  select_scatter_20 = select_97 = None
        select_114: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_21, 2, 6)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_109: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 6)
        select_111: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_21, 2, 5)
        div_13: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_109, select_111);  select_109 = select_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_13: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_13)
        select_113: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 5)
        mul_28: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_13, select_113);  neg_13 = select_113 = None
        add_20: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_114, mul_28);  select_114 = mul_28 = None
        select_scatter_24: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_21, add_20, 2, 6);  select_scatter_21 = add_20 = None
        select_117: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_24, 2, 6);  select_117 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_105: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_22, 2, 5)
        select_scatter_23: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_22, select_105, 2, 5);  select_scatter_22 = select_105 = None
        select_122: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_23, 2, 6)
        neg_14: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        select_121: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_23, 2, 5)
        mul_29: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_14, select_121);  neg_14 = select_121 = None
        add_21: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_122, mul_29);  select_122 = mul_29 = None
        convert_element_type_5: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        select_scatter_26: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_23, convert_element_type_5, 2, 6);  select_scatter_23 = convert_element_type_5 = None
        select_125: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_26, 2, 6);  select_125 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_115: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_24, 2, 6)
        select_scatter_25: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_24, select_115, 2, 6);  select_scatter_24 = select_115 = None
        select_132: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_25, 2, 7)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_127: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 7)
        select_129: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_25, 2, 6)
        div_14: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_127, select_129);  select_127 = select_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_15: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_14)
        select_131: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 6)
        mul_30: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_15, select_131);  neg_15 = select_131 = None
        add_22: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_132, mul_30);  select_132 = mul_30 = None
        select_scatter_28: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_25, add_22, 2, 7);  select_scatter_25 = add_22 = None
        select_135: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_28, 2, 7);  select_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_123: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_26, 2, 6)
        select_scatter_27: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_26, select_123, 2, 6);  select_scatter_26 = select_123 = None
        select_140: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_27, 2, 7)
        neg_16: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_14);  div_14 = None
        select_139: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_27, 2, 6)
        mul_31: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_16, select_139);  neg_16 = select_139 = None
        add_23: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_140, mul_31);  select_140 = mul_31 = None
        convert_element_type_6: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        select_scatter_30: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_27, convert_element_type_6, 2, 7);  select_scatter_27 = convert_element_type_6 = None
        select_143: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_30, 2, 7);  select_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_133: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_28, 2, 7)
        select_scatter_29: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_28, select_133, 2, 7);  select_scatter_28 = select_133 = None
        select_150: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_29, 2, 8)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_145: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 8)
        select_147: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_29, 2, 7)
        div_15: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_145, select_147);  select_145 = select_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_17: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_15)
        select_149: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 7)
        mul_32: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_17, select_149);  neg_17 = select_149 = None
        add_24: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_150, mul_32);  select_150 = mul_32 = None
        select_scatter_32: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_29, add_24, 2, 8);  select_scatter_29 = add_24 = None
        select_153: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_32, 2, 8);  select_153 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_141: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_30, 2, 7)
        select_scatter_31: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_30, select_141, 2, 7);  select_scatter_30 = select_141 = None
        select_158: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_31, 2, 8)
        neg_18: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        select_157: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_31, 2, 7)
        mul_33: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_18, select_157);  neg_18 = select_157 = None
        add_25: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_158, mul_33);  select_158 = mul_33 = None
        convert_element_type_7: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None
        select_scatter_34: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_31, convert_element_type_7, 2, 8);  select_scatter_31 = convert_element_type_7 = None
        select_161: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_34, 2, 8);  select_161 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_151: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_32, 2, 8)
        select_scatter_33: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_32, select_151, 2, 8);  select_scatter_32 = select_151 = None
        select_168: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_33, 2, 9)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_163: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 9)
        select_165: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_33, 2, 8)
        div_16: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_163, select_165);  select_163 = select_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_19: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_16)
        select_167: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 8)
        mul_34: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_19, select_167);  neg_19 = select_167 = None
        add_26: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_168, mul_34);  select_168 = mul_34 = None
        select_scatter_36: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_33, add_26, 2, 9);  select_scatter_33 = add_26 = None
        select_171: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_36, 2, 9);  select_171 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_159: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_34, 2, 8)
        select_scatter_35: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_34, select_159, 2, 8);  select_scatter_34 = select_159 = None
        select_176: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_35, 2, 9)
        neg_20: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_16);  div_16 = None
        select_175: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_35, 2, 8)
        mul_35: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_20, select_175);  neg_20 = select_175 = None
        add_27: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_176, mul_35);  select_176 = mul_35 = None
        convert_element_type_8: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        select_scatter_38: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_35, convert_element_type_8, 2, 9);  select_scatter_35 = convert_element_type_8 = None
        select_179: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_38, 2, 9);  select_179 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_169: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_36, 2, 9)
        select_scatter_37: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_36, select_169, 2, 9);  select_scatter_36 = select_169 = None
        select_186: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_37, 2, 10)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_181: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 10)
        select_183: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_37, 2, 9)
        div_17: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_181, select_183);  select_181 = select_183 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_21: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_17)
        select_185: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 9)
        mul_36: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_21, select_185);  neg_21 = select_185 = None
        add_28: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_186, mul_36);  select_186 = mul_36 = None
        select_scatter_40: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_37, add_28, 2, 10);  select_scatter_37 = add_28 = None
        select_189: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_40, 2, 10);  select_189 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_177: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_38, 2, 9)
        select_scatter_39: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_38, select_177, 2, 9);  select_scatter_38 = select_177 = None
        select_194: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_39, 2, 10)
        neg_22: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        select_193: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_39, 2, 9)
        mul_37: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_22, select_193);  neg_22 = select_193 = None
        add_29: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_194, mul_37);  select_194 = mul_37 = None
        convert_element_type_9: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        select_scatter_42: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_39, convert_element_type_9, 2, 10);  select_scatter_39 = convert_element_type_9 = None
        select_197: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_42, 2, 10);  select_197 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_187: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_40, 2, 10)
        select_scatter_41: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_40, select_187, 2, 10);  select_scatter_40 = select_187 = None
        select_204: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_41, 2, 11)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_199: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 11)
        select_201: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_41, 2, 10)
        div_18: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_199, select_201);  select_199 = select_201 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_23: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_18)
        select_203: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 10)
        mul_38: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_23, select_203);  neg_23 = select_203 = None
        add_30: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_204, mul_38);  select_204 = mul_38 = None
        select_scatter_44: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_41, add_30, 2, 11);  select_scatter_41 = add_30 = None
        select_207: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_44, 2, 11);  select_207 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_195: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_42, 2, 10)
        select_scatter_43: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_42, select_195, 2, 10);  select_scatter_42 = select_195 = None
        select_212: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_43, 2, 11)
        neg_24: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_18);  div_18 = None
        select_211: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_43, 2, 10)
        mul_39: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_24, select_211);  neg_24 = select_211 = None
        add_31: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_212, mul_39);  select_212 = mul_39 = None
        convert_element_type_10: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        select_scatter_46: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_43, convert_element_type_10, 2, 11);  select_scatter_43 = convert_element_type_10 = None
        select_215: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_46, 2, 11);  select_215 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_205: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_44, 2, 11)
        select_scatter_45: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_44, select_205, 2, 11);  select_scatter_44 = select_205 = None
        select_222: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_45, 2, 12)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_217: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 12)
        select_219: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_45, 2, 11)
        div_19: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_217, select_219);  select_217 = select_219 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_25: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_19)
        select_221: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 11)
        mul_40: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_25, select_221);  neg_25 = select_221 = None
        add_32: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_222, mul_40);  select_222 = mul_40 = None
        select_scatter_48: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_45, add_32, 2, 12);  select_scatter_45 = add_32 = None
        select_225: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_48, 2, 12);  select_225 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_213: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_46, 2, 11)
        select_scatter_47: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_46, select_213, 2, 11);  select_scatter_46 = select_213 = None
        select_230: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_47, 2, 12)
        neg_26: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        select_229: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_47, 2, 11)
        mul_41: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_26, select_229);  neg_26 = select_229 = None
        add_33: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_230, mul_41);  select_230 = mul_41 = None
        convert_element_type_11: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        select_scatter_50: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_47, convert_element_type_11, 2, 12);  select_scatter_47 = convert_element_type_11 = None
        select_233: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_50, 2, 12);  select_233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_223: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_48, 2, 12)
        select_scatter_49: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_48, select_223, 2, 12);  select_scatter_48 = select_223 = None
        select_240: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_49, 2, 13)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_235: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 13)
        select_237: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_49, 2, 12)
        div_20: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_235, select_237);  select_235 = select_237 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_27: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_20)
        select_239: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 12)
        mul_42: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_27, select_239);  neg_27 = select_239 = None
        add_34: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_240, mul_42);  select_240 = mul_42 = None
        select_scatter_52: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_49, add_34, 2, 13);  select_scatter_49 = add_34 = None
        select_243: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_52, 2, 13);  select_243 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_231: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_50, 2, 12)
        select_scatter_51: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_50, select_231, 2, 12);  select_scatter_50 = select_231 = None
        select_248: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_51, 2, 13)
        neg_28: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_20);  div_20 = None
        select_247: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_51, 2, 12)
        mul_43: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_28, select_247);  neg_28 = select_247 = None
        add_35: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_248, mul_43);  select_248 = mul_43 = None
        convert_element_type_12: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        select_scatter_54: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_51, convert_element_type_12, 2, 13);  select_scatter_51 = convert_element_type_12 = None
        select_251: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_54, 2, 13);  select_251 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_241: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_52, 2, 13)
        select_scatter_53: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_52, select_241, 2, 13);  select_scatter_52 = select_241 = None
        select_258: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_53, 2, 14)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_253: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 14)
        select_255: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_53, 2, 13)
        div_21: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_253, select_255);  select_253 = select_255 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_29: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_21)
        select_257: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 13)
        mul_44: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_29, select_257);  neg_29 = select_257 = None
        add_36: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_258, mul_44);  select_258 = mul_44 = None
        select_scatter_56: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_53, add_36, 2, 14);  select_scatter_53 = add_36 = None
        select_261: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_56, 2, 14);  select_261 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_249: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_54, 2, 13)
        select_scatter_55: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_54, select_249, 2, 13);  select_scatter_54 = select_249 = None
        select_266: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_55, 2, 14)
        neg_30: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        select_265: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_55, 2, 13)
        mul_45: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_30, select_265);  neg_30 = select_265 = None
        add_37: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_266, mul_45);  select_266 = mul_45 = None
        convert_element_type_13: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None
        select_scatter_58: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_55, convert_element_type_13, 2, 14);  select_scatter_55 = convert_element_type_13 = None
        select_269: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_58, 2, 14);  select_269 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_259: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_56, 2, 14)
        select_scatter_57: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_56, select_259, 2, 14);  select_scatter_56 = select_259 = None
        select_276: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_57, 2, 15)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_271: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 15)
        select_273: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_57, 2, 14)
        div_22: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_271, select_273);  select_271 = select_273 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_31: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_22)
        select_275: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 14)
        mul_46: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_31, select_275);  neg_31 = select_275 = None
        add_38: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_276, mul_46);  select_276 = mul_46 = None
        select_scatter_60: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_57, add_38, 2, 15);  select_scatter_57 = add_38 = None
        select_279: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_60, 2, 15);  select_279 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_267: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_58, 2, 14)
        select_scatter_59: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_58, select_267, 2, 14);  select_scatter_58 = select_267 = None
        select_284: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_59, 2, 15)
        neg_32: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_22);  div_22 = None
        select_283: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_59, 2, 14)
        mul_47: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_32, select_283);  neg_32 = select_283 = None
        add_39: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_284, mul_47);  select_284 = mul_47 = None
        convert_element_type_14: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        select_scatter_62: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_59, convert_element_type_14, 2, 15);  select_scatter_59 = convert_element_type_14 = None
        select_287: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_62, 2, 15);  select_287 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_277: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_60, 2, 15)
        select_scatter_61: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_60, select_277, 2, 15);  select_scatter_60 = select_277 = None
        select_294: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_61, 2, 16)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_289: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 16)
        select_291: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_61, 2, 15)
        div_23: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_289, select_291);  select_289 = select_291 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_33: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_23)
        select_293: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 15)
        mul_48: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_33, select_293);  neg_33 = select_293 = None
        add_40: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_294, mul_48);  select_294 = mul_48 = None
        select_scatter_64: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_61, add_40, 2, 16);  select_scatter_61 = add_40 = None
        select_297: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_64, 2, 16);  select_297 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_285: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_62, 2, 15)
        select_scatter_63: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_62, select_285, 2, 15);  select_scatter_62 = select_285 = None
        select_302: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_63, 2, 16)
        neg_34: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_23);  div_23 = None
        select_301: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_63, 2, 15)
        mul_49: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_34, select_301);  neg_34 = select_301 = None
        add_41: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_302, mul_49);  select_302 = mul_49 = None
        convert_element_type_15: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        select_scatter_66: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_63, convert_element_type_15, 2, 16);  select_scatter_63 = convert_element_type_15 = None
        select_305: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_66, 2, 16);  select_305 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_295: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_64, 2, 16)
        select_scatter_65: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_64, select_295, 2, 16);  select_scatter_64 = select_295 = None
        select_312: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_65, 2, 17)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_307: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 17)
        select_309: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_65, 2, 16)
        div_24: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_307, select_309);  select_307 = select_309 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_35: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_24)
        select_311: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 16)
        mul_50: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_35, select_311);  neg_35 = select_311 = None
        add_42: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_312, mul_50);  select_312 = mul_50 = None
        select_scatter_68: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_65, add_42, 2, 17);  select_scatter_65 = add_42 = None
        select_315: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_68, 2, 17);  select_315 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_303: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_66, 2, 16)
        select_scatter_67: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_66, select_303, 2, 16);  select_scatter_66 = select_303 = None
        select_320: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_67, 2, 17)
        neg_36: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_24);  div_24 = None
        select_319: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_67, 2, 16)
        mul_51: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_36, select_319);  neg_36 = select_319 = None
        add_43: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_320, mul_51);  select_320 = mul_51 = None
        convert_element_type_16: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        select_scatter_70: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_67, convert_element_type_16, 2, 17);  select_scatter_67 = convert_element_type_16 = None
        select_323: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_70, 2, 17);  select_323 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_313: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_68, 2, 17)
        select_scatter_69: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_68, select_313, 2, 17);  select_scatter_68 = select_313 = None
        select_330: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_69, 2, 18)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_325: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 18)
        select_327: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_69, 2, 17)
        div_25: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_325, select_327);  select_325 = select_327 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_37: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_25)
        select_329: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 17)
        mul_52: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_37, select_329);  neg_37 = select_329 = None
        add_44: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_330, mul_52);  select_330 = mul_52 = None
        select_scatter_72: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_69, add_44, 2, 18);  select_scatter_69 = add_44 = None
        select_333: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_72, 2, 18);  select_333 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_321: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_70, 2, 17)
        select_scatter_71: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_70, select_321, 2, 17);  select_scatter_70 = select_321 = None
        select_338: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_71, 2, 18)
        neg_38: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_25);  div_25 = None
        select_337: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_71, 2, 17)
        mul_53: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_38, select_337);  neg_38 = select_337 = None
        add_45: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_338, mul_53);  select_338 = mul_53 = None
        convert_element_type_17: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        select_scatter_74: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_71, convert_element_type_17, 2, 18);  select_scatter_71 = convert_element_type_17 = None
        select_341: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_74, 2, 18);  select_341 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_331: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_72, 2, 18)
        select_scatter_73: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_72, select_331, 2, 18);  select_scatter_72 = select_331 = None
        select_348: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_73, 2, 19)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_343: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 19)
        select_345: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_73, 2, 18)
        div_26: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_343, select_345);  select_343 = select_345 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_39: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_26)
        select_347: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 18)
        mul_54: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_39, select_347);  neg_39 = select_347 = None
        add_46: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_348, mul_54);  select_348 = mul_54 = None
        select_scatter_76: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_73, add_46, 2, 19);  select_scatter_73 = add_46 = None
        select_351: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_76, 2, 19);  select_351 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_339: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_74, 2, 18)
        select_scatter_75: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_74, select_339, 2, 18);  select_scatter_74 = select_339 = None
        select_356: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_75, 2, 19)
        neg_40: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_26);  div_26 = None
        select_355: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_75, 2, 18)
        mul_55: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_40, select_355);  neg_40 = select_355 = None
        add_47: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_356, mul_55);  select_356 = mul_55 = None
        convert_element_type_18: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        select_scatter_78: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_75, convert_element_type_18, 2, 19);  select_scatter_75 = convert_element_type_18 = None
        select_359: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_78, 2, 19);  select_359 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_349: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_76, 2, 19)
        select_scatter_77: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_76, select_349, 2, 19);  select_scatter_76 = select_349 = None
        select_366: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_77, 2, 20)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_361: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 20)
        select_363: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_77, 2, 19)
        div_27: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_361, select_363);  select_361 = select_363 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_41: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_27)
        select_365: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 19)
        mul_56: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_41, select_365);  neg_41 = select_365 = None
        add_48: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_366, mul_56);  select_366 = mul_56 = None
        select_scatter_80: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_77, add_48, 2, 20);  select_scatter_77 = add_48 = None
        select_369: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_80, 2, 20);  select_369 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_357: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_78, 2, 19)
        select_scatter_79: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_78, select_357, 2, 19);  select_scatter_78 = select_357 = None
        select_374: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_79, 2, 20)
        neg_42: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_27);  div_27 = None
        select_373: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_79, 2, 19)
        mul_57: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_42, select_373);  neg_42 = select_373 = None
        add_49: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_374, mul_57);  select_374 = mul_57 = None
        convert_element_type_19: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        select_scatter_82: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_79, convert_element_type_19, 2, 20);  select_scatter_79 = convert_element_type_19 = None
        select_377: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_82, 2, 20);  select_377 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_367: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_80, 2, 20)
        select_scatter_81: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_80, select_367, 2, 20);  select_scatter_80 = select_367 = None
        select_384: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_81, 2, 21)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_379: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 21)
        select_381: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_81, 2, 20)
        div_28: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_379, select_381);  select_379 = select_381 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_43: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_28)
        select_383: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 20)
        mul_58: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_43, select_383);  neg_43 = select_383 = None
        add_50: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_384, mul_58);  select_384 = mul_58 = None
        select_scatter_84: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_81, add_50, 2, 21);  select_scatter_81 = add_50 = None
        select_387: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_84, 2, 21);  select_387 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_375: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_82, 2, 20)
        select_scatter_83: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_82, select_375, 2, 20);  select_scatter_82 = select_375 = None
        select_392: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_83, 2, 21)
        neg_44: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_28);  div_28 = None
        select_391: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_83, 2, 20)
        mul_59: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_44, select_391);  neg_44 = select_391 = None
        add_51: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_392, mul_59);  select_392 = mul_59 = None
        convert_element_type_20: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        select_scatter_86: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_83, convert_element_type_20, 2, 21);  select_scatter_83 = convert_element_type_20 = None
        select_395: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_86, 2, 21);  select_395 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_385: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_84, 2, 21)
        select_scatter_85: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_84, select_385, 2, 21);  select_scatter_84 = select_385 = None
        select_402: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_85, 2, 22)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_397: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 22)
        select_399: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_85, 2, 21)
        div_29: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_397, select_399);  select_397 = select_399 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_45: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_29)
        select_401: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 21)
        mul_60: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_45, select_401);  neg_45 = select_401 = None
        add_52: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_402, mul_60);  select_402 = mul_60 = None
        select_scatter_88: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_85, add_52, 2, 22);  select_scatter_85 = add_52 = None
        select_405: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_88, 2, 22);  select_405 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_393: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_86, 2, 21)
        select_scatter_87: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_86, select_393, 2, 21);  select_scatter_86 = select_393 = None
        select_410: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_87, 2, 22)
        neg_46: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_29);  div_29 = None
        select_409: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_87, 2, 21)
        mul_61: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_46, select_409);  neg_46 = select_409 = None
        add_53: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_410, mul_61);  select_410 = mul_61 = None
        convert_element_type_21: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        select_scatter_90: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_87, convert_element_type_21, 2, 22);  select_scatter_87 = convert_element_type_21 = None
        select_413: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_90, 2, 22);  select_413 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_403: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_88, 2, 22)
        select_scatter_89: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_88, select_403, 2, 22);  select_scatter_88 = select_403 = None
        select_420: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_89, 2, 23)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_415: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 23)
        select_417: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_89, 2, 22)
        div_30: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_415, select_417);  select_415 = select_417 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_47: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_30)
        select_419: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 22)
        mul_62: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_47, select_419);  neg_47 = select_419 = None
        add_54: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_420, mul_62);  select_420 = mul_62 = None
        select_scatter_92: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_89, add_54, 2, 23);  select_scatter_89 = add_54 = None
        select_423: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_92, 2, 23);  select_423 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_411: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_90, 2, 22)
        select_scatter_91: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_90, select_411, 2, 22);  select_scatter_90 = select_411 = None
        select_428: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_91, 2, 23)
        neg_48: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_30);  div_30 = None
        select_427: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_91, 2, 22)
        mul_63: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_48, select_427);  neg_48 = select_427 = None
        add_55: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_428, mul_63);  select_428 = mul_63 = None
        convert_element_type_22: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        select_scatter_94: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_91, convert_element_type_22, 2, 23);  select_scatter_91 = convert_element_type_22 = None
        select_431: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_94, 2, 23);  select_431 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_421: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_92, 2, 23)
        select_scatter_93: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_92, select_421, 2, 23);  select_scatter_92 = select_421 = None
        select_438: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_93, 2, 24)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_433: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 24)
        select_435: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_93, 2, 23)
        div_31: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_433, select_435);  select_433 = select_435 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_49: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_31)
        select_437: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 23)
        mul_64: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_49, select_437);  neg_49 = select_437 = None
        add_56: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_438, mul_64);  select_438 = mul_64 = None
        select_scatter_96: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_93, add_56, 2, 24);  select_scatter_93 = add_56 = None
        select_441: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_96, 2, 24);  select_441 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_429: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_94, 2, 23)
        select_scatter_95: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_94, select_429, 2, 23);  select_scatter_94 = select_429 = None
        select_446: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_95, 2, 24)
        neg_50: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_31);  div_31 = None
        select_445: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_95, 2, 23)
        mul_65: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_50, select_445);  neg_50 = select_445 = None
        add_57: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_446, mul_65);  select_446 = mul_65 = None
        convert_element_type_23: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None
        select_scatter_98: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_95, convert_element_type_23, 2, 24);  select_scatter_95 = convert_element_type_23 = None
        select_449: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_98, 2, 24);  select_449 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_439: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_96, 2, 24)
        select_scatter_97: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_96, select_439, 2, 24);  select_scatter_96 = select_439 = None
        select_456: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_97, 2, 25)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:13 in solve_tridiag, code: w = a[..., i] / b[..., i - 1]
        select_451: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_15, 2, 25);  mul_15 = None
        select_453: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_97, 2, 24)
        div_32: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_451, select_453);  select_451 = select_453 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        neg_51: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_32)
        select_455: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 24)
        mul_66: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_51, select_455);  neg_51 = select_455 = None
        add_58: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_456, mul_66);  select_456 = mul_66 = None
        select_scatter_100: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_97, add_58, 2, 25);  select_scatter_97 = add_58 = None
        select_459: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_100, 2, 25);  select_459 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_447: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_98, 2, 24)
        select_scatter_99: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_98, select_447, 2, 24);  select_scatter_98 = select_447 = None
        select_464: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_99, 2, 25)
        neg_52: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(div_32);  div_32 = None
        select_463: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_99, 2, 24)
        mul_67: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_52, select_463);  neg_52 = select_463 = None
        add_59: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.add.Tensor(select_464, mul_67);  select_464 = mul_67 = None
        convert_element_type_24: "bf16[200, 200][200, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        select_scatter_102: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_99, convert_element_type_24, 2, 25);  select_scatter_99 = convert_element_type_24 = None
        select_467: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_102, 2, 25);  select_467 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:261 in integrate_tke, code: torch.tensor([0.0], device=tke.device), tke[2:-2, 2:-2, -1, taup1]
        _tensor_constant1: "f32[1][1]cuda:0" = self._tensor_constant1;  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:193 in integrate_tke, code: flux_east = torch.zeros_like(maskU)
        full: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:267 in integrate_tke, code: flux_east[:-1, :, :] = (
        slice_120: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full, 0, 0, -1);  slice_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:273 in integrate_tke, code: flux_east[-1, :, :] = 0.0
        _tensor_constant2: "bf16[][]cpu" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:281 in integrate_tke, code: flux_north[:, -1, :] = 0.0
        _tensor_constant3: "bf16[][]cpu" = self._tensor_constant3;  _tensor_constant3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:250 in integrate_tke, code: tke[2:-2, 2:-2, :, taup1] = torch.where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])
        slice_63: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_64: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_63, 1, 2, -2)
        slice_61: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_62: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_61, 1, 2, -2);  slice_61 = None
        select_726: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_62, 3, 1);  slice_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:17 in solve_tridiag, code: out = torch.empty_like(a)
        empty: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.empty.memory_format([200, 200, 26], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:18 in solve_tridiag, code: out[..., -1] = d[..., -1] / b[..., -1]
        select_473: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(empty, 2, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:15 in solve_tridiag, code: d[..., i] += -w * d[..., i - 1]
        select_465: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_102, 2, 25)
        select_scatter_103: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_102, select_465, 2, 25);  select_scatter_102 = select_465 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:18 in solve_tridiag, code: out[..., -1] = d[..., -1] / b[..., -1]
        select_471: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:14 in solve_tridiag, code: b[..., i] += -w * c[..., i - 1]
        select_457: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_100, 2, 25)
        select_scatter_101: "f32[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_100, select_457, 2, 25);  select_scatter_100 = select_457 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:18 in solve_tridiag, code: out[..., -1] = d[..., -1] / b[..., -1]
        select_472: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, -1)
        div_33: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(select_471, select_472);  select_471 = select_472 = None
        copy_58: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_473, div_33);  select_473 = div_33 = None
        select_scatter_104: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(empty, copy_58, 2, -1);  empty = copy_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:21 in solve_tridiag, code: out[..., i] = (d[..., i] - c[..., i] * out[..., i + 1]) / b[..., i]
        select_483: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_104, 2, 24)
        select_479: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 24)
        select_476: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 24)
        select_478: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_104, 2, 25)
        mul_68: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_476, select_478);  select_476 = select_478 = None
        sub_1: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_479, mul_68);  select_479 = mul_68 = None
        select_481: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 24)
        div_34: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_1, select_481);  sub_1 = select_481 = None
        copy_59: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_483, div_34);  select_483 = div_34 = None
        select_scatter_105: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_104, copy_59, 2, 24);  select_scatter_104 = copy_59 = None
        select_493: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_105, 2, 23)
        select_489: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 23)
        select_486: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 23)
        select_488: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_105, 2, 24)
        mul_69: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_486, select_488);  select_486 = select_488 = None
        sub_2: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_489, mul_69);  select_489 = mul_69 = None
        select_491: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 23)
        div_35: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_2, select_491);  sub_2 = select_491 = None
        copy_60: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_493, div_35);  select_493 = div_35 = None
        select_scatter_106: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_105, copy_60, 2, 23);  select_scatter_105 = copy_60 = None
        select_503: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_106, 2, 22)
        select_499: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 22)
        select_496: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 22)
        select_498: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_106, 2, 23)
        mul_70: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_496, select_498);  select_496 = select_498 = None
        sub_3: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_499, mul_70);  select_499 = mul_70 = None
        select_501: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 22)
        div_36: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_3, select_501);  sub_3 = select_501 = None
        copy_61: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_503, div_36);  select_503 = div_36 = None
        select_scatter_107: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_106, copy_61, 2, 22);  select_scatter_106 = copy_61 = None
        select_513: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_107, 2, 21)
        select_509: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 21)
        select_506: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 21)
        select_508: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_107, 2, 22)
        mul_71: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_506, select_508);  select_506 = select_508 = None
        sub_4: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_509, mul_71);  select_509 = mul_71 = None
        select_511: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 21)
        div_37: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_4, select_511);  sub_4 = select_511 = None
        copy_62: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_513, div_37);  select_513 = div_37 = None
        select_scatter_108: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_107, copy_62, 2, 21);  select_scatter_107 = copy_62 = None
        select_523: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_108, 2, 20)
        select_519: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 20)
        select_516: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 20)
        select_518: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_108, 2, 21)
        mul_72: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_516, select_518);  select_516 = select_518 = None
        sub_5: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_519, mul_72);  select_519 = mul_72 = None
        select_521: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 20)
        div_38: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_5, select_521);  sub_5 = select_521 = None
        copy_63: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_523, div_38);  select_523 = div_38 = None
        select_scatter_109: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_108, copy_63, 2, 20);  select_scatter_108 = copy_63 = None
        select_533: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_109, 2, 19)
        select_529: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 19)
        select_526: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 19)
        select_528: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_109, 2, 20)
        mul_73: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_526, select_528);  select_526 = select_528 = None
        sub_6: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_529, mul_73);  select_529 = mul_73 = None
        select_531: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 19)
        div_39: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_6, select_531);  sub_6 = select_531 = None
        copy_64: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_533, div_39);  select_533 = div_39 = None
        select_scatter_110: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_109, copy_64, 2, 19);  select_scatter_109 = copy_64 = None
        select_543: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_110, 2, 18)
        select_539: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 18)
        select_536: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 18)
        select_538: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_110, 2, 19)
        mul_74: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_536, select_538);  select_536 = select_538 = None
        sub_7: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_539, mul_74);  select_539 = mul_74 = None
        select_541: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 18)
        div_40: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_7, select_541);  sub_7 = select_541 = None
        copy_65: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_543, div_40);  select_543 = div_40 = None
        select_scatter_111: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_110, copy_65, 2, 18);  select_scatter_110 = copy_65 = None
        select_553: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_111, 2, 17)
        select_549: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 17)
        select_546: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 17)
        select_548: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_111, 2, 18)
        mul_75: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_546, select_548);  select_546 = select_548 = None
        sub_8: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_549, mul_75);  select_549 = mul_75 = None
        select_551: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 17)
        div_41: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_8, select_551);  sub_8 = select_551 = None
        copy_66: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_553, div_41);  select_553 = div_41 = None
        select_scatter_112: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_111, copy_66, 2, 17);  select_scatter_111 = copy_66 = None
        select_563: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_112, 2, 16)
        select_559: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 16)
        select_556: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 16)
        select_558: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_112, 2, 17)
        mul_76: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_556, select_558);  select_556 = select_558 = None
        sub_9: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_559, mul_76);  select_559 = mul_76 = None
        select_561: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 16)
        div_42: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_9, select_561);  sub_9 = select_561 = None
        copy_67: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_563, div_42);  select_563 = div_42 = None
        select_scatter_113: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_112, copy_67, 2, 16);  select_scatter_112 = copy_67 = None
        select_573: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_113, 2, 15)
        select_569: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 15)
        select_566: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 15)
        select_568: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_113, 2, 16)
        mul_77: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_566, select_568);  select_566 = select_568 = None
        sub_10: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_569, mul_77);  select_569 = mul_77 = None
        select_571: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 15)
        div_43: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_10, select_571);  sub_10 = select_571 = None
        copy_68: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_573, div_43);  select_573 = div_43 = None
        select_scatter_114: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_113, copy_68, 2, 15);  select_scatter_113 = copy_68 = None
        select_583: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_114, 2, 14)
        select_579: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 14)
        select_576: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 14)
        select_578: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_114, 2, 15)
        mul_78: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_576, select_578);  select_576 = select_578 = None
        sub_11: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_579, mul_78);  select_579 = mul_78 = None
        select_581: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 14)
        div_44: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_11, select_581);  sub_11 = select_581 = None
        copy_69: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_583, div_44);  select_583 = div_44 = None
        select_scatter_115: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_114, copy_69, 2, 14);  select_scatter_114 = copy_69 = None
        select_593: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_115, 2, 13)
        select_589: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 13)
        select_586: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 13)
        select_588: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_115, 2, 14)
        mul_79: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_586, select_588);  select_586 = select_588 = None
        sub_12: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_589, mul_79);  select_589 = mul_79 = None
        select_591: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 13)
        div_45: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_12, select_591);  sub_12 = select_591 = None
        copy_70: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_593, div_45);  select_593 = div_45 = None
        select_scatter_116: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_115, copy_70, 2, 13);  select_scatter_115 = copy_70 = None
        select_603: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_116, 2, 12)
        select_599: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 12)
        select_596: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 12)
        select_598: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_116, 2, 13)
        mul_80: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_596, select_598);  select_596 = select_598 = None
        sub_13: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_599, mul_80);  select_599 = mul_80 = None
        select_601: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 12)
        div_46: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_13, select_601);  sub_13 = select_601 = None
        copy_71: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_603, div_46);  select_603 = div_46 = None
        select_scatter_117: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_116, copy_71, 2, 12);  select_scatter_116 = copy_71 = None
        select_613: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_117, 2, 11)
        select_609: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 11)
        select_606: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 11)
        select_608: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_117, 2, 12)
        mul_81: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_606, select_608);  select_606 = select_608 = None
        sub_14: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_609, mul_81);  select_609 = mul_81 = None
        select_611: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 11)
        div_47: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_14, select_611);  sub_14 = select_611 = None
        copy_72: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_613, div_47);  select_613 = div_47 = None
        select_scatter_118: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_117, copy_72, 2, 11);  select_scatter_117 = copy_72 = None
        select_623: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_118, 2, 10)
        select_619: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 10)
        select_616: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 10)
        select_618: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_118, 2, 11)
        mul_82: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_616, select_618);  select_616 = select_618 = None
        sub_15: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_619, mul_82);  select_619 = mul_82 = None
        select_621: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 10)
        div_48: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_15, select_621);  sub_15 = select_621 = None
        copy_73: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_623, div_48);  select_623 = div_48 = None
        select_scatter_119: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_118, copy_73, 2, 10);  select_scatter_118 = copy_73 = None
        select_633: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_119, 2, 9)
        select_629: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 9)
        select_626: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 9)
        select_628: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_119, 2, 10)
        mul_83: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_626, select_628);  select_626 = select_628 = None
        sub_16: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_629, mul_83);  select_629 = mul_83 = None
        select_631: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 9)
        div_49: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_16, select_631);  sub_16 = select_631 = None
        copy_74: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_633, div_49);  select_633 = div_49 = None
        select_scatter_120: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_119, copy_74, 2, 9);  select_scatter_119 = copy_74 = None
        select_643: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_120, 2, 8)
        select_639: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 8)
        select_636: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 8)
        select_638: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_120, 2, 9)
        mul_84: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_636, select_638);  select_636 = select_638 = None
        sub_17: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_639, mul_84);  select_639 = mul_84 = None
        select_641: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 8)
        div_50: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_17, select_641);  sub_17 = select_641 = None
        copy_75: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_643, div_50);  select_643 = div_50 = None
        select_scatter_121: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_120, copy_75, 2, 8);  select_scatter_120 = copy_75 = None
        select_653: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_121, 2, 7)
        select_649: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 7)
        select_646: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 7)
        select_648: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_121, 2, 8)
        mul_85: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_646, select_648);  select_646 = select_648 = None
        sub_18: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_649, mul_85);  select_649 = mul_85 = None
        select_651: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 7)
        div_51: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_18, select_651);  sub_18 = select_651 = None
        copy_76: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_653, div_51);  select_653 = div_51 = None
        select_scatter_122: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_121, copy_76, 2, 7);  select_scatter_121 = copy_76 = None
        select_663: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_122, 2, 6)
        select_659: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 6)
        select_656: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 6)
        select_658: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_122, 2, 7)
        mul_86: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_656, select_658);  select_656 = select_658 = None
        sub_19: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_659, mul_86);  select_659 = mul_86 = None
        select_661: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 6)
        div_52: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_19, select_661);  sub_19 = select_661 = None
        copy_77: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_663, div_52);  select_663 = div_52 = None
        select_scatter_123: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_122, copy_77, 2, 6);  select_scatter_122 = copy_77 = None
        select_673: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_123, 2, 5)
        select_669: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 5)
        select_666: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 5)
        select_668: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_123, 2, 6)
        mul_87: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_666, select_668);  select_666 = select_668 = None
        sub_20: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_669, mul_87);  select_669 = mul_87 = None
        select_671: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 5)
        div_53: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_20, select_671);  sub_20 = select_671 = None
        copy_78: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_673, div_53);  select_673 = div_53 = None
        select_scatter_124: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_123, copy_78, 2, 5);  select_scatter_123 = copy_78 = None
        select_683: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_124, 2, 4)
        select_679: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 4)
        select_676: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 4)
        select_678: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_124, 2, 5)
        mul_88: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_676, select_678);  select_676 = select_678 = None
        sub_21: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_679, mul_88);  select_679 = mul_88 = None
        select_681: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 4)
        div_54: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_21, select_681);  sub_21 = select_681 = None
        copy_79: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_683, div_54);  select_683 = div_54 = None
        select_scatter_125: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_124, copy_79, 2, 4);  select_scatter_124 = copy_79 = None
        select_693: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_125, 2, 3)
        select_689: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 3)
        select_686: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 3)
        select_688: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_125, 2, 4)
        mul_89: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_686, select_688);  select_686 = select_688 = None
        sub_22: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_689, mul_89);  select_689 = mul_89 = None
        select_691: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 3)
        div_55: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_22, select_691);  sub_22 = select_691 = None
        copy_80: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_693, div_55);  select_693 = div_55 = None
        select_scatter_126: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_125, copy_80, 2, 3);  select_scatter_125 = copy_80 = None
        select_703: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_126, 2, 2)
        select_699: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 2)
        select_696: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 2)
        select_698: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_126, 2, 3)
        mul_90: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_696, select_698);  select_696 = select_698 = None
        sub_23: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_699, mul_90);  select_699 = mul_90 = None
        select_701: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 2)
        div_56: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_23, select_701);  sub_23 = select_701 = None
        copy_81: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_703, div_56);  select_703 = div_56 = None
        select_scatter_127: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_126, copy_81, 2, 2);  select_scatter_126 = copy_81 = None
        select_713: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_127, 2, 1)
        select_709: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 1)
        select_706: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 1)
        select_708: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_127, 2, 2)
        mul_91: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_706, select_708);  select_706 = select_708 = None
        sub_24: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_709, mul_91);  select_709 = mul_91 = None
        select_711: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 1)
        div_57: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_24, select_711);  sub_24 = select_711 = None
        copy_82: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_713, div_57);  select_713 = div_57 = None
        select_scatter_128: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_127, copy_82, 2, 1);  select_scatter_127 = copy_82 = None
        select_723: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_128, 2, 0)
        select_719: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_103, 2, 0);  select_scatter_103 = None
        select_716: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(mul_16, 2, 0);  mul_16 = None
        select_718: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_128, 2, 1)
        mul_92: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_716, select_718);  select_716 = select_718 = None
        sub_25: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_719, mul_92);  select_719 = mul_92 = None
        select_721: "f32[200, 200][5200, 26]cuda:0" = torch.ops.aten.select.int(select_scatter_101, 2, 0);  select_scatter_101 = None
        div_58: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_25, select_721);  sub_25 = select_721 = None
        copy_83: "bf16[200, 200][5200, 26]cuda:0" = torch.ops.aten.copy.default(select_723, div_58);  select_723 = div_58 = None
        select_scatter_129: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_128, copy_83, 2, 0);  select_scatter_128 = copy_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:250 in integrate_tke, code: tke[2:-2, 2:-2, :, taup1] = torch.where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])
        slice_59: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_60: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_59, 1, 2, -2);  slice_59 = None
        select_725: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_60, 3, 1);  slice_60 = None
        where_2: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.where.self(bitwise_and_1, select_scatter_129, select_725);  bitwise_and_1 = select_scatter_129 = select_725 = None
        copy_84: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.copy.default(select_726, where_2);  select_726 = where_2 = None
        select_scatter_130: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_64, copy_84, 3, 1);  slice_64 = copy_84 = None
        slice_scatter_4: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_63, select_scatter_130, 1, 2, -2);  slice_63 = select_scatter_130 = None
        slice_scatter_5: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(arg0_1, slice_scatter_4, 0, 2, -2);  arg0_1 = slice_scatter_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:260 in integrate_tke, code: tke[2:-2, 2:-2, -1, taup1] = torch.maximum(
        slice_108: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_109: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_108, 1, 2, -2)
        select_749: "bf16[200, 200, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_109, 2, -1)
        slice_106: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_107: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_106, 1, 2, -2);  slice_106 = None
        select_747: "bf16[200, 200, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_107, 2, -1);  slice_107 = None
        select_748: "bf16[200, 200][15912, 78]cuda:0" = torch.ops.aten.select.int(select_747, 2, 1);  select_747 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:261 in integrate_tke, code: torch.tensor([0.0], device=tke.device), tke[2:-2, 2:-2, -1, taup1]
        full_default_5: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_97: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_98: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_97, 1, 2, -2);  slice_97 = None
        select_742: "bf16[200, 200, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_98, 2, -1);  slice_98 = None
        select_743: "bf16[200, 200][15912, 78]cuda:0" = torch.ops.aten.select.int(select_742, 2, 1);  select_742 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:260 in integrate_tke, code: tke[2:-2, 2:-2, -1, taup1] = torch.maximum(
        maximum_1: "f32[200, 200][200, 1]cuda:0" = torch.ops.aten.maximum.default(full_default_5, select_743);  full_default_5 = select_743 = None
        copy_86: "bf16[200, 200][15912, 78]cuda:0" = torch.ops.aten.copy.default(select_748, maximum_1);  select_748 = maximum_1 = None
        select_scatter_131: "bf16[200, 200, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_749, copy_86, 2, 1);  select_749 = copy_86 = None
        select_scatter_132: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_109, select_scatter_131, 2, -1);  slice_109 = select_scatter_131 = None
        slice_scatter_8: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_108, select_scatter_132, 1, 2, -2);  slice_108 = select_scatter_132 = None
        slice_scatter_9: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_scatter_5, slice_scatter_8, 0, 2, -2);  slice_scatter_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:282 in integrate_tke, code: tke[2:-2, 2:-2, :, taup1] += (
        slice_166: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_9, 0, 2, -2)
        slice_167: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_166, 1, 2, -2)
        slice_164: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_9, 0, 2, -2)
        slice_165: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_164, 1, 2, -2);  slice_164 = None
        select_767: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_165, 3, 1);  slice_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:284 in integrate_tke, code: * maskW[2:-2, 2:-2, :]
        slice_138: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_139: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_138, 1, 2, -2);  slice_138 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:283 in integrate_tke, code: dt_tke
        mul_101: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_139, 1);  slice_139 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:269 in integrate_tke, code: * (tke[1:, :, :, tau] - tke[:-1, :, :, tau])
        slice_116: "bf16[203, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_9, 0, 1, 9223372036854775807)
        select_754: "bf16[203, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_116, 3, 0);  slice_116 = None
        slice_117: "bf16[203, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_9, 0, 0, -1)
        select_755: "bf16[203, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_117, 3, 0);  slice_117 = None
        sub_26: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_754, select_755);  select_754 = select_755 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:268 in integrate_tke, code: K_h_tke
        mul_95: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, 2000.0);  sub_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:270 in integrate_tke, code: / (cost[None, :, None] * dxu[:-1, None, None])
        unsqueeze_17: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        unsqueeze_18: "bf16[1, 204, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 2);  unsqueeze_17 = None
        slice_118: "bf16[203][1]cuda:0" = torch.ops.aten.slice.Tensor(arg11_1, 0, 0, -1);  arg11_1 = None
        unsqueeze_19: "bf16[203, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_118, 1);  slice_118 = None
        unsqueeze_20: "bf16[203, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        mul_96: "bf16[203, 204, 1][204, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_18, unsqueeze_20);  unsqueeze_18 = unsqueeze_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:268 in integrate_tke, code: K_h_tke
        div_60: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:271 in integrate_tke, code: * maskU[:-1, :, :]
        slice_119: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, -1);  arg2_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:268 in integrate_tke, code: K_h_tke
        mul_97: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, slice_119);  div_60 = slice_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:267 in integrate_tke, code: flux_east[:-1, :, :] = (
        slice_scatter_10: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full, mul_97, 0, 0, -1);  full = mul_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:273 in integrate_tke, code: flux_east[-1, :, :] = 0.0
        select_757: "bf16[204, 26][26, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_10, 0, -1)
        full_default_6: "bf16[][]cpu" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_88: "bf16[204, 26][26, 1]cuda:0" = torch.ops.aten.copy.default(select_757, full_default_6);  select_757 = full_default_6 = None
        select_scatter_133: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_10, copy_88, 0, -1);  slice_scatter_10 = copy_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:286 in integrate_tke, code: (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        slice_146: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_133, 0, 2, -2)
        slice_147: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_146, 1, 2, -2);  slice_146 = None
        slice_148: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_133, 0, 1, -3)
        slice_149: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_148, 1, 2, -2);  slice_148 = None
        sub_28: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_147, slice_149);  slice_147 = slice_149 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:287 in integrate_tke, code: / (cost[None, 2:-2, None] * dxt[2:-2, None, None])
        unsqueeze_25: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_150: "bf16[1, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_25, 1, 2, -2);  unsqueeze_25 = None
        unsqueeze_26: "bf16[1, 200, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_150, 2);  slice_150 = None
        slice_151: "bf16[200][1]cuda:0" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        unsqueeze_27: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_151, 1);  slice_151 = None
        unsqueeze_28: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
        mul_102: "bf16[200, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_26, unsqueeze_28);  unsqueeze_26 = unsqueeze_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:286 in integrate_tke, code: (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        div_62: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_28, mul_102);  sub_28 = mul_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:194 in integrate_tke, code: flux_north = torch.zeros_like(maskU)
        full_1: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:274 in integrate_tke, code: flux_north[:, :-1, :] = (
        slice_131: "bf16[204, 203, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_1, 1, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:276 in integrate_tke, code: * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau])
        slice_126: "bf16[204, 203, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_9, 1, 1, 9223372036854775807)
        select_761: "bf16[204, 203, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_126, 3, 0);  slice_126 = None
        slice_127: "bf16[204, 203, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_9, 1, 0, -1)
        select_762: "bf16[204, 203, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_127, 3, 0);  slice_127 = None
        sub_27: "bf16[204, 203, 26][5278, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_761, select_762);  select_761 = select_762 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:275 in integrate_tke, code: K_h_tke
        mul_98: "bf16[204, 203, 26][5278, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, 2000.0);  sub_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:277 in integrate_tke, code: / dyu[None, :-1, None]
        unsqueeze_21: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg12_1, 0);  arg12_1 = None
        slice_128: "bf16[1, 203][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_21, 1, 0, -1);  unsqueeze_21 = None
        unsqueeze_22: "bf16[1, 203, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_128, 2);  slice_128 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:275 in integrate_tke, code: K_h_tke
        div_61: "bf16[204, 203, 26][5278, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_98, unsqueeze_22);  mul_98 = unsqueeze_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:278 in integrate_tke, code: * maskV[:, :-1, :]
        slice_129: "bf16[204, 203, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg13_1, 1, 0, -1);  arg13_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:275 in integrate_tke, code: K_h_tke
        mul_99: "bf16[204, 203, 26][5278, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, slice_129);  div_61 = slice_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:279 in integrate_tke, code: * cosu[None, :-1, None]
        unsqueeze_23: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        slice_130: "bf16[1, 203][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_23, 1, 0, -1);  unsqueeze_23 = None
        unsqueeze_24: "bf16[1, 203, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_130, 2);  slice_130 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:275 in integrate_tke, code: K_h_tke
        mul_100: "bf16[204, 203, 26][5278, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_24);  mul_99 = unsqueeze_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:274 in integrate_tke, code: flux_north[:, :-1, :] = (
        copy_89: "bf16[204, 203, 26][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_131, mul_100);  slice_131 = mul_100 = None
        slice_scatter_11: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_1, copy_89, 1, 0, -1);  full_1 = copy_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:281 in integrate_tke, code: flux_north[:, -1, :] = 0.0
        select_764: "bf16[204, 26][5304, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_11, 1, -1)
        full_default_7: "bf16[][]cpu" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_90: "bf16[204, 26][5304, 1]cuda:0" = torch.ops.aten.copy.default(select_764, full_default_7);  select_764 = full_default_7 = None
        select_scatter_134: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_11, copy_90, 1, -1);  slice_scatter_11 = copy_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:288 in integrate_tke, code: + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
        slice_158: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_134, 0, 2, -2)
        slice_159: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_158, 1, 2, -2);  slice_158 = None
        slice_160: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_134, 0, 2, -2)
        slice_161: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_160, 1, 1, -3);  slice_160 = None
        sub_29: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_159, slice_161);  slice_159 = slice_161 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:289 in integrate_tke, code: / (cost[None, 2:-2, None] * dyt[None, 2:-2, None])
        unsqueeze_29: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_162: "bf16[1, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_29, 1, 2, -2);  unsqueeze_29 = None
        unsqueeze_30: "bf16[1, 200, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_162, 2);  slice_162 = None
        unsqueeze_31: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg17_1, 0)
        slice_163: "bf16[1, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_31, 1, 2, -2);  unsqueeze_31 = None
        unsqueeze_32: "bf16[1, 200, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_163, 2);  slice_163 = None
        mul_103: "bf16[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_30, unsqueeze_32);  unsqueeze_30 = unsqueeze_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:288 in integrate_tke, code: + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
        div_63: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_29, mul_103);  sub_29 = mul_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:286 in integrate_tke, code: (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        add_60: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(div_62, div_63);  div_62 = div_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:283 in integrate_tke, code: dt_tke
        mul_104: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, add_60);  mul_101 = add_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:282 in integrate_tke, code: tke[2:-2, 2:-2, :, taup1] += (
        add_61: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(select_767, mul_104);  select_767 = mul_104 = None
        select_scatter_135: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_167, add_61, 3, 1);  slice_167 = add_61 = None
        slice_scatter_12: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_166, select_scatter_135, 1, 2, -2);  slice_166 = select_scatter_135 = None
        slice_scatter_13: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_scatter_9, slice_scatter_12, 0, 2, -2);  slice_scatter_9 = slice_scatter_12 = None
        slice_175: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 2, -2)
        slice_176: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_175, 1, 2, -2);  slice_175 = None
        select_770: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_176, 3, 1);  slice_176 = select_770 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:137 in adv_flux_superbee_wgrid, code: maskUtr = torch.zeros_like(maskW)
        full_9: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:138 in adv_flux_superbee_wgrid, code: maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]
        slice_183: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_9, 0, 0, -1);  slice_183 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:139 in adv_flux_superbee_wgrid, code: adv_fe[...] = 0.0
        _tensor_constant4: "bf16[][]cpu" = self._tensor_constant4;  _tensor_constant4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:61 in limiter, code: torch.tensor([0.0], device=cr.device),
        _tensor_constant5: "f32[1][1]cuda:0" = self._tensor_constant5;  _tensor_constant5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        _tensor_constant6: "f32[1][1]cuda:0" = self._tensor_constant6;  _tensor_constant6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:64 in limiter, code: torch.minimum(torch.tensor([2.0], device=cr.device), cr),
        _tensor_constant7: "f32[1][1]cuda:0" = self._tensor_constant7;  _tensor_constant7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:146 in adv_flux_superbee_wgrid, code: adv_fn[...] = 0.0
        _tensor_constant8: "bf16[][]cpu" = self._tensor_constant8;  _tensor_constant8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:61 in limiter, code: torch.tensor([0.0], device=cr.device),
        _tensor_constant9: "f32[1][1]cuda:0" = self._tensor_constant9;  _tensor_constant9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        _tensor_constant10: "f32[1][1]cuda:0" = self._tensor_constant10;  _tensor_constant10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:64 in limiter, code: torch.minimum(torch.tensor([2.0], device=cr.device), cr),
        _tensor_constant11: "f32[1][1]cuda:0" = self._tensor_constant11;  _tensor_constant11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:153 in adv_flux_superbee_wgrid, code: adv_ft[...] = 0.0
        _tensor_constant12: "bf16[][]cpu" = self._tensor_constant12;  _tensor_constant12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:61 in limiter, code: torch.tensor([0.0], device=cr.device),
        _tensor_constant13: "f32[1][1]cuda:0" = self._tensor_constant13;  _tensor_constant13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        _tensor_constant14: "f32[1][1]cuda:0" = self._tensor_constant14;  _tensor_constant14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:64 in limiter, code: torch.minimum(torch.tensor([2.0], device=cr.device), cr),
        _tensor_constant15: "f32[1][1]cuda:0" = self._tensor_constant15;  _tensor_constant15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:313 in integrate_tke, code: dtke[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (
        slice_523: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg1_1, 0, 2, -2)
        slice_524: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_523, 1, 2, -2)
        slice_521: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg1_1, 0, 2, -2)
        slice_522: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_521, 1, 2, -2);  slice_521 = None
        select_809: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_522, 3, 0);  slice_522 = None
        slice_495: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_496: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_495, 1, 2, -2);  slice_495 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:139 in adv_flux_superbee_wgrid, code: adv_fe[...] = 0.0
        full_default_8: "bf16[][]cpu" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_93: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(select_scatter_133, full_default_8);  select_scatter_133 = full_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:140 in adv_flux_superbee_wgrid, code: adv_fe[1:-2, 2:-2, :] = _adv_superbee(
        slice_255: "bf16[201, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(copy_93, 0, 1, -2)
        slice_253: "bf16[201, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(copy_93, 0, 1, -2)
        slice_254: "bf16[201, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_253, 1, 2, -2);  slice_253 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:301 in integrate_tke, code: u[..., tau],
        select_773: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(arg18_1, 3, 0);  arg18_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:78 in _adv_superbee, code: vel[1:-2, 2:-2, :] * (var[2:-1, 2:-2, :] + var[1:-2, 2:-2, :]) * 0.5
        slice_236: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_773, 0, 1, -2)
        slice_237: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_236, 1, 2, -2);  slice_236 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:282 in integrate_tke, code: tke[2:-2, 2:-2, :, taup1] += (
        slice_177: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 2, -2)
        slice_178: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_177, 1, 2, -2)
        slice_168: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 2, -2)
        slice_169: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_168, 1, 2, -2);  slice_168 = None
        select_768: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_169, 3, 1);  slice_169 = None
        select_scatter_136: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_178, select_768, 3, 1);  slice_178 = select_768 = None
        slice_scatter_14: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_177, select_scatter_136, 1, 2, -2);  slice_177 = select_scatter_136 = None
        slice_scatter_15: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_scatter_13, slice_scatter_14, 0, 2, -2);  slice_scatter_13 = slice_scatter_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:78 in _adv_superbee, code: vel[1:-2, 2:-2, :] * (var[2:-1, 2:-2, :] + var[1:-2, 2:-2, :]) * 0.5
        select_791: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_244: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_791, 0, 2, -1);  select_791 = None
        slice_245: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_244, 1, 2, -2);  slice_244 = None
        select_792: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_246: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_792, 0, 1, -2);  select_792 = None
        slice_247: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_246, 1, 2, -2);  slice_246 = None
        add_62: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_245, slice_247);  slice_245 = slice_247 = None
        mul_112: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_237, add_62);  slice_237 = add_62 = None
        mul_113: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, 0.5);  mul_112 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:79 in _adv_superbee, code: - torch.abs(vel[1:-2, 2:-2, :]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
        slice_248: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_773, 0, 1, -2)
        slice_249: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_248, 1, 2, -2);  slice_248 = None
        abs_3: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.abs.default(slice_249);  slice_249 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:61 in limiter, code: torch.tensor([0.0], device=cr.device),
        full_default_10: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        full_default_11: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:76 in _adv_superbee, code: cr = limiter(_calc_cr(rjp, rj, rjm, vel[1:-2, 2:-2, :]))
        slice_234: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_773, 0, 1, -2)
        slice_235: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_234, 1, 2, -2);  slice_234 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        gt: "b8[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.gt.Scalar(slice_235, 0.0);  slice_235 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:75 in _adv_superbee, code: rjm = (var[1:-2, 2:-2, :] - var[:-3, 2:-2, :]) * mask[:-3, 2:-2, :]
        select_787: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_225: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_787, 0, 1, -2);  select_787 = None
        slice_226: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_225, 1, 2, -2);  slice_225 = None
        select_788: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_227: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_788, 0, 0, -3);  select_788 = None
        slice_228: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_227, 1, 2, -2);  slice_227 = None
        sub_32: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_226, slice_228);  slice_226 = slice_228 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:138 in adv_flux_superbee_wgrid, code: maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]
        slice_181: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 0, 1, 9223372036854775807)
        slice_182: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 0, 0, -1)
        mul_105: "bf16[203, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_181, slice_182);  slice_181 = slice_182 = None
        slice_scatter_16: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_9, mul_105, 0, 0, -1);  full_9 = mul_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:75 in _adv_superbee, code: rjm = (var[1:-2, 2:-2, :] - var[:-3, 2:-2, :]) * mask[:-3, 2:-2, :]
        slice_232: "bf16[201, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_16, 0, 0, -3)
        slice_233: "bf16[201, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_232, 1, 2, -2);  slice_232 = None
        mul_110: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, slice_233);  sub_32 = slice_233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:73 in _adv_superbee, code: rjp = (var[3:, 2:-2, :] - var[2:-1, 2:-2, :]) * mask[2:-1, 2:-2, :]
        select_779: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_195: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_779, 0, 3, 9223372036854775807);  select_779 = None
        slice_196: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_195, 1, 2, -2);  slice_195 = None
        select_780: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_197: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_780, 0, 2, -1);  select_780 = None
        slice_198: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_197, 1, 2, -2);  slice_197 = None
        sub_30: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_196, slice_198);  slice_196 = slice_198 = None
        slice_202: "bf16[201, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_16, 0, 2, -1)
        slice_203: "bf16[201, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_202, 1, 2, -2);  slice_202 = None
        mul_108: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, slice_203);  sub_30 = slice_203 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        where_4: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.where.self(gt, mul_110, mul_108);  gt = mul_110 = mul_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:74 in _adv_superbee, code: rj = (var[2:-1, 2:-2, :] - var[1:-2, 2:-2, :]) * mask[1:-2, 2:-2, :]
        select_783: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_210: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_783, 0, 2, -1);  select_783 = None
        slice_211: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_210, 1, 2, -2);  slice_210 = None
        select_784: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_212: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_784, 0, 1, -2);  select_784 = None
        slice_213: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_212, 1, 2, -2);  slice_212 = None
        sub_31: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_211, slice_213);  slice_211 = slice_213 = None
        slice_217: "bf16[201, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_16, 0, 1, -2);  slice_scatter_16 = None
        slice_218: "bf16[201, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_217, 1, 2, -2);  slice_217 = None
        mul_109: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, slice_218);  sub_31 = slice_218 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        abs_2: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.abs.default(mul_109)
        lt_1: "b8[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.lt.Scalar(abs_2, 1e-20);  abs_2 = None
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 1.0005576689441423e-20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.where.self(lt_1, full_default_9, mul_109);  lt_1 = full_default_9 = None
        div_65: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(where_4, where_5);  where_4 = where_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        mul_111: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_65, 2)
        minimum: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.minimum.default(full_default_11, mul_111);  full_default_11 = mul_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:64 in limiter, code: torch.minimum(torch.tensor([2.0], device=cr.device), cr),
        full_default_12: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.minimum.default(full_default_12, div_65);  full_default_12 = div_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:62 in limiter, code: torch.maximum(
        maximum_2: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.maximum.default(minimum, minimum_1);  minimum = minimum_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:60 in limiter, code: return torch.maximum(
        maximum_3: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.maximum.default(full_default_10, maximum_2);  full_default_10 = maximum_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:79 in _adv_superbee, code: - torch.abs(vel[1:-2, 2:-2, :]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
        sub_33: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, maximum_3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:72 in _adv_superbee, code: uCFL = torch.abs(vel[1:-2, 2:-2, :] * dt_tracer / dx)
        slice_187: "bf16[201, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_773, 0, 1, -2);  select_773 = None
        slice_188: "bf16[201, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_187, 1, 2, -2);  slice_187 = None
        mul_107: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_188, 1.0);  slice_188 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:71 in _adv_superbee, code: dx = cost[None, 2:-2, None] * dx[1:-2, None, None]
        unsqueeze_33: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_185: "bf16[1, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_33, 1, 2, -2);  unsqueeze_33 = None
        unsqueeze_34: "bf16[1, 200, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_185, 2);  slice_185 = None
        slice_186: "bf16[201][1]cuda:0" = torch.ops.aten.slice.Tensor(arg16_1, 0, 1, -2)
        unsqueeze_35: "bf16[201, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_186, 1);  slice_186 = None
        unsqueeze_36: "bf16[201, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_35, 2);  unsqueeze_35 = None
        mul_106: "bf16[201, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_34, unsqueeze_36);  unsqueeze_34 = unsqueeze_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:72 in _adv_superbee, code: uCFL = torch.abs(vel[1:-2, 2:-2, :] * dt_tracer / dx)
        div_64: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_107, mul_106);  mul_107 = mul_106 = None
        abs_1: "bf16[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.abs.default(div_64);  div_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:79 in _adv_superbee, code: - torch.abs(vel[1:-2, 2:-2, :]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
        mul_114: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(abs_1, maximum_3);  abs_1 = maximum_3 = None
        add_63: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(sub_33, mul_114);  sub_33 = mul_114 = None
        mul_115: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(abs_3, add_63);  abs_3 = add_63 = None
        mul_116: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, mul_109);  mul_115 = mul_109 = None
        mul_117: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, 0.5);  mul_116 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:78 in _adv_superbee, code: vel[1:-2, 2:-2, :] * (var[2:-1, 2:-2, :] + var[1:-2, 2:-2, :]) * 0.5
        sub_34: "f32[201, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_113, mul_117);  mul_113 = mul_117 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:140 in adv_flux_superbee_wgrid, code: adv_fe[1:-2, 2:-2, :] = _adv_superbee(
        copy_94: "bf16[201, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_254, sub_34);  slice_254 = sub_34 = None
        slice_scatter_17: "bf16[201, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_255, copy_94, 1, 2, -2);  slice_255 = copy_94 = None
        slice_scatter_18: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(copy_93, slice_scatter_17, 0, 1, -2);  copy_93 = slice_scatter_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:314 in integrate_tke, code: -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        slice_503: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_18, 0, 2, -2)
        slice_504: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_503, 1, 2, -2);  slice_503 = None
        slice_505: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_18, 0, 1, -3);  slice_scatter_18 = None
        slice_506: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_505, 1, 2, -2);  slice_505 = None
        sub_45: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_504, slice_506);  slice_504 = slice_506 = None
        neg_54: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.neg.default(sub_45);  sub_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:315 in integrate_tke, code: / (cost[None, 2:-2, None] * dxt[2:-2, None, None])
        unsqueeze_43: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_507: "bf16[1, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_43, 1, 2, -2);  unsqueeze_43 = None
        unsqueeze_44: "bf16[1, 200, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_507, 2);  slice_507 = None
        slice_508: "bf16[200][1]cuda:0" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2);  arg16_1 = None
        unsqueeze_45: "bf16[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_508, 1);  slice_508 = None
        unsqueeze_46: "bf16[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        mul_146: "bf16[200, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_44, unsqueeze_46);  unsqueeze_44 = unsqueeze_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:314 in integrate_tke, code: -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        div_70: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(neg_54, mul_146);  neg_54 = mul_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:146 in adv_flux_superbee_wgrid, code: adv_fn[...] = 0.0
        full_default_13: "bf16[][]cpu" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_96: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(select_scatter_134, full_default_13);  select_scatter_134 = full_default_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:147 in adv_flux_superbee_wgrid, code: adv_fn[2:-2, 1:-2, :] = _adv_superbee(
        slice_332: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(copy_96, 0, 2, -2)
        slice_330: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(copy_96, 0, 2, -2)
        slice_331: "bf16[200, 201, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_330, 1, 1, -2);  slice_330 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:84 in _adv_superbee, code: velfac = cosu[None, 1:-2, None]
        unsqueeze_39: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        slice_263: "bf16[1, 201][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_39, 1, 1, -2);  unsqueeze_39 = None
        unsqueeze_40: "bf16[1, 201, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_263, 2);  slice_263 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:302 in integrate_tke, code: v[..., tau],
        select_774: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(arg19_1, 3, 0);  arg19_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:92 in _adv_superbee, code: * vel[2:-2, 1:-2, :]
        slice_313: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_774, 0, 2, -2)
        slice_314: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_313, 1, 1, -2);  slice_313 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:91 in _adv_superbee, code: velfac
        mul_126: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_40, slice_314);  slice_314 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:93 in _adv_superbee, code: * (var[2:-2, 2:-1, :] + var[2:-2, 1:-2, :])
        select_807: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_321: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_807, 0, 2, -2);  select_807 = None
        slice_322: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_321, 1, 2, -1);  slice_321 = None
        select_808: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_323: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_808, 0, 2, -2);  select_808 = None
        slice_324: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_323, 1, 1, -2);  slice_323 = None
        add_64: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_322, slice_324);  slice_322 = slice_324 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:91 in _adv_superbee, code: velfac
        mul_127: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, add_64);  mul_126 = add_64 = None
        mul_128: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, 0.5);  mul_127 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:95 in _adv_superbee, code: - torch.abs(velfac * vel[2:-2, 1:-2, :])
        slice_325: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_774, 0, 2, -2)
        slice_326: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_325, 1, 1, -2);  slice_325 = None
        mul_129: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_40, slice_326);  slice_326 = None
        abs_6: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.abs.default(mul_129);  mul_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:61 in limiter, code: torch.tensor([0.0], device=cr.device),
        full_default_15: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        full_default_16: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:89 in _adv_superbee, code: cr = limiter(_calc_cr(rjp, rj, rjm, vel[2:-2, 1:-2, :]))
        slice_311: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_774, 0, 2, -2)
        slice_312: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_311, 1, 1, -2);  slice_311 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        gt_1: "b8[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.gt.Scalar(slice_312, 0.0);  slice_312 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:88 in _adv_superbee, code: rjm = (var[2:-2, 1:-2, :] - var[2:-2, :-3, :]) * mask[2:-2, :-3, :]
        select_803: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_302: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_803, 0, 2, -2);  select_803 = None
        slice_303: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_302, 1, 1, -2);  slice_302 = None
        select_804: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_304: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_804, 0, 2, -2);  select_804 = None
        slice_305: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_304, 1, 0, -3);  slice_304 = None
        sub_37: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_303, slice_305);  slice_303 = slice_305 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:144 in adv_flux_superbee_wgrid, code: maskVtr = torch.zeros_like(maskW)
        full_10: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:145 in adv_flux_superbee_wgrid, code: maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]
        slice_260: "bf16[204, 203, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_10, 1, 0, -1)
        slice_258: "bf16[204, 203, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 1, 1, 9223372036854775807)
        slice_259: "bf16[204, 203, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 1, 0, -1)
        mul_118: "bf16[204, 203, 26][5278, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_258, slice_259);  slice_258 = slice_259 = None
        copy_95: "bf16[204, 203, 26][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_260, mul_118);  slice_260 = mul_118 = None
        slice_scatter_19: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_10, copy_95, 1, 0, -1);  full_10 = copy_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:88 in _adv_superbee, code: rjm = (var[2:-2, 1:-2, :] - var[2:-2, :-3, :]) * mask[2:-2, :-3, :]
        slice_309: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 2, -2)
        slice_310: "bf16[200, 201, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_309, 1, 0, -3);  slice_309 = None
        mul_124: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, slice_310);  sub_37 = slice_310 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:86 in _adv_superbee, code: rjp = (var[2:-2, 3:, :] - var[2:-2, 2:-1, :]) * mask[2:-2, 2:-1, :]
        select_795: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_272: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_795, 0, 2, -2);  select_795 = None
        slice_273: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_272, 1, 3, 9223372036854775807);  slice_272 = None
        select_796: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_274: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_796, 0, 2, -2);  select_796 = None
        slice_275: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_274, 1, 2, -1);  slice_274 = None
        sub_35: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_273, slice_275);  slice_273 = slice_275 = None
        slice_279: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 2, -2)
        slice_280: "bf16[200, 201, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_279, 1, 2, -1);  slice_279 = None
        mul_122: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, slice_280);  sub_35 = slice_280 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        where_6: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.where.self(gt_1, mul_124, mul_122);  gt_1 = mul_124 = mul_122 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:87 in _adv_superbee, code: rj = (var[2:-2, 2:-1, :] - var[2:-2, 1:-2, :]) * mask[2:-2, 1:-2, :]
        select_799: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_287: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_799, 0, 2, -2);  select_799 = None
        slice_288: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_287, 1, 2, -1);  slice_287 = None
        select_800: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)
        slice_289: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_800, 0, 2, -2);  select_800 = None
        slice_290: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_289, 1, 1, -2);  slice_289 = None
        sub_36: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_288, slice_290);  slice_288 = slice_290 = None
        slice_294: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 2, -2);  slice_scatter_19 = None
        slice_295: "bf16[200, 201, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_294, 1, 1, -2);  slice_294 = None
        mul_123: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, slice_295);  sub_36 = slice_295 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        abs_5: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.abs.default(mul_123)
        lt_2: "b8[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.lt.Scalar(abs_5, 1e-20);  abs_5 = None
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 1.0005576689441423e-20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.where.self(lt_2, full_default_14, mul_123);  lt_2 = full_default_14 = None
        div_67: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(where_6, where_7);  where_6 = where_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        mul_125: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, 2)
        minimum_2: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.minimum.default(full_default_16, mul_125);  full_default_16 = mul_125 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:64 in limiter, code: torch.minimum(torch.tensor([2.0], device=cr.device), cr),
        full_default_17: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_3: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.minimum.default(full_default_17, div_67);  full_default_17 = div_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:62 in limiter, code: torch.maximum(
        maximum_4: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.maximum.default(minimum_2, minimum_3);  minimum_2 = minimum_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:60 in limiter, code: return torch.maximum(
        maximum_5: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.maximum.default(full_default_15, maximum_4);  full_default_15 = maximum_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:96 in _adv_superbee, code: * ((1.0 - cr) + uCFL * cr)
        sub_38: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, maximum_5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:85 in _adv_superbee, code: uCFL = torch.abs(velfac * vel[2:-2, 1:-2, :] * dt_tracer / dx)
        slice_264: "bf16[200, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(select_774, 0, 2, -2);  select_774 = None
        slice_265: "bf16[200, 201, 26][15912, 78, 3]cuda:0" = torch.ops.aten.slice.Tensor(slice_264, 1, 1, -2);  slice_264 = None
        mul_120: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_40, slice_265);  unsqueeze_40 = slice_265 = None
        mul_121: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 1.0);  mul_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:83 in _adv_superbee, code: dx = (cost * dx)[None, 1:-2, None]
        mul_119: "bf16[204][1]cuda:0" = torch.ops.aten.mul.Tensor(arg10_1, arg17_1)
        unsqueeze_37: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_119, 0);  mul_119 = None
        slice_262: "bf16[1, 201][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_37, 1, 1, -2);  unsqueeze_37 = None
        unsqueeze_38: "bf16[1, 201, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_262, 2);  slice_262 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:85 in _adv_superbee, code: uCFL = torch.abs(velfac * vel[2:-2, 1:-2, :] * dt_tracer / dx)
        div_66: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_121, unsqueeze_38);  mul_121 = unsqueeze_38 = None
        abs_4: "bf16[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.abs.default(div_66);  div_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:96 in _adv_superbee, code: * ((1.0 - cr) + uCFL * cr)
        mul_130: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(abs_4, maximum_5);  abs_4 = maximum_5 = None
        add_65: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(sub_38, mul_130);  sub_38 = mul_130 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:95 in _adv_superbee, code: - torch.abs(velfac * vel[2:-2, 1:-2, :])
        mul_131: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(abs_6, add_65);  abs_6 = add_65 = None
        mul_132: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, mul_123);  mul_131 = mul_123 = None
        mul_133: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, 0.5);  mul_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:91 in _adv_superbee, code: velfac
        sub_39: "f32[200, 201, 26][5226, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_128, mul_133);  mul_128 = mul_133 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:147 in adv_flux_superbee_wgrid, code: adv_fn[2:-2, 1:-2, :] = _adv_superbee(
        copy_97: "bf16[200, 201, 26][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_331, sub_39);  slice_331 = sub_39 = None
        slice_scatter_20: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_332, copy_97, 1, 1, -2);  slice_332 = copy_97 = None
        slice_scatter_21: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(copy_96, slice_scatter_20, 0, 2, -2);  copy_96 = slice_scatter_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:316 in integrate_tke, code: - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
        slice_515: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_21, 0, 2, -2)
        slice_516: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_515, 1, 2, -2);  slice_515 = None
        slice_517: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_21, 0, 2, -2);  slice_scatter_21 = None
        slice_518: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_517, 1, 1, -3);  slice_517 = None
        sub_46: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_516, slice_518);  slice_516 = slice_518 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:317 in integrate_tke, code: / (cost[None, 2:-2, None] * dyt[None, 2:-2, None])
        unsqueeze_47: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg10_1, 0);  arg10_1 = None
        slice_519: "bf16[1, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_47, 1, 2, -2);  unsqueeze_47 = None
        unsqueeze_48: "bf16[1, 200, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_519, 2);  slice_519 = None
        unsqueeze_49: "bf16[1, 204][204, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg17_1, 0);  arg17_1 = None
        slice_520: "bf16[1, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_49, 1, 2, -2);  unsqueeze_49 = None
        unsqueeze_50: "bf16[1, 200, 1][204, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_520, 2);  slice_520 = None
        mul_147: "bf16[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(unsqueeze_48, unsqueeze_50);  unsqueeze_48 = unsqueeze_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:316 in integrate_tke, code: - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
        div_71: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.div.Tensor(sub_46, mul_147);  sub_46 = mul_147 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:314 in integrate_tke, code: -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        sub_47: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(div_70, div_71);  div_70 = div_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:313 in integrate_tke, code: dtke[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (
        mul_148: "bf16[200, 200, 26][5200, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_496, sub_47);  slice_496 = sub_47 = None
        copy_104: "bf16[200, 200, 26][15912, 78, 3]cuda:0" = torch.ops.aten.copy.default(select_809, mul_148);  select_809 = mul_148 = None
        select_scatter_137: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_524, copy_104, 3, 0);  slice_524 = copy_104 = None
        slice_scatter_29: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_523, select_scatter_137, 1, 2, -2);  slice_523 = select_scatter_137 = None
        slice_scatter_30: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(arg1_1, slice_scatter_29, 0, 2, -2);  arg1_1 = slice_scatter_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:319 in integrate_tke, code: dtke[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0]
        select_819: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_30, 2, 0)
        select_817: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_30, 2, 0)
        select_818: "bf16[204, 204][15912, 78]cuda:0" = torch.ops.aten.select.int(select_817, 2, 0);  select_817 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:195 in integrate_tke, code: flux_top = torch.zeros_like(maskU)
        full_default: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:153 in adv_flux_superbee_wgrid, code: adv_ft[...] = 0.0
        full_default_18: "bf16[][]cpu" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_99: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(full_default, full_default_18);  full_default = full_default_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:154 in adv_flux_superbee_wgrid, code: adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(
        slice_490: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(copy_99, 0, 2, -2)
        slice_491: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_490, 1, 2, -2)
        slice_487: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(copy_99, 0, 2, -2)
        slice_488: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_487, 1, 2, -2);  slice_487 = None
        slice_489: "bf16[200, 200, 25][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_488, 2, 0, -1);  slice_488 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:54 in pad_z_edges, code: out = torch.zeros(arr_shape, dtype=arr.dtype, device=arr.device)
        full_12: "bf16[204, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 28], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:55 in pad_z_edges, code: out[:, :, 1:-1] = arr
        slice_339: "bf16[204, 204, 26][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_12, 2, 1, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:303 in integrate_tke, code: w[..., tau],
        select_775: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(arg20_1, 3, 0);  arg20_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:55 in pad_z_edges, code: out[:, :, 1:-1] = arr
        copy_100: "bf16[204, 204, 26][5712, 28, 1]cuda:0" = torch.ops.aten.copy.default(slice_339, select_775);  slice_339 = select_775 = None
        slice_scatter_23: "bf16[204, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_12, copy_100, 2, 1, -1);  full_12 = copy_100 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:109 in _adv_superbee, code: vel[2:-2, 2:-2, 1:-2]
        slice_469: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_470: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_469, 1, 2, -2);  slice_469 = None
        slice_471: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_470, 2, 1, -2);  slice_470 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:54 in pad_z_edges, code: out = torch.zeros(arr_shape, dtype=arr.dtype, device=arr.device)
        full_13: "bf16[204, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 28], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:55 in pad_z_edges, code: out[:, :, 1:-1] = arr
        slice_341: "bf16[204, 204, 26][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_13, 2, 1, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:300 in integrate_tke, code: tke[:, :, :, tau],
        select_776: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:55 in pad_z_edges, code: out[:, :, 1:-1] = arr
        copy_101: "bf16[204, 204, 26][5712, 28, 1]cuda:0" = torch.ops.aten.copy.default(slice_341, select_776);  slice_341 = select_776 = None
        slice_scatter_24: "bf16[204, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_13, copy_101, 2, 1, -1);  full_13 = copy_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:110 in _adv_superbee, code: * (var[2:-2, 2:-2, 2:-1] + var[2:-2, 2:-2, 1:-2])
        slice_463: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2)
        slice_464: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_463, 1, 2, -2);  slice_463 = None
        slice_465: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_464, 2, 2, -1);  slice_464 = None
        slice_466: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2)
        slice_467: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_466, 1, 2, -2);  slice_466 = None
        slice_468: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_467, 2, 1, -2);  slice_467 = None
        add_66: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_465, slice_468);  slice_465 = slice_468 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:109 in _adv_superbee, code: vel[2:-2, 2:-2, 1:-2]
        mul_140: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_471, add_66);  slice_471 = add_66 = None
        mul_141: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, 0.5);  mul_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:112 in _adv_superbee, code: - torch.abs(vel[2:-2, 2:-2, 1:-2]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
        slice_478: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_479: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_478, 1, 2, -2);  slice_478 = None
        slice_480: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_479, 2, 1, -2);  slice_479 = None
        abs_9: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.abs.default(slice_480);  slice_480 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:61 in limiter, code: torch.tensor([0.0], device=cr.device),
        full_default_20: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        full_default_21: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:107 in _adv_superbee, code: cr = limiter(_calc_cr(rjp, rj, rjm, vel[2:-2, 2:-2, 1:-2]))
        slice_442: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_443: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_442, 1, 2, -2);  slice_442 = None
        slice_444: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_443, 2, 1, -2);  slice_443 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        gt_2: "b8[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.gt.Scalar(slice_444, 0.0);  slice_444 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:106 in _adv_superbee, code: rjm = (var[2:-2, 2:-2, 1:-2] - var[2:-2, 2:-2, :-3]) * mask[2:-2, 2:-2, :-3]
        slice_421: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2)
        slice_422: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_421, 1, 2, -2);  slice_421 = None
        slice_423: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_422, 2, 1, -2);  slice_422 = None
        slice_424: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2)
        slice_425: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_424, 1, 2, -2);  slice_424 = None
        slice_426: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_425, 2, 0, -3);  slice_425 = None
        sub_42: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_423, slice_426);  slice_423 = slice_426 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:54 in pad_z_edges, code: out = torch.zeros(arr_shape, dtype=arr.dtype, device=arr.device)
        full_14: "bf16[204, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 28], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:55 in pad_z_edges, code: out[:, :, 1:-1] = arr
        slice_343: "bf16[204, 204, 26][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_14, 2, 1, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:151 in adv_flux_superbee_wgrid, code: maskWtr = torch.zeros_like(maskW)
        full_11: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:152 in adv_flux_superbee_wgrid, code: maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]
        slice_337: "bf16[204, 204, 25][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_11, 2, 0, -1)
        slice_335: "bf16[204, 204, 25][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 2, 1, 9223372036854775807)
        slice_336: "bf16[204, 204, 25][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg15_1, 2, 0, -1);  arg15_1 = None
        mul_134: "bf16[204, 204, 25][5100, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_335, slice_336);  slice_335 = slice_336 = None
        copy_98: "bf16[204, 204, 25][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_337, mul_134);  slice_337 = mul_134 = None
        slice_scatter_22: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_11, copy_98, 2, 0, -1);  full_11 = copy_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:55 in pad_z_edges, code: out[:, :, 1:-1] = arr
        copy_102: "bf16[204, 204, 26][5712, 28, 1]cuda:0" = torch.ops.aten.copy.default(slice_343, slice_scatter_22);  slice_343 = slice_scatter_22 = None
        slice_scatter_25: "bf16[204, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_14, copy_102, 2, 1, -1);  full_14 = copy_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:106 in _adv_superbee, code: rjm = (var[2:-2, 2:-2, 1:-2] - var[2:-2, 2:-2, :-3]) * mask[2:-2, 2:-2, :-3]
        slice_433: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 2, -2)
        slice_434: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_433, 1, 2, -2);  slice_433 = None
        slice_435: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_434, 2, 0, -3);  slice_434 = None
        mul_138: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, slice_435);  sub_42 = slice_435 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:104 in _adv_superbee, code: rjp = (var[2:-2, 2:-2, 3:] - var[2:-2, 2:-2, 2:-1]) * mask[2:-2, 2:-2, 2:-1]
        slice_367: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2)
        slice_368: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_367, 1, 2, -2);  slice_367 = None
        slice_369: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_368, 2, 3, 9223372036854775807);  slice_368 = None
        slice_370: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2)
        slice_371: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_370, 1, 2, -2);  slice_370 = None
        slice_372: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_371, 2, 2, -1);  slice_371 = None
        sub_40: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_369, slice_372);  slice_369 = slice_372 = None
        slice_379: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 2, -2)
        slice_380: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_379, 1, 2, -2);  slice_379 = None
        slice_381: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_380, 2, 2, -1);  slice_380 = None
        mul_136: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, slice_381);  sub_40 = slice_381 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        where_8: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.where.self(gt_2, mul_138, mul_136);  gt_2 = mul_138 = mul_136 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:105 in _adv_superbee, code: rj = (var[2:-2, 2:-2, 2:-1] - var[2:-2, 2:-2, 1:-2]) * mask[2:-2, 2:-2, 1:-2]
        slice_394: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2)
        slice_395: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_394, 1, 2, -2);  slice_394 = None
        slice_396: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_395, 2, 2, -1);  slice_395 = None
        slice_397: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 2, -2);  slice_scatter_24 = None
        slice_398: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_397, 1, 2, -2);  slice_397 = None
        slice_399: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_398, 2, 1, -2);  slice_398 = None
        sub_41: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_396, slice_399);  slice_396 = slice_399 = None
        slice_406: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 2, -2);  slice_scatter_25 = None
        slice_407: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_406, 1, 2, -2);  slice_406 = None
        slice_408: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_407, 2, 1, -2);  slice_407 = None
        mul_137: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, slice_408);  sub_41 = slice_408 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:48 in _calc_cr, code: return torch.where(vel > 0.0, rjm, rjp) / torch.where(torch.abs(rj) < eps, eps, rj)
        abs_8: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.abs.default(mul_137)
        lt_3: "b8[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.lt.Scalar(abs_8, 1e-20);  abs_8 = None
        full_default_19: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 1.0005576689441423e-20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.where.self(lt_3, full_default_19, mul_137);  lt_3 = full_default_19 = None
        div_69: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.div.Tensor(where_8, where_9);  where_8 = where_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:63 in limiter, code: torch.minimum(torch.tensor([1.0], device=cr.device), 2 * cr),
        mul_139: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_69, 2)
        minimum_4: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.minimum.default(full_default_21, mul_139);  full_default_21 = mul_139 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:64 in limiter, code: torch.minimum(torch.tensor([2.0], device=cr.device), cr),
        full_default_22: "f32[1][1]cuda:0" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_5: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.minimum.default(full_default_22, div_69);  full_default_22 = div_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:62 in limiter, code: torch.maximum(
        maximum_6: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.maximum.default(minimum_4, minimum_5);  minimum_4 = minimum_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:60 in limiter, code: return torch.maximum(
        maximum_7: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.maximum.default(full_default_20, maximum_6);  full_default_20 = maximum_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:112 in _adv_superbee, code: - torch.abs(vel[2:-2, 2:-2, 1:-2]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
        sub_43: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.sub.Tensor(1.0, maximum_7)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:103 in _adv_superbee, code: uCFL = torch.abs(vel[2:-2, 2:-2, 1:-2] * dt_tracer / dx)
        slice_352: "bf16[200, 204, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2);  slice_scatter_23 = None
        slice_353: "bf16[200, 200, 28][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_352, 1, 2, -2);  slice_352 = None
        slice_354: "bf16[200, 200, 25][5712, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_353, 2, 1, -2);  slice_353 = None
        mul_135: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_354, 1.0);  slice_354 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:102 in _adv_superbee, code: dx = dx[None, None, :-1]
        unsqueeze_41: "bf16[1, 26][26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_42: "bf16[1, 1, 26][26, 26, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_41, 1);  unsqueeze_41 = None
        slice_345: "bf16[1, 1, 25][26, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze_42, 2, 0, -1);  unsqueeze_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:103 in _adv_superbee, code: uCFL = torch.abs(vel[2:-2, 2:-2, 1:-2] * dt_tracer / dx)
        div_68: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_135, slice_345);  mul_135 = slice_345 = None
        abs_7: "bf16[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.abs.default(div_68);  div_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:112 in _adv_superbee, code: - torch.abs(vel[2:-2, 2:-2, 1:-2]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
        mul_142: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(abs_7, maximum_7);  abs_7 = maximum_7 = None
        add_67: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.add.Tensor(sub_43, mul_142);  sub_43 = mul_142 = None
        mul_143: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(abs_9, add_67);  abs_9 = add_67 = None
        mul_144: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, mul_137);  mul_143 = mul_137 = None
        mul_145: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, 0.5);  mul_144 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:109 in _adv_superbee, code: vel[2:-2, 2:-2, 1:-2]
        sub_44: "f32[200, 200, 25][5000, 25, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_141, mul_145);  mul_141 = mul_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:154 in adv_flux_superbee_wgrid, code: adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(
        copy_103: "bf16[200, 200, 25][5304, 26, 1]cuda:0" = torch.ops.aten.copy.default(slice_489, sub_44);  slice_489 = sub_44 = None
        slice_scatter_26: "bf16[200, 200, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_491, copy_103, 2, 0, -1);  slice_491 = copy_103 = None
        slice_scatter_27: "bf16[200, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_490, slice_scatter_26, 1, 2, -2);  slice_490 = slice_scatter_26 = None
        slice_scatter_28: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.slice_scatter.default(copy_99, slice_scatter_27, 0, 2, -2);  copy_99 = slice_scatter_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:319 in integrate_tke, code: dtke[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0]
        select_815: "bf16[204, 204][5304, 26]cuda:0" = torch.ops.aten.select.int(slice_scatter_28, 2, 0)
        neg_55: "bf16[204, 204][204, 1]cuda:0" = torch.ops.aten.neg.default(select_815);  select_815 = None
        select_816: "bf16[][]cuda:0" = torch.ops.aten.select.int(arg6_1, 0, 0)
        div_72: "bf16[204, 204][204, 1]cuda:0" = torch.ops.aten.div.Tensor(neg_55, select_816);  neg_55 = select_816 = None
        add_68: "bf16[204, 204][204, 1]cuda:0" = torch.ops.aten.add.Tensor(select_818, div_72);  select_818 = div_72 = None
        select_scatter_138: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_819, add_68, 2, 0);  select_819 = add_68 = None
        select_scatter_139: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_30, select_scatter_138, 2, 0);  slice_scatter_30 = select_scatter_138 = None
        select_825: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(select_scatter_139, 2, 0)
        select_826: "bf16[204, 204][15912, 78]cuda:0" = torch.ops.aten.select.int(select_825, 2, 0);  select_825 = select_826 = None
        select_827: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(select_scatter_139, 2, 0)
        select_820: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(select_scatter_139, 2, 0)
        select_821: "bf16[204, 204][15912, 78]cuda:0" = torch.ops.aten.select.int(select_820, 2, 0);  select_820 = None
        select_scatter_140: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_827, select_821, 2, 0);  select_827 = select_821 = None
        select_scatter_141: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_139, select_scatter_140, 2, 0);  select_scatter_139 = select_scatter_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:320 in integrate_tke, code: dtke[:, :, 1:-1, tau] += -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1]
        slice_535: "bf16[204, 204, 24, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_141, 2, 1, -1)
        slice_534: "bf16[204, 204, 24, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_141, 2, 1, -1)
        select_831: "bf16[204, 204, 24][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_534, 3, 0);  slice_534 = None
        slice_531: "bf16[204, 204, 24][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_28, 2, 1, -1)
        slice_532: "bf16[204, 204, 24][5304, 26, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_28, 2, 0, -2)
        sub_48: "bf16[204, 204, 24][4896, 24, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_531, slice_532);  slice_531 = slice_532 = None
        neg_56: "bf16[204, 204, 24][4896, 24, 1]cuda:0" = torch.ops.aten.neg.default(sub_48);  sub_48 = None
        slice_533: "bf16[24][1]cuda:0" = torch.ops.aten.slice.Tensor(arg6_1, 0, 1, -1)
        div_73: "bf16[204, 204, 24][4896, 24, 1]cuda:0" = torch.ops.aten.div.Tensor(neg_56, slice_533);  neg_56 = slice_533 = None
        add_69: "bf16[204, 204, 24][4896, 24, 1]cuda:0" = torch.ops.aten.add.Tensor(select_831, div_73);  select_831 = div_73 = None
        select_scatter_142: "bf16[204, 204, 24, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_535, add_69, 3, 0);  slice_535 = add_69 = None
        slice_scatter_31: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_141, select_scatter_142, 2, 1, -1);  select_scatter_141 = select_scatter_142 = None
        slice_539: "bf16[204, 204, 24, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_31, 2, 1, -1)
        select_834: "bf16[204, 204, 24][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_539, 3, 0);  slice_539 = select_834 = None
        slice_540: "bf16[204, 204, 24, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_31, 2, 1, -1)
        slice_536: "bf16[204, 204, 24, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_31, 2, 1, -1)
        select_832: "bf16[204, 204, 24][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_536, 3, 0);  slice_536 = None
        select_scatter_143: "bf16[204, 204, 24, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_540, select_832, 3, 0);  slice_540 = select_832 = None
        slice_scatter_32: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_scatter_31, select_scatter_143, 2, 1, -1);  slice_scatter_31 = select_scatter_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:321 in integrate_tke, code: dtke[:, :, -1, tau] += -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1])
        select_846: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_32, 2, -1)
        select_844: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_32, 2, -1)
        select_845: "bf16[204, 204][15912, 78]cuda:0" = torch.ops.aten.select.int(select_844, 2, 0);  select_844 = None
        select_841: "bf16[204, 204][5304, 26]cuda:0" = torch.ops.aten.select.int(slice_scatter_28, 2, -1)
        select_842: "bf16[204, 204][5304, 26]cuda:0" = torch.ops.aten.select.int(slice_scatter_28, 2, -2);  slice_scatter_28 = None
        sub_49: "bf16[204, 204][204, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_841, select_842);  select_841 = select_842 = None
        neg_57: "bf16[204, 204][204, 1]cuda:0" = torch.ops.aten.neg.default(sub_49);  sub_49 = None
        select_843: "bf16[][]cuda:0" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_149: "bf16[][]cuda:0" = torch.ops.aten.mul.Tensor(select_843, 0.5);  select_843 = None
        div_74: "bf16[204, 204][204, 1]cuda:0" = torch.ops.aten.div.Tensor(neg_57, mul_149);  neg_57 = mul_149 = None
        add_70: "bf16[204, 204][204, 1]cuda:0" = torch.ops.aten.add.Tensor(select_845, div_74);  select_845 = div_74 = None
        select_scatter_144: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_846, add_70, 2, 0);  select_846 = add_70 = None
        select_scatter_145: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_32, select_scatter_144, 2, -1);  slice_scatter_32 = select_scatter_144 = None
        select_852: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(select_scatter_145, 2, -1)
        select_853: "bf16[204, 204][15912, 78]cuda:0" = torch.ops.aten.select.int(select_852, 2, 0);  select_852 = select_853 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:326 in integrate_tke, code: tke[:, :, :, taup1] += dt_tracer * (
        select_862: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 3, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:321 in integrate_tke, code: dtke[:, :, -1, tau] += -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1])
        select_854: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(select_scatter_145, 2, -1)
        select_847: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(select_scatter_145, 2, -1)
        select_848: "bf16[204, 204][15912, 78]cuda:0" = torch.ops.aten.select.int(select_847, 2, 0);  select_847 = None
        select_scatter_146: "bf16[204, 204, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_854, select_848, 2, 0);  select_854 = select_848 = None
        select_scatter_147: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_145, select_scatter_146, 2, -1);  select_scatter_145 = select_scatter_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:327 in integrate_tke, code: (1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1]
        select_859: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(select_scatter_147, 3, 0)
        mul_150: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_859, 1.6);  select_859 = None
        select_861: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(select_scatter_147, 3, 2)
        mul_151: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_861, 0.6);  select_861 = None
        sub_50: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:326 in integrate_tke, code: tke[:, :, :, taup1] += dt_tracer * (
        mul_152: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, 1.0);  sub_50 = None
        add_71: "bf16[204, 204, 26][5304, 26, 1]cuda:0" = torch.ops.aten.add.Tensor(select_862, mul_152);  select_862 = mul_152 = None
        select_scatter_148: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_15, add_71, 3, 1);  slice_scatter_15 = add_71 = None
        select_865: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(select_scatter_148, 3, 1);  select_865 = None
        select_863: "bf16[204, 204, 26][15912, 78, 3]cuda:0" = torch.ops.aten.select.int(select_scatter_148, 3, 1)
        select_scatter_149: "bf16[204, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.select_scatter.default(select_scatter_148, select_863, 3, 1);  select_scatter_148 = select_863 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:255 in integrate_tke, code: tke_surf_corr = torch.zeros(maskU.shape[:2], device=maskU.device)
        full_8: "f32[204, 204][204, 1]cuda:0" = torch.ops.aten.full.default([204, 204], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:257 in integrate_tke, code: tke_surf_corr[2:-2, 2:-2] = torch.where(
        slice_87: "f32[200, 204][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_8, 0, 2, -2)
        slice_85: "f32[200, 204][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_8, 0, 2, -2)
        slice_86: "f32[200, 200][204, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_85, 1, 2, -2);  slice_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:256 in integrate_tke, code: mask = tke[2:-2, 2:-2, -1, taup1] < 0.0
        slice_74: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_75: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_74, 1, 2, -2);  slice_74 = None
        select_731: "bf16[200, 200, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_75, 2, -1);  slice_75 = None
        select_732: "bf16[200, 200][15912, 78]cuda:0" = torch.ops.aten.select.int(select_731, 2, 1);  select_731 = None
        lt: "b8[200, 200][200, 1]cuda:0" = torch.ops.aten.lt.Scalar(select_732, 0.0);  select_732 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:258 in integrate_tke, code: mask, -tke[2:-2, 2:-2, -1, taup1] * 0.5 * dzw[-1] / dt_tke, 0.0
        slice_83: "bf16[200, 204, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2);  slice_scatter_5 = None
        slice_84: "bf16[200, 200, 26, 3][15912, 78, 3, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_83, 1, 2, -2);  slice_83 = None
        select_736: "bf16[200, 200, 3][15912, 78, 1]cuda:0" = torch.ops.aten.select.int(slice_84, 2, -1);  slice_84 = None
        select_737: "bf16[200, 200][15912, 78]cuda:0" = torch.ops.aten.select.int(select_736, 2, 1);  select_736 = None
        neg_53: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.neg.default(select_737);  select_737 = None
        mul_93: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_53, 0.5);  neg_53 = None
        select_738: "bf16[][]cuda:0" = torch.ops.aten.select.int(arg6_1, 0, -1);  arg6_1 = None
        mul_94: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, select_738);  mul_93 = select_738 = None
        div_59: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_94, 1);  mul_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_turbulent_kinetic_energy/tke_pytorch.py:257 in integrate_tke, code: tke_surf_corr[2:-2, 2:-2] = torch.where(
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[200, 200][200, 1]cuda:0" = torch.ops.aten.where.self(lt, div_59, full_default_4);  lt = div_59 = full_default_4 = None
        copy_85: "f32[200, 200][204, 1]cuda:0" = torch.ops.aten.copy.default(slice_86, where_3);  slice_86 = where_3 = None
        slice_scatter_6: "f32[200, 204][204, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_87, copy_85, 1, 2, -2);  slice_87 = copy_85 = None
        slice_scatter_7: "f32[204, 204][204, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_6, 0, 2, -2);  full_8 = slice_scatter_6 = None
        return (select_scatter_149, select_scatter_147, slice_scatter_7)

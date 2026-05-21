class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f64[204, 204, 26]", arg1_1: "f64[204, 204, 26, 3]", arg2_1: "f64[204, 204, 26, 3]", arg3_1: "f64[26]", arg4_1: "f64[204, 204, 26]", arg5_1: "f64[204, 204, 26]", arg6_1: "f64[26]", arg7_1: "f64[204, 204, 26]", arg8_1: "f64[204]", arg9_1: "f64[204]", arg10_1: "f64[204, 204, 26]", arg11_1: "f64[204]", arg12_1: "f64[204, 204, 26]", arg13_1: "f64[204, 204, 26, 2, 2]", arg14_1: "f64[26]", arg15_1: "f64[204, 204, 26, 2, 2]", arg16_1: "f64[204, 204, 26]", arg17_1: "f64[204, 204, 26, 2, 2]", arg18_1: "f64[204]", arg19_1: "f64[204, 204, 26, 2, 2]", arg20_1: "f64[204]", arg21_1: "f64[204]", arg22_1: "f64[204, 204, 26]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:21 in get_drhodS, code: return betaS * rho0 * torch.ones_like(temp)
        full_6: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 1, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:69 in isoneutral_diffusion_pre, code: dTdx = torch.zeros_like(K_11)
        full: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:99 in isoneutral_diffusion_pre, code: dTdx[:-1, :, :] = (
        slice_17: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(full, 0, 0, -1);  slice_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:70 in isoneutral_diffusion_pre, code: dSdx = torch.zeros_like(K_11)
        full_1: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:104 in isoneutral_diffusion_pre, code: dSdx[:-1, :, :] = (
        slice_23: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, -1);  slice_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        _tensor_constant0: "f32[1]" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant1: "f32[1]" = self._tensor_constant1;  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:136 in isoneutral_diffusion_pre, code: sumz = torch.zeros_like(K_11)[1:-2, 2:-2]
        full_8: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_136: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_8, 0, 1, -2)
        slice_137: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_136, 1, 2, -2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:136 in isoneutral_diffusion_pre, code: sumz = torch.zeros_like(K_11)[1:-2, 2:-2]
        slice_72: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_8, 0, 1, -2)
        slice_73: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_72, 1, 2, -2);  slice_72 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_122: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_73, 2, 1, 9223372036854775807);  slice_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_16: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_17: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        slice_123: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_17, 2, 0, 25);  unsqueeze_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:158 in isoneutral_diffusion_pre, code: * maskU[1:-2, 2:-2, ki:]
        slice_124: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_125: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_124, 1, 2, -2);  slice_124 = None
        slice_126: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_125, 2, 1, 9223372036854775807);  slice_125 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_23: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_123, slice_126);  slice_123 = slice_126 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_2: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:127 in isoneutral_diffusion_pre, code: diffloc = torch.zeros_like(K_11)
        full_7: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:128 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, 1:] = 0.25 * (
        slice_52: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_7, 0, 1, -2)
        slice_53: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_52, 1, 2, -2)
        slice_49: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_7, 0, 1, -2)
        slice_50: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_49, 1, 2, -2);  slice_49 = None
        slice_51: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_50, 2, 1, 9223372036854775807);  slice_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:129 in isoneutral_diffusion_pre, code: K_iso[1:-2, 2:-2, 1:]
        slice_37: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 1, -2)
        slice_38: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_37, 1, 2, -2);  slice_37 = None
        slice_39: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_38, 2, 1, 9223372036854775807);  slice_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:130 in isoneutral_diffusion_pre, code: + K_iso[1:-2, 2:-2, :-1]
        slice_40: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 1, -2)
        slice_41: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_40, 1, 2, -2);  slice_40 = None
        slice_42: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_41, 2, 0, -1);  slice_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:129 in isoneutral_diffusion_pre, code: K_iso[1:-2, 2:-2, 1:]
        add_1: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_39, slice_42);  slice_39 = slice_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:131 in isoneutral_diffusion_pre, code: + K_iso[2:-1, 2:-2, 1:]
        slice_43: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -1)
        slice_44: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_43, 1, 2, -2);  slice_43 = None
        slice_45: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_44, 2, 1, 9223372036854775807);  slice_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:129 in isoneutral_diffusion_pre, code: K_iso[1:-2, 2:-2, 1:]
        add_2: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(add_1, slice_45);  add_1 = slice_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:132 in isoneutral_diffusion_pre, code: + K_iso[2:-1, 2:-2, :-1]
        slice_46: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -1)
        slice_47: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_46, 1, 2, -2);  slice_46 = None
        slice_48: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_47, 2, 0, -1);  slice_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:129 in isoneutral_diffusion_pre, code: K_iso[1:-2, 2:-2, 1:]
        add_3: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(add_2, slice_48);  add_2 = slice_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:128 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, 1:] = 0.25 * (
        mul_16: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_3, 0.25);  add_3 = None
        copy_6: "f64[201, 200, 25]" = torch.ops.aten.copy.default(slice_51, mul_16);  slice_51 = mul_16 = None
        slice_scatter_6: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_53, copy_6, 2, 1, 9223372036854775807);  slice_53 = copy_6 = None
        slice_scatter_7: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_52, slice_scatter_6, 1, 2, -2);  slice_52 = slice_scatter_6 = None
        slice_scatter_8: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_7, slice_scatter_7, 0, 1, -2);  full_7 = slice_scatter_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:134 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, 0] = 0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])
        slice_68: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 1, -2)
        slice_69: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_68, 1, 2, -2)
        slice_66: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 1, -2)
        slice_67: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_66, 1, 2, -2);  slice_66 = None
        select_17: "f64[201, 200]" = torch.ops.aten.select.int(slice_67, 2, 0);  slice_67 = None
        slice_57: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 1, -2)
        slice_58: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_57, 1, 2, -2);  slice_57 = None
        select_14: "f64[201, 200]" = torch.ops.aten.select.int(slice_58, 2, 0);  slice_58 = None
        slice_59: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -1)
        slice_60: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_59, 1, 2, -2);  slice_59 = None
        select_15: "f64[201, 200]" = torch.ops.aten.select.int(slice_60, 2, 0);  slice_60 = None
        add_4: "f64[201, 200]" = torch.ops.aten.add.Tensor(select_14, select_15);  select_14 = select_15 = None
        mul_17: "f64[201, 200]" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        copy_7: "f64[201, 200]" = torch.ops.aten.copy.default(select_17, mul_17);  select_17 = mul_17 = None
        select_scatter: "f64[201, 200, 26]" = torch.ops.aten.select_scatter.default(slice_69, copy_7, 2, 0);  slice_69 = copy_7 = None
        slice_scatter_9: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_68, select_scatter, 1, 2, -2);  slice_68 = select_scatter = None
        slice_scatter_10: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_8, slice_scatter_9, 0, 1, -2);  slice_scatter_8 = slice_scatter_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        slice_133: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 0, 1, -2)
        slice_134: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_133, 1, 2, -2);  slice_133 = None
        slice_135: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_134, 2, 1, 9223372036854775807);  slice_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:79 in isoneutral_diffusion_pre, code: drdT = maskT * get_drhodT(salt[:, :, :, tau], temp[:, :, :, tau], torch.abs(zt))
        select: "f64[204, 204, 26]" = torch.ops.aten.select.int(arg2_1, 3, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:14 in get_drhodT, code: thetas = temp - theta0
        sub_1: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(select, 9.850000000000023);  select = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:15 in get_drhodT, code: return -(betaTs * thetas + betaT * (1 - gammas * grav * zz * rho0)) * rho0
        mul: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_1, 1e-05);  sub_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:79 in isoneutral_diffusion_pre, code: drdT = maskT * get_drhodT(salt[:, :, :, tau], temp[:, :, :, tau], torch.abs(zt))
        abs_1: "f64[26]" = torch.ops.aten.abs.default(arg3_1);  arg3_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:13 in get_drhodT, code: zz = -p - z0
        neg: "f64[26]" = torch.ops.aten.neg.default(abs_1);  abs_1 = None
        sub: "f64[26]" = torch.ops.aten.sub.Tensor(neg, 0.0);  neg = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:15 in get_drhodT, code: return -(betaTs * thetas + betaT * (1 - gammas * grav * zz * rho0)) * rho0
        mul_1: "f64[26]" = torch.ops.aten.mul.Tensor(sub, 1.0790999999999999e-07);  sub = None
        mul_2: "f64[26]" = torch.ops.aten.mul.Tensor(mul_1, 1024.0);  mul_1 = None
        sub_2: "f64[26]" = torch.ops.aten.sub.Tensor(1, mul_2);  mul_2 = None
        mul_3: "f64[26]" = torch.ops.aten.mul.Tensor(sub_2, 0.000167);  sub_2 = None
        add: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul, mul_3);  mul = mul_3 = None
        neg_1: "f64[204, 204, 26]" = torch.ops.aten.neg.default(add);  add = None
        mul_4: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(neg_1, 1024.0);  neg_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:79 in isoneutral_diffusion_pre, code: drdT = maskT * get_drhodT(salt[:, :, :, tau], temp[:, :, :, tau], torch.abs(zt))
        mul_5: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg4_1, mul_4);  mul_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        slice_74: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 1, -2)
        slice_75: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_74, 1, 2, -2);  slice_74 = None
        slice_76: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_75, 2, 1, 9223372036854775807);  slice_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:100 in isoneutral_diffusion_pre, code: maskU[:-1, :, :]
        slice_13: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:101 in isoneutral_diffusion_pre, code: * (temp[1:, :, :, tau] - temp[:-1, :, :, tau])
        slice_14: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 1, 9223372036854775807)
        select_6: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_14, 3, 0);  slice_14 = None
        slice_15: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, -1)
        select_7: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_15, 3, 0);  slice_15 = None
        sub_5: "f64[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_6, select_7);  select_6 = select_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:100 in isoneutral_diffusion_pre, code: maskU[:-1, :, :]
        mul_10: "f64[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_13, sub_5);  slice_13 = sub_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:102 in isoneutral_diffusion_pre, code: / (dxu[:-1, None, None] * cost[None, :, None])
        slice_16: "f64[203]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, -1)
        unsqueeze_4: "f64[203, 1]" = torch.ops.aten.unsqueeze.default(slice_16, 1);  slice_16 = None
        unsqueeze_5: "f64[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        unsqueeze_7: "f64[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        mul_11: "f64[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_5, unsqueeze_7);  unsqueeze_5 = unsqueeze_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:100 in isoneutral_diffusion_pre, code: maskU[:-1, :, :]
        div_2: "f64[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:99 in isoneutral_diffusion_pre, code: dTdx[:-1, :, :] = (
        slice_scatter_2: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full, div_2, 0, 0, -1);  full = div_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        slice_83: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -2)
        slice_84: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_83, 1, 2, -2);  slice_83 = None
        slice_85: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_84, 2, 1, 9223372036854775807);  slice_84 = None
        mul_18: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_76, slice_85);  slice_76 = slice_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:21 in get_drhodS, code: return betaS * rho0 * torch.ones_like(temp)
        full_default: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0.79872, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:80 in isoneutral_diffusion_pre, code: drdS = maskT * get_drhodS(salt[:, :, :, tau], temp[:, :, :, tau], torch.abs(zt))
        mul_7: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg4_1, full_default);  arg4_1 = full_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:146 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
        slice_86: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 1, -2)
        slice_87: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_86, 1, 2, -2);  slice_86 = None
        slice_88: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_87, 2, 1, 9223372036854775807);  slice_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:105 in isoneutral_diffusion_pre, code: maskU[:-1, :, :]
        slice_19: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:106 in isoneutral_diffusion_pre, code: * (salt[1:, :, :, tau] - salt[:-1, :, :, tau])
        slice_20: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, 9223372036854775807)
        select_8: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_20, 3, 0);  slice_20 = None
        slice_21: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, -1)
        select_9: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_21, 3, 0);  slice_21 = None
        sub_6: "f64[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_8, select_9);  select_8 = select_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:105 in isoneutral_diffusion_pre, code: maskU[:-1, :, :]
        mul_12: "f64[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_19, sub_6);  slice_19 = sub_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:107 in isoneutral_diffusion_pre, code: / (dxu[:-1, None, None] * cost[None, :, None])
        slice_22: "f64[203]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, -1)
        unsqueeze_8: "f64[203, 1]" = torch.ops.aten.unsqueeze.default(slice_22, 1);  slice_22 = None
        unsqueeze_9: "f64[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        unsqueeze_11: "f64[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        mul_13: "f64[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_9, unsqueeze_11);  unsqueeze_9 = unsqueeze_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:105 in isoneutral_diffusion_pre, code: maskU[:-1, :, :]
        div_3: "f64[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:104 in isoneutral_diffusion_pre, code: dSdx[:-1, :, :] = (
        slice_scatter_3: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_1, div_3, 0, 0, -1);  full_1 = div_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:146 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
        slice_95: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 1, -2)
        slice_96: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_95, 1, 2, -2);  slice_95 = None
        slice_97: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_96, 2, 1, 9223372036854775807);  slice_96 = None
        mul_19: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_88, slice_97);  slice_88 = slice_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        add_5: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        neg_2: "f64[201, 200, 25]" = torch.ops.aten.neg.default(add_5);  add_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_98: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 1, -2)
        slice_99: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_98, 1, 2, -2);  slice_98 = None
        slice_100: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_99, 2, 1, 9223372036854775807);  slice_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:73 in isoneutral_diffusion_pre, code: dTdz = torch.zeros_like(K_11)
        full_4: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:85 in isoneutral_diffusion_pre, code: dTdz[:, :, :-1] = (
        slice_5: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_4, 2, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:86 in isoneutral_diffusion_pre, code: maskW[:, :, :-1]
        slice_1: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg5_1, 2, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:87 in isoneutral_diffusion_pre, code: * (temp[:, :, 1:, tau] - temp[:, :, :-1, tau])
        slice_2: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 2, 1, 9223372036854775807)
        select_2: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_2, 3, 0);  slice_2 = None
        slice_3: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 2, 0, -1)
        select_3: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_3, 3, 0);  slice_3 = None
        sub_3: "f64[204, 204, 25]" = torch.ops.aten.sub.Tensor(select_2, select_3);  select_2 = select_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:86 in isoneutral_diffusion_pre, code: maskW[:, :, :-1]
        mul_8: "f64[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_1, sub_3);  slice_1 = sub_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:88 in isoneutral_diffusion_pre, code: / dzw[None, None, :-1]
        unsqueeze: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_1: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        slice_4: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 2, 0, -1);  unsqueeze_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:86 in isoneutral_diffusion_pre, code: maskW[:, :, :-1]
        div: "f64[204, 204, 25]" = torch.ops.aten.div.Tensor(mul_8, slice_4);  mul_8 = slice_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:85 in isoneutral_diffusion_pre, code: dTdz[:, :, :-1] = (
        copy: "f64[204, 204, 25]" = torch.ops.aten.copy.default(slice_5, div);  slice_5 = div = None
        slice_scatter: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_4, copy, 2, 0, -1);  full_4 = copy = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_107: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 1, -2)
        slice_108: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_107, 1, 2, -2);  slice_107 = None
        slice_109: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_108, 2, 0, 25);  slice_108 = None
        mul_20: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_100, slice_109);  slice_100 = slice_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:150 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_110: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 1, -2)
        slice_111: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_110, 1, 2, -2);  slice_110 = None
        slice_112: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_111, 2, 1, 9223372036854775807);  slice_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:74 in isoneutral_diffusion_pre, code: dSdz = torch.zeros_like(K_11)
        full_5: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:90 in isoneutral_diffusion_pre, code: dSdz[:, :, :-1] = (
        slice_11: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_5, 2, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:91 in isoneutral_diffusion_pre, code: maskW[:, :, :-1]
        slice_7: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg5_1, 2, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:92 in isoneutral_diffusion_pre, code: * (salt[:, :, 1:, tau] - salt[:, :, :-1, tau])
        slice_8: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 2, 1, 9223372036854775807)
        select_4: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_8, 3, 0);  slice_8 = None
        slice_9: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 2, 0, -1)
        select_5: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_9, 3, 0);  slice_9 = None
        sub_4: "f64[204, 204, 25]" = torch.ops.aten.sub.Tensor(select_4, select_5);  select_4 = select_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:91 in isoneutral_diffusion_pre, code: maskW[:, :, :-1]
        mul_9: "f64[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_7, sub_4);  slice_7 = sub_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:93 in isoneutral_diffusion_pre, code: / dzw[None, None, :-1]
        unsqueeze_2: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_3: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        slice_10: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, -1);  unsqueeze_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:91 in isoneutral_diffusion_pre, code: maskW[:, :, :-1]
        div_1: "f64[204, 204, 25]" = torch.ops.aten.div.Tensor(mul_9, slice_10);  mul_9 = slice_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:90 in isoneutral_diffusion_pre, code: dSdz[:, :, :-1] = (
        copy_1: "f64[204, 204, 25]" = torch.ops.aten.copy.default(slice_11, div_1);  slice_11 = div_1 = None
        slice_scatter_1: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_5, copy_1, 2, 0, -1);  full_5 = copy_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:150 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_119: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 1, -2)
        slice_120: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_119, 1, 2, -2);  slice_119 = None
        slice_121: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_120, 2, 0, 25);  slice_120 = None
        mul_21: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_112, slice_121);  slice_112 = slice_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        add_6: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        full_default_1: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum: "f64[201, 200, 25]" = torch.ops.aten.minimum.default(add_6, full_default_1);  add_6 = full_default_1 = None
        sub_9: "f64[201, 200, 25]" = torch.ops.aten.sub.Tensor(minimum, 1e-20);  minimum = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        div_6: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(neg_2, sub_9);  neg_2 = sub_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_3: "f64[201, 200, 25]" = torch.ops.aten.abs.default(div_6)
        neg_3: "f64[201, 200, 25]" = torch.ops.aten.neg.default(abs_3);  abs_3 = None
        add_7: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(neg_3, 0.001);  neg_3 = None
        div_7: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(add_7, 0.001);  add_7 = None
        tanh: "f64[201, 200, 25]" = torch.ops.aten.tanh.default(div_7);  div_7 = None
        add_8: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_22: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        mul_24: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_135, mul_22);  slice_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:159 in isoneutral_diffusion_pre, code: * torch.max(
        maximum: "f64[201, 200, 25]" = torch.ops.aten.maximum.default(full_default_2, mul_24);  full_default_2 = mul_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_25: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_23, maximum);  mul_23 = maximum = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_9: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_122, mul_25);  slice_122 = mul_25 = None
        slice_scatter_11: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_137, add_9, 2, 1, 9223372036854775807);  slice_137 = add_9 = None
        slice_scatter_12: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_136, slice_scatter_11, 1, 2, -2);  slice_136 = slice_scatter_11 = None
        slice_scatter_13: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_12, 0, 1, -2);  full_8 = slice_scatter_12 = None
        slice_144: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 1, -2)
        slice_145: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_144, 1, 2, -2);  slice_144 = None
        slice_146: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_145, 2, 1, 9223372036854775807);  slice_145 = slice_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        _tensor_constant2: "f32[1]" = self._tensor_constant2;  _tensor_constant2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant3: "f32[1]" = self._tensor_constant3;  _tensor_constant3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_147: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 1, -2)
        slice_148: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_147, 1, 2, -2)
        slice_138: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 1, -2)
        slice_139: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_138, 1, 2, -2);  slice_138 = None
        slice_140: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_139, 2, 1, 9223372036854775807);  slice_139 = None
        slice_scatter_14: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_148, slice_140, 2, 1, 9223372036854775807);  slice_148 = slice_140 = None
        slice_scatter_15: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_147, slice_scatter_14, 1, 2, -2);  slice_147 = slice_scatter_14 = None
        slice_scatter_16: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_13, slice_scatter_15, 0, 1, -2);  slice_scatter_13 = slice_scatter_15 = None
        slice_231: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_16, 0, 1, -2)
        slice_232: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_231, 1, 2, -2)
        slice_228: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_16, 0, 1, -2)
        slice_229: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_228, 1, 2, -2);  slice_228 = None
        slice_230: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_229, 2, 1, 9223372036854775807);  slice_229 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_18: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_19: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
        slice_215: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_19, 2, 0, 25);  unsqueeze_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:158 in isoneutral_diffusion_pre, code: * maskU[1:-2, 2:-2, ki:]
        slice_216: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_217: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_216, 1, 2, -2);  slice_216 = None
        slice_218: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_217, 2, 1, 9223372036854775807);  slice_217 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_33: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_215, slice_218);  slice_215 = slice_218 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_4: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        slice_225: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 0, 1, -2)
        slice_226: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_225, 1, 2, -2);  slice_225 = None
        slice_227: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_226, 2, 1, 9223372036854775807);  slice_226 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        slice_164: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -1)
        slice_165: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_164, 1, 2, -2);  slice_164 = None
        slice_166: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_165, 2, 1, 9223372036854775807);  slice_165 = None
        slice_173: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -2)
        slice_174: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_173, 1, 2, -2);  slice_173 = None
        slice_175: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_174, 2, 1, 9223372036854775807);  slice_174 = None
        mul_28: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_166, slice_175);  slice_166 = slice_175 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:146 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
        slice_176: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -1)
        slice_177: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_176, 1, 2, -2);  slice_176 = None
        slice_178: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_177, 2, 1, 9223372036854775807);  slice_177 = None
        slice_185: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 1, -2)
        slice_186: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_185, 1, 2, -2);  slice_185 = None
        slice_187: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_186, 2, 1, 9223372036854775807);  slice_186 = None
        mul_29: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_178, slice_187);  slice_178 = slice_187 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        add_10: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        neg_4: "f64[201, 200, 25]" = torch.ops.aten.neg.default(add_10);  add_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_188: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -1)
        slice_189: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_188, 1, 2, -2);  slice_188 = None
        slice_190: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_189, 2, 1, 9223372036854775807);  slice_189 = None
        slice_197: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -1)
        slice_198: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_197, 1, 2, -2);  slice_197 = None
        slice_199: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_198, 2, 0, 25);  slice_198 = None
        mul_30: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_190, slice_199);  slice_190 = slice_199 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:150 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_200: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -1)
        slice_201: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_200, 1, 2, -2);  slice_200 = None
        slice_202: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_201, 2, 1, 9223372036854775807);  slice_201 = None
        slice_209: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -1)
        slice_210: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_209, 1, 2, -2);  slice_209 = None
        slice_211: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_210, 2, 0, 25);  slice_210 = None
        mul_31: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_202, slice_211);  slice_202 = slice_211 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        add_11: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        full_default_3: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "f64[201, 200, 25]" = torch.ops.aten.minimum.default(add_11, full_default_3);  add_11 = full_default_3 = None
        sub_10: "f64[201, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_1, 1e-20);  minimum_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        div_8: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(neg_4, sub_10);  neg_4 = sub_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_4: "f64[201, 200, 25]" = torch.ops.aten.abs.default(div_8)
        neg_5: "f64[201, 200, 25]" = torch.ops.aten.neg.default(abs_4);  abs_4 = None
        add_12: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(neg_5, 0.001);  neg_5 = None
        div_9: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(add_12, 0.001);  add_12 = None
        tanh_1: "f64[201, 200, 25]" = torch.ops.aten.tanh.default(div_9);  div_9 = None
        add_13: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_32: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        mul_34: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_227, mul_32);  slice_227 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:159 in isoneutral_diffusion_pre, code: * torch.max(
        maximum_1: "f64[201, 200, 25]" = torch.ops.aten.maximum.default(full_default_4, mul_34);  full_default_4 = mul_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_35: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_33, maximum_1);  mul_33 = maximum_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_14: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_230, mul_35);  slice_230 = mul_35 = None
        slice_scatter_20: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_232, add_14, 2, 1, 9223372036854775807);  slice_232 = add_14 = None
        slice_scatter_21: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_231, slice_scatter_20, 1, 2, -2);  slice_231 = slice_scatter_20 = None
        slice_scatter_22: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_16, slice_scatter_21, 0, 1, -2);  slice_scatter_16 = slice_scatter_21 = None
        slice_239: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 1, -2)
        slice_240: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_239, 1, 2, -2);  slice_239 = None
        slice_241: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_240, 2, 1, 9223372036854775807);  slice_240 = slice_241 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        _tensor_constant4: "f32[1]" = self._tensor_constant4;  _tensor_constant4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant5: "f32[1]" = self._tensor_constant5;  _tensor_constant5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_242: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 1, -2)
        slice_243: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_242, 1, 2, -2)
        slice_233: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 1, -2)
        slice_234: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_233, 1, 2, -2);  slice_233 = None
        slice_235: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_234, 2, 1, 9223372036854775807);  slice_234 = None
        slice_scatter_23: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_243, slice_235, 2, 1, 9223372036854775807);  slice_243 = slice_235 = None
        slice_scatter_24: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_242, slice_scatter_23, 1, 2, -2);  slice_242 = slice_scatter_23 = None
        slice_scatter_25: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_22, slice_scatter_24, 0, 1, -2);  slice_scatter_22 = slice_scatter_24 = None
        slice_310: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 1, -2)
        slice_308: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 1, -2)
        slice_309: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_308, 1, 2, -2);  slice_308 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_20: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_21: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:158 in isoneutral_diffusion_pre, code: * maskU[1:-2, 2:-2, ki:]
        slice_301: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_302: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_301, 1, 2, -2);  slice_301 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_43: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_21, slice_302);  unsqueeze_21 = slice_302 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_6: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        slice_306: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 0, 1, -2)
        slice_307: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_306, 1, 2, -2);  slice_306 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        slice_271: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 1, -2)
        slice_272: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_271, 1, 2, -2);  slice_271 = None
        slice_276: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -2)
        slice_277: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_276, 1, 2, -2);  slice_276 = None
        mul_38: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_272, slice_277);  slice_272 = slice_277 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:146 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
        slice_278: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 1, -2)
        slice_279: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_278, 1, 2, -2);  slice_278 = None
        slice_283: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 1, -2)
        slice_284: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_283, 1, 2, -2);  slice_283 = None
        mul_39: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_279, slice_284);  slice_279 = slice_284 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        add_15: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        neg_6: "f64[201, 200, 26]" = torch.ops.aten.neg.default(add_15);  add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_285: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 1, -2)
        slice_286: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_285, 1, 2, -2);  slice_285 = None
        slice_290: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 1, -2)
        slice_291: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_290, 1, 2, -2);  slice_290 = None
        mul_40: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_286, slice_291);  slice_286 = slice_291 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:150 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_292: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 1, -2)
        slice_293: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_292, 1, 2, -2);  slice_292 = None
        slice_297: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 1, -2)
        slice_298: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_297, 1, 2, -2);  slice_297 = None
        mul_41: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_293, slice_298);  slice_293 = slice_298 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        add_16: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        full_default_5: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_2: "f64[201, 200, 26]" = torch.ops.aten.minimum.default(add_16, full_default_5);  add_16 = full_default_5 = None
        sub_11: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(minimum_2, 1e-20);  minimum_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        div_10: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(neg_6, sub_11);  neg_6 = sub_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_5: "f64[201, 200, 26]" = torch.ops.aten.abs.default(div_10)
        neg_7: "f64[201, 200, 26]" = torch.ops.aten.neg.default(abs_5);  abs_5 = None
        add_17: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(neg_7, 0.001);  neg_7 = None
        div_11: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(add_17, 0.001);  add_17 = None
        tanh_2: "f64[201, 200, 26]" = torch.ops.aten.tanh.default(div_11);  div_11 = None
        add_18: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_42: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        mul_44: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_307, mul_42);  slice_307 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:159 in isoneutral_diffusion_pre, code: * torch.max(
        maximum_2: "f64[201, 200, 26]" = torch.ops.aten.maximum.default(full_default_6, mul_44);  full_default_6 = mul_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_45: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_43, maximum_2);  mul_43 = maximum_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_19: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_309, mul_45);  slice_309 = mul_45 = None
        slice_scatter_29: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_310, add_19, 1, 2, -2);  slice_310 = add_19 = None
        slice_scatter_30: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_25, slice_scatter_29, 0, 1, -2);  slice_scatter_25 = slice_scatter_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:136 in isoneutral_diffusion_pre, code: sumz = torch.zeros_like(K_11)[1:-2, 2:-2]
        slice_314: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_30, 0, 1, -2)
        slice_315: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_314, 1, 2, -2);  slice_314 = slice_315 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        _tensor_constant6: "f32[1]" = self._tensor_constant6;  _tensor_constant6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant7: "f32[1]" = self._tensor_constant7;  _tensor_constant7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_316: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_30, 0, 1, -2)
        slice_312: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_30, 0, 1, -2)
        slice_313: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_312, 1, 2, -2);  slice_312 = None
        slice_scatter_31: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_316, slice_313, 1, 2, -2);  slice_316 = slice_313 = None
        slice_scatter_32: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_30, slice_scatter_31, 0, 1, -2);  slice_scatter_30 = slice_scatter_31 = None
        slice_371: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 1, -2)
        slice_369: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 1, -2)
        slice_370: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_369, 1, 2, -2);  slice_369 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_22: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_23: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:158 in isoneutral_diffusion_pre, code: * maskU[1:-2, 2:-2, ki:]
        slice_362: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_363: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_362, 1, 2, -2);  slice_362 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_53: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_23, slice_363);  unsqueeze_23 = slice_363 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:160 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_8: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        slice_367: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 0, 1, -2)
        slice_368: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_367, 1, 2, -2);  slice_367 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        slice_334: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -1)
        slice_335: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_334, 1, 2, -2);  slice_334 = None
        slice_339: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -2)
        slice_340: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_339, 1, 2, -2);  slice_339 = None
        mul_48: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_335, slice_340);  slice_335 = slice_340 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:146 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
        slice_341: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -1)
        slice_342: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_341, 1, 2, -2);  slice_341 = None
        slice_346: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 1, -2)
        slice_347: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_346, 1, 2, -2);  slice_346 = None
        mul_49: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_342, slice_347);  slice_342 = slice_347 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:145 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
        add_20: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        neg_8: "f64[201, 200, 26]" = torch.ops.aten.neg.default(add_20);  add_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_348: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -1)
        slice_349: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_348, 1, 2, -2);  slice_348 = None
        slice_353: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -1)
        slice_354: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_353, 1, 2, -2);  slice_353 = None
        mul_50: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_349, slice_354);  slice_349 = slice_354 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:150 in isoneutral_diffusion_pre, code: + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdz[1 + ip : -2 + ip, 2:-2, :su]
        slice_355: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -1)
        slice_356: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_355, 1, 2, -2);  slice_355 = None
        slice_360: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -1)
        slice_361: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_360, 1, 2, -2);  slice_360 = None
        mul_51: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_356, slice_361);  slice_356 = slice_361 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:149 in isoneutral_diffusion_pre, code: drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdz[1 + ip : -2 + ip, 2:-2, :su]
        add_21: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:153 in isoneutral_diffusion_pre, code: torch.min(drodze, torch.tensor([0.0], device=device)) - epsln
        full_default_7: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_3: "f64[201, 200, 26]" = torch.ops.aten.minimum.default(add_21, full_default_7);  add_21 = full_default_7 = None
        sub_12: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(minimum_3, 1e-20);  minimum_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:152 in isoneutral_diffusion_pre, code: sxe = -drodxe / (
        div_12: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(neg_8, sub_12);  neg_8 = sub_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_6: "f64[201, 200, 26]" = torch.ops.aten.abs.default(div_12)
        neg_9: "f64[201, 200, 26]" = torch.ops.aten.neg.default(abs_6);  abs_6 = None
        add_22: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(neg_9, 0.001);  neg_9 = None
        div_13: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(add_22, 0.001);  add_22 = None
        tanh_3: "f64[201, 200, 26]" = torch.ops.aten.tanh.default(div_13);  div_13 = None
        add_23: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_52: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(add_23, 0.5);  add_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:161 in isoneutral_diffusion_pre, code: diffloc[1:-2, 2:-2, ki:] * taper,
        mul_54: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_368, mul_52);  slice_368 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:159 in isoneutral_diffusion_pre, code: * torch.max(
        maximum_3: "f64[201, 200, 26]" = torch.ops.aten.maximum.default(full_default_8, mul_54);  full_default_8 = mul_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:157 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_55: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_53, maximum_3);  mul_53 = maximum_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_24: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_370, mul_55);  slice_370 = mul_55 = None
        slice_scatter_35: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_371, add_24, 1, 2, -2);  slice_371 = add_24 = None
        slice_scatter_36: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_32, slice_scatter_35, 0, 1, -2);  slice_scatter_32 = slice_scatter_35 = None
        slice_375: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_36, 0, 1, -2)
        slice_376: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_375, 1, 2, -2);  slice_375 = slice_376 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:164 in isoneutral_diffusion_pre, code: Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * sxe * maskU[1:-2, 2:-2, ki:]
        slice_158: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 1, -2)
        slice_159: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_158, 1, 2, -2)
        slice_160: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_159, 2, 1, 9223372036854775807)
        select_21: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_160, 3, 0)
        slice_155: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 1, -2)
        slice_156: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_155, 1, 2, -2);  slice_155 = None
        slice_157: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_156, 2, 1, 9223372036854775807);  slice_156 = None
        select_19: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_157, 3, 0);  slice_157 = None
        select_20: "f64[201, 200, 25]" = torch.ops.aten.select.int(select_19, 3, 0);  select_19 = None
        mul_26: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_22, div_6);  mul_22 = div_6 = None
        slice_152: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_153: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_152, 1, 2, -2);  slice_152 = None
        slice_154: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_153, 2, 1, 9223372036854775807);  slice_153 = None
        mul_27: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_26, slice_154);  mul_26 = slice_154 = None
        copy_9: "f64[201, 200, 25]" = torch.ops.aten.copy.default(select_20, mul_27);  select_20 = mul_27 = None
        select_scatter_1: "f64[201, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_21, copy_9, 3, 0);  select_21 = copy_9 = None
        select_scatter_2: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_160, select_scatter_1, 3, 0);  slice_160 = select_scatter_1 = None
        slice_scatter_17: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_159, select_scatter_2, 2, 1, 9223372036854775807);  slice_159 = select_scatter_2 = None
        slice_scatter_18: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_158, slice_scatter_17, 1, 2, -2);  slice_158 = slice_scatter_17 = None
        slice_scatter_19: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg13_1, slice_scatter_18, 0, 1, -2);  slice_scatter_18 = None
        slice_265: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 1, -2)
        slice_266: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_265, 1, 2, -2)
        slice_267: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_266, 2, 1, 9223372036854775807)
        select_29: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_267, 3, 1)
        slice_262: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 1, -2)
        slice_263: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_262, 1, 2, -2);  slice_262 = None
        slice_264: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_263, 2, 1, 9223372036854775807);  slice_263 = None
        select_27: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_264, 3, 1);  slice_264 = None
        select_28: "f64[201, 200, 25]" = torch.ops.aten.select.int(select_27, 3, 0);  select_27 = None
        mul_36: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_32, div_8);  mul_32 = div_8 = None
        slice_247: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_248: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_247, 1, 2, -2);  slice_247 = None
        slice_249: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_248, 2, 1, 9223372036854775807);  slice_248 = None
        mul_37: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_36, slice_249);  mul_36 = slice_249 = None
        copy_11: "f64[201, 200, 25]" = torch.ops.aten.copy.default(select_28, mul_37);  select_28 = mul_37 = None
        select_scatter_3: "f64[201, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_29, copy_11, 3, 0);  select_29 = copy_11 = None
        select_scatter_4: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_267, select_scatter_3, 3, 1);  slice_267 = select_scatter_3 = None
        slice_scatter_26: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_266, select_scatter_4, 2, 1, 9223372036854775807);  slice_266 = select_scatter_4 = None
        slice_scatter_27: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_265, slice_scatter_26, 1, 2, -2);  slice_265 = slice_scatter_26 = None
        slice_scatter_28: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_19, slice_scatter_27, 0, 1, -2);  slice_scatter_19 = slice_scatter_27 = None
        slice_330: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 0, 1, -2)
        slice_331: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_330, 1, 2, -2)
        select_37: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_331, 3, 0)
        slice_328: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 0, 1, -2)
        slice_329: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_328, 1, 2, -2);  slice_328 = None
        select_35: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_329, 3, 0);  slice_329 = None
        select_36: "f64[201, 200, 26]" = torch.ops.aten.select.int(select_35, 3, 1);  select_35 = None
        mul_46: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_42, div_10);  mul_42 = div_10 = None
        slice_319: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_320: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_319, 1, 2, -2);  slice_319 = None
        mul_47: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_46, slice_320);  mul_46 = slice_320 = None
        copy_13: "f64[201, 200, 26]" = torch.ops.aten.copy.default(select_36, mul_47);  select_36 = mul_47 = None
        select_scatter_5: "f64[201, 200, 26, 2]" = torch.ops.aten.select_scatter.default(select_37, copy_13, 3, 1);  select_37 = copy_13 = None
        select_scatter_6: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_331, select_scatter_5, 3, 0);  slice_331 = select_scatter_5 = None
        slice_scatter_33: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_330, select_scatter_6, 1, 2, -2);  slice_330 = select_scatter_6 = None
        slice_scatter_34: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_28, slice_scatter_33, 0, 1, -2);  slice_scatter_28 = slice_scatter_33 = None
        slice_391: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_34, 0, 1, -2)
        slice_392: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_391, 1, 2, -2)
        select_45: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_392, 3, 1)
        slice_389: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_34, 0, 1, -2)
        slice_390: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_389, 1, 2, -2);  slice_389 = None
        select_43: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_390, 3, 1);  slice_390 = None
        select_44: "f64[201, 200, 26]" = torch.ops.aten.select.int(select_43, 3, 1);  select_43 = None
        mul_56: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_52, div_12);  mul_52 = div_12 = None
        slice_380: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2);  arg7_1 = None
        slice_381: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_380, 1, 2, -2);  slice_380 = None
        mul_57: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_56, slice_381);  mul_56 = slice_381 = None
        copy_15: "f64[201, 200, 26]" = torch.ops.aten.copy.default(select_44, mul_57);  select_44 = mul_57 = None
        select_scatter_7: "f64[201, 200, 26, 2]" = torch.ops.aten.select_scatter.default(select_45, copy_15, 3, 1);  select_45 = copy_15 = None
        select_scatter_8: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_392, select_scatter_7, 3, 1);  slice_392 = select_scatter_7 = None
        slice_scatter_39: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_391, select_scatter_8, 1, 2, -2);  slice_391 = select_scatter_8 = None
        slice_scatter_40: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_34, slice_scatter_39, 0, 1, -2);  slice_scatter_34 = slice_scatter_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:165 in isoneutral_diffusion_pre, code: K_11[1:-2, 2:-2, :] = sumz / (4.0 * dzt[None, None, :])
        slice_397: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 1, -2)
        slice_395: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 1, -2)
        slice_396: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_395, 1, 2, -2);  slice_395 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:156 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_377: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_36, 0, 1, -2)
        slice_373: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_36, 0, 1, -2)
        slice_374: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_373, 1, 2, -2);  slice_373 = None
        slice_scatter_37: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_377, slice_374, 1, 2, -2);  slice_377 = slice_374 = None
        slice_scatter_38: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_36, slice_scatter_37, 0, 1, -2);  slice_scatter_36 = slice_scatter_37 = None
        slice_378: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 0, 1, -2);  slice_scatter_38 = None
        slice_379: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_378, 1, 2, -2);  slice_378 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:165 in isoneutral_diffusion_pre, code: K_11[1:-2, 2:-2, :] = sumz / (4.0 * dzt[None, None, :])
        unsqueeze_24: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        unsqueeze_25: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 1);  unsqueeze_24 = None
        mul_58: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_25, 4.0);  unsqueeze_25 = None
        div_14: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(slice_379, mul_58);  slice_379 = mul_58 = None
        copy_16: "f64[201, 200, 26]" = torch.ops.aten.copy.default(slice_396, div_14);  slice_396 = div_14 = None
        slice_scatter_41: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_397, copy_16, 1, 2, -2);  slice_397 = copy_16 = None
        slice_scatter_42: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg0_1, slice_scatter_41, 0, 1, -2);  slice_scatter_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:170 in isoneutral_diffusion_pre, code: diffloc[...] = 0
        _tensor_constant8: "f64[]" = self._tensor_constant8;  _tensor_constant8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        _tensor_constant9: "f32[1]" = self._tensor_constant9;  _tensor_constant9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant10: "f32[1]" = self._tensor_constant10;  _tensor_constant10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:179 in isoneutral_diffusion_pre, code: sumz = torch.zeros_like(K_11)[2:-2, 1:-2]
        full_9: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_505: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_9, 0, 2, -2)
        slice_506: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_505, 1, 1, -2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:179 in isoneutral_diffusion_pre, code: sumz = torch.zeros_like(K_11)[2:-2, 1:-2]
        slice_441: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_9, 0, 2, -2)
        slice_442: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_441, 1, 1, -2);  slice_441 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_491: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_442, 2, 1, 9223372036854775807);  slice_442 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_26: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_27: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 1);  unsqueeze_26 = None
        slice_492: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_27, 2, 0, 25);  unsqueeze_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:201 in isoneutral_diffusion_pre, code: * maskV[2:-2, 1:-2, ki:]
        slice_493: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_494: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_493, 1, 1, -2);  slice_493 = None
        slice_495: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_494, 2, 1, 9223372036854775807);  slice_494 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_66: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_492, slice_495);  slice_492 = slice_495 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_11: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:170 in isoneutral_diffusion_pre, code: diffloc[...] = 0
        full_default_9: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_17: "f64[204, 204, 26]" = torch.ops.aten.copy.default(slice_scatter_10, full_default_9);  slice_scatter_10 = full_default_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:171 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, 1:] = 0.25 * (
        slice_421: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_17, 0, 2, -2)
        slice_422: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_421, 1, 1, -2)
        slice_418: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_17, 0, 2, -2)
        slice_419: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_418, 1, 1, -2);  slice_418 = None
        slice_420: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_419, 2, 1, 9223372036854775807);  slice_419 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:172 in isoneutral_diffusion_pre, code: K_iso[2:-2, 1:-2, 1:]
        slice_400: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_401: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_400, 1, 1, -2);  slice_400 = None
        slice_402: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_401, 2, 1, 9223372036854775807);  slice_401 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:173 in isoneutral_diffusion_pre, code: + K_iso[2:-2, 1:-2, :-1]
        slice_403: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_404: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_403, 1, 1, -2);  slice_403 = None
        slice_405: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_404, 2, 0, -1);  slice_404 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:172 in isoneutral_diffusion_pre, code: K_iso[2:-2, 1:-2, 1:]
        add_25: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_402, slice_405);  slice_402 = slice_405 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:174 in isoneutral_diffusion_pre, code: + K_iso[2:-2, 2:-1, 1:]
        slice_406: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_407: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_406, 1, 2, -1);  slice_406 = None
        slice_408: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_407, 2, 1, 9223372036854775807);  slice_407 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:172 in isoneutral_diffusion_pre, code: K_iso[2:-2, 1:-2, 1:]
        add_26: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(add_25, slice_408);  add_25 = slice_408 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:175 in isoneutral_diffusion_pre, code: + K_iso[2:-2, 2:-1, :-1]
        slice_409: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_410: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_409, 1, 2, -1);  slice_409 = None
        slice_411: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_410, 2, 0, -1);  slice_410 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:172 in isoneutral_diffusion_pre, code: K_iso[2:-2, 1:-2, 1:]
        add_27: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(add_26, slice_411);  add_26 = slice_411 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:171 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, 1:] = 0.25 * (
        mul_59: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_27, 0.25);  add_27 = None
        copy_18: "f64[200, 201, 25]" = torch.ops.aten.copy.default(slice_420, mul_59);  slice_420 = mul_59 = None
        slice_scatter_43: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_422, copy_18, 2, 1, 9223372036854775807);  slice_422 = copy_18 = None
        slice_scatter_44: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_421, slice_scatter_43, 1, 1, -2);  slice_421 = slice_scatter_43 = None
        slice_scatter_45: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_17, slice_scatter_44, 0, 2, -2);  copy_17 = slice_scatter_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:177 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, 0] = 0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])
        slice_437: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_45, 0, 2, -2)
        slice_438: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_437, 1, 1, -2)
        slice_435: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_45, 0, 2, -2)
        slice_436: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_435, 1, 1, -2);  slice_435 = None
        select_51: "f64[200, 201]" = torch.ops.aten.select.int(slice_436, 2, 0);  slice_436 = None
        slice_426: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_427: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_426, 1, 1, -2);  slice_426 = None
        select_48: "f64[200, 201]" = torch.ops.aten.select.int(slice_427, 2, 0);  slice_427 = None
        slice_428: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_429: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_428, 1, 2, -1);  slice_428 = None
        select_49: "f64[200, 201]" = torch.ops.aten.select.int(slice_429, 2, 0);  slice_429 = None
        add_28: "f64[200, 201]" = torch.ops.aten.add.Tensor(select_48, select_49);  select_48 = select_49 = None
        mul_60: "f64[200, 201]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
        copy_19: "f64[200, 201]" = torch.ops.aten.copy.default(select_51, mul_60);  select_51 = mul_60 = None
        select_scatter_9: "f64[200, 201, 26]" = torch.ops.aten.select_scatter.default(slice_438, copy_19, 2, 0);  slice_438 = copy_19 = None
        slice_scatter_46: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_437, select_scatter_9, 1, 1, -2);  slice_437 = select_scatter_9 = None
        slice_scatter_47: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_45, slice_scatter_46, 0, 2, -2);  slice_scatter_45 = slice_scatter_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        slice_502: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_503: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_502, 1, 1, -2);  slice_502 = None
        slice_504: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_503, 2, 1, 9223372036854775807);  slice_503 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        slice_443: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_444: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_443, 1, 1, -2);  slice_443 = None
        slice_445: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_444, 2, 1, 9223372036854775807);  slice_444 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:71 in isoneutral_diffusion_pre, code: dTdy = torch.zeros_like(K_11)
        full_2: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:113 in isoneutral_diffusion_pre, code: dTdy[:, :-1, :] = (
        slice_29: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_2, 1, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:114 in isoneutral_diffusion_pre, code: maskV[:, :-1, :]
        slice_25: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 1, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:115 in isoneutral_diffusion_pre, code: * (temp[:, 1:, :, tau] - temp[:, :-1, :, tau])
        slice_26: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 1, 9223372036854775807)
        select_10: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_26, 3, 0);  slice_26 = None
        slice_27: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, -1);  arg2_1 = None
        select_11: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_27, 3, 0);  slice_27 = None
        sub_7: "f64[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_10, select_11);  select_10 = select_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:114 in isoneutral_diffusion_pre, code: maskV[:, :-1, :]
        mul_14: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_25, sub_7);  slice_25 = sub_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:116 in isoneutral_diffusion_pre, code: / dyu[None, :-1, None]
        unsqueeze_12: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg11_1, 0)
        slice_28: "f64[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_12, 1, 0, -1);  unsqueeze_12 = None
        unsqueeze_13: "f64[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_28, 2);  slice_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:114 in isoneutral_diffusion_pre, code: maskV[:, :-1, :]
        div_4: "f64[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_14, unsqueeze_13);  mul_14 = unsqueeze_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:113 in isoneutral_diffusion_pre, code: dTdy[:, :-1, :] = (
        copy_4: "f64[204, 203, 26]" = torch.ops.aten.copy.default(slice_29, div_4);  slice_29 = div_4 = None
        slice_scatter_4: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_2, copy_4, 1, 0, -1);  full_2 = copy_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        slice_452: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2)
        slice_453: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_452, 1, 1, -2);  slice_452 = None
        slice_454: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_453, 2, 1, 9223372036854775807);  slice_453 = None
        mul_61: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_445, slice_454);  slice_445 = slice_454 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:189 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
        slice_455: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_456: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_455, 1, 1, -2);  slice_455 = None
        slice_457: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_456, 2, 1, 9223372036854775807);  slice_456 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:72 in isoneutral_diffusion_pre, code: dSdy = torch.zeros_like(K_11)
        full_3: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:118 in isoneutral_diffusion_pre, code: dSdy[:, :-1, :] = (
        slice_35: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_3, 1, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:119 in isoneutral_diffusion_pre, code: maskV[:, :-1, :]
        slice_31: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 1, 0, -1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:120 in isoneutral_diffusion_pre, code: * (salt[:, 1:, :, tau] - salt[:, :-1, :, tau])
        slice_32: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 1, 9223372036854775807)
        select_12: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_32, 3, 0);  slice_32 = None
        slice_33: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, -1);  arg1_1 = None
        select_13: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_33, 3, 0);  slice_33 = None
        sub_8: "f64[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_12, select_13);  select_12 = select_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:119 in isoneutral_diffusion_pre, code: maskV[:, :-1, :]
        mul_15: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_31, sub_8);  slice_31 = sub_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:121 in isoneutral_diffusion_pre, code: / dyu[None, :-1, None]
        unsqueeze_14: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg11_1, 0)
        slice_34: "f64[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_14, 1, 0, -1);  unsqueeze_14 = None
        unsqueeze_15: "f64[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_34, 2);  slice_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:119 in isoneutral_diffusion_pre, code: maskV[:, :-1, :]
        div_5: "f64[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_15, unsqueeze_15);  mul_15 = unsqueeze_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:118 in isoneutral_diffusion_pre, code: dSdy[:, :-1, :] = (
        copy_5: "f64[204, 203, 26]" = torch.ops.aten.copy.default(slice_35, div_5);  slice_35 = div_5 = None
        slice_scatter_5: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_3, copy_5, 1, 0, -1);  full_3 = copy_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:189 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
        slice_464: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_465: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_464, 1, 1, -2);  slice_464 = None
        slice_466: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_465, 2, 1, 9223372036854775807);  slice_465 = None
        mul_62: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_457, slice_466);  slice_457 = slice_466 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        add_29: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        neg_10: "f64[200, 201, 25]" = torch.ops.aten.neg.default(add_29);  add_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        full_default_10: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_467: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_468: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_467, 1, 1, -2);  slice_467 = None
        slice_469: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_468, 2, 1, 9223372036854775807);  slice_468 = None
        slice_476: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -2)
        slice_477: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_476, 1, 1, -2);  slice_476 = None
        slice_478: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_477, 2, 0, 25);  slice_477 = None
        mul_63: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_469, slice_478);  slice_469 = slice_478 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:193 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_479: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_480: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_479, 1, 1, -2);  slice_479 = None
        slice_481: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_480, 2, 1, 9223372036854775807);  slice_480 = None
        slice_488: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -2)
        slice_489: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_488, 1, 1, -2);  slice_488 = None
        slice_490: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_489, 2, 0, 25);  slice_489 = None
        mul_64: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_481, slice_490);  slice_481 = slice_490 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        add_30: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        minimum_4: "f64[200, 201, 25]" = torch.ops.aten.minimum.default(full_default_10, add_30);  full_default_10 = add_30 = None
        sub_13: "f64[200, 201, 25]" = torch.ops.aten.sub.Tensor(minimum_4, 1e-20);  minimum_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        div_15: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(neg_10, sub_13);  neg_10 = sub_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_7: "f64[200, 201, 25]" = torch.ops.aten.abs.default(div_15)
        neg_11: "f64[200, 201, 25]" = torch.ops.aten.neg.default(abs_7);  abs_7 = None
        add_31: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(neg_11, 0.001);  neg_11 = None
        div_16: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(add_31, 0.001);  add_31 = None
        tanh_4: "f64[200, 201, 25]" = torch.ops.aten.tanh.default(div_16);  div_16 = None
        add_32: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_65: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        mul_67: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_504, mul_65);  slice_504 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:202 in isoneutral_diffusion_pre, code: * torch.max(
        maximum_4: "f64[200, 201, 25]" = torch.ops.aten.maximum.default(full_default_11, mul_67);  full_default_11 = mul_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_68: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_66, maximum_4);  mul_66 = maximum_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_33: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_491, mul_68);  slice_491 = mul_68 = None
        slice_scatter_48: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_506, add_33, 2, 1, 9223372036854775807);  slice_506 = add_33 = None
        slice_scatter_49: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_505, slice_scatter_48, 1, 1, -2);  slice_505 = slice_scatter_48 = None
        slice_scatter_50: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_49, 0, 2, -2);  full_9 = slice_scatter_49 = None
        slice_513: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 2, -2)
        slice_514: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_513, 1, 1, -2);  slice_513 = None
        slice_515: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_514, 2, 1, 9223372036854775807);  slice_514 = slice_515 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        _tensor_constant11: "f32[1]" = self._tensor_constant11;  _tensor_constant11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant12: "f32[1]" = self._tensor_constant12;  _tensor_constant12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_516: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 2, -2)
        slice_517: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_516, 1, 1, -2)
        slice_507: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 2, -2)
        slice_508: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_507, 1, 1, -2);  slice_507 = None
        slice_509: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_508, 2, 1, 9223372036854775807);  slice_508 = None
        slice_scatter_51: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_517, slice_509, 2, 1, 9223372036854775807);  slice_517 = slice_509 = None
        slice_scatter_52: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_516, slice_scatter_51, 1, 1, -2);  slice_516 = slice_scatter_51 = None
        slice_scatter_53: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_50, slice_scatter_52, 0, 2, -2);  slice_scatter_50 = slice_scatter_52 = None
        slice_600: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_53, 0, 2, -2)
        slice_601: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_600, 1, 1, -2)
        slice_597: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_53, 0, 2, -2)
        slice_598: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_597, 1, 1, -2);  slice_597 = None
        slice_599: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_598, 2, 1, 9223372036854775807);  slice_598 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_28: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_29: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 1);  unsqueeze_28 = None
        slice_584: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_29, 2, 0, 25);  unsqueeze_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:201 in isoneutral_diffusion_pre, code: * maskV[2:-2, 1:-2, ki:]
        slice_585: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_586: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_585, 1, 1, -2);  slice_585 = None
        slice_587: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_586, 2, 1, 9223372036854775807);  slice_586 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_76: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_584, slice_587);  slice_584 = slice_587 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_13: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        slice_594: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_595: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_594, 1, 1, -2);  slice_594 = None
        slice_596: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_595, 2, 1, 9223372036854775807);  slice_595 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        slice_533: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_534: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_533, 1, 2, -1);  slice_533 = None
        slice_535: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_534, 2, 1, 9223372036854775807);  slice_534 = None
        slice_542: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2)
        slice_543: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_542, 1, 1, -2);  slice_542 = None
        slice_544: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_543, 2, 1, 9223372036854775807);  slice_543 = None
        mul_71: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_535, slice_544);  slice_535 = slice_544 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:189 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
        slice_545: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_546: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_545, 1, 2, -1);  slice_545 = None
        slice_547: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_546, 2, 1, 9223372036854775807);  slice_546 = None
        slice_554: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_555: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_554, 1, 1, -2);  slice_554 = None
        slice_556: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_555, 2, 1, 9223372036854775807);  slice_555 = None
        mul_72: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_547, slice_556);  slice_547 = slice_556 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        add_34: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        neg_12: "f64[200, 201, 25]" = torch.ops.aten.neg.default(add_34);  add_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        full_default_12: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_557: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_558: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_557, 1, 2, -1);  slice_557 = None
        slice_559: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_558, 2, 1, 9223372036854775807);  slice_558 = None
        slice_566: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -2)
        slice_567: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_566, 1, 2, -1);  slice_566 = None
        slice_568: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_567, 2, 0, 25);  slice_567 = None
        mul_73: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_559, slice_568);  slice_559 = slice_568 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:193 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_569: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_570: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_569, 1, 2, -1);  slice_569 = None
        slice_571: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_570, 2, 1, 9223372036854775807);  slice_570 = None
        slice_578: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -2)
        slice_579: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_578, 1, 2, -1);  slice_578 = None
        slice_580: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_579, 2, 0, 25);  slice_579 = None
        mul_74: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_571, slice_580);  slice_571 = slice_580 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        add_35: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        minimum_5: "f64[200, 201, 25]" = torch.ops.aten.minimum.default(full_default_12, add_35);  full_default_12 = add_35 = None
        sub_14: "f64[200, 201, 25]" = torch.ops.aten.sub.Tensor(minimum_5, 1e-20);  minimum_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        div_17: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(neg_12, sub_14);  neg_12 = sub_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_8: "f64[200, 201, 25]" = torch.ops.aten.abs.default(div_17)
        neg_13: "f64[200, 201, 25]" = torch.ops.aten.neg.default(abs_8);  abs_8 = None
        add_36: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(neg_13, 0.001);  neg_13 = None
        div_18: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(add_36, 0.001);  add_36 = None
        tanh_5: "f64[200, 201, 25]" = torch.ops.aten.tanh.default(div_18);  div_18 = None
        add_37: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_75: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_37, 0.5);  add_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        mul_77: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_596, mul_75);  slice_596 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:202 in isoneutral_diffusion_pre, code: * torch.max(
        maximum_5: "f64[200, 201, 25]" = torch.ops.aten.maximum.default(full_default_13, mul_77);  full_default_13 = mul_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_78: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_76, maximum_5);  mul_76 = maximum_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_38: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_599, mul_78);  slice_599 = mul_78 = None
        slice_scatter_57: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_601, add_38, 2, 1, 9223372036854775807);  slice_601 = add_38 = None
        slice_scatter_58: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_600, slice_scatter_57, 1, 1, -2);  slice_600 = slice_scatter_57 = None
        slice_scatter_59: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_53, slice_scatter_58, 0, 2, -2);  slice_scatter_53 = slice_scatter_58 = None
        slice_608: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 2, -2)
        slice_609: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_608, 1, 1, -2);  slice_608 = None
        slice_610: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_609, 2, 1, 9223372036854775807);  slice_609 = slice_610 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        _tensor_constant13: "f32[1]" = self._tensor_constant13;  _tensor_constant13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant14: "f32[1]" = self._tensor_constant14;  _tensor_constant14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_611: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 2, -2)
        slice_612: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_611, 1, 1, -2)
        slice_602: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 2, -2)
        slice_603: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_602, 1, 1, -2);  slice_602 = None
        slice_604: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_603, 2, 1, 9223372036854775807);  slice_603 = None
        slice_scatter_60: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_612, slice_604, 2, 1, 9223372036854775807);  slice_612 = slice_604 = None
        slice_scatter_61: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_611, slice_scatter_60, 1, 1, -2);  slice_611 = slice_scatter_60 = None
        slice_scatter_62: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_59, slice_scatter_61, 0, 2, -2);  slice_scatter_59 = slice_scatter_61 = None
        slice_679: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_62, 0, 2, -2)
        slice_677: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_62, 0, 2, -2)
        slice_678: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_677, 1, 1, -2);  slice_677 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_30: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_31: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 1);  unsqueeze_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:201 in isoneutral_diffusion_pre, code: * maskV[2:-2, 1:-2, ki:]
        slice_670: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_671: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_670, 1, 1, -2);  slice_670 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_86: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_31, slice_671);  unsqueeze_31 = slice_671 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_15: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        slice_675: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_676: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_675, 1, 1, -2);  slice_675 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        slice_640: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_641: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_640, 1, 1, -2);  slice_640 = None
        slice_645: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2)
        slice_646: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_645, 1, 1, -2);  slice_645 = None
        mul_81: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_641, slice_646);  slice_641 = slice_646 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:189 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
        slice_647: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_648: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_647, 1, 1, -2);  slice_647 = None
        slice_652: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_653: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_652, 1, 1, -2);  slice_652 = None
        mul_82: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_648, slice_653);  slice_648 = slice_653 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        add_39: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        neg_14: "f64[200, 201, 26]" = torch.ops.aten.neg.default(add_39);  add_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        full_default_14: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_654: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_655: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_654, 1, 1, -2);  slice_654 = None
        slice_659: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -2)
        slice_660: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_659, 1, 1, -2);  slice_659 = None
        mul_83: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_655, slice_660);  slice_655 = slice_660 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:193 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_661: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_662: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_661, 1, 1, -2);  slice_661 = None
        slice_666: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -2)
        slice_667: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_666, 1, 1, -2);  slice_666 = None
        mul_84: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_662, slice_667);  slice_662 = slice_667 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        add_40: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        minimum_6: "f64[200, 201, 26]" = torch.ops.aten.minimum.default(full_default_14, add_40);  full_default_14 = add_40 = None
        sub_15: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(minimum_6, 1e-20);  minimum_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        div_19: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(neg_14, sub_15);  neg_14 = sub_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_9: "f64[200, 201, 26]" = torch.ops.aten.abs.default(div_19)
        neg_15: "f64[200, 201, 26]" = torch.ops.aten.neg.default(abs_9);  abs_9 = None
        add_41: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(neg_15, 0.001);  neg_15 = None
        div_20: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(add_41, 0.001);  add_41 = None
        tanh_6: "f64[200, 201, 26]" = torch.ops.aten.tanh.default(div_20);  div_20 = None
        add_42: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_85: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        mul_87: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_676, mul_85);  slice_676 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:202 in isoneutral_diffusion_pre, code: * torch.max(
        maximum_6: "f64[200, 201, 26]" = torch.ops.aten.maximum.default(full_default_15, mul_87);  full_default_15 = mul_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_88: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_86, maximum_6);  mul_86 = maximum_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_43: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_678, mul_88);  slice_678 = mul_88 = None
        slice_scatter_66: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_679, add_43, 1, 1, -2);  slice_679 = add_43 = None
        slice_scatter_67: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_62, slice_scatter_66, 0, 2, -2);  slice_scatter_62 = slice_scatter_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:179 in isoneutral_diffusion_pre, code: sumz = torch.zeros_like(K_11)[2:-2, 1:-2]
        slice_683: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_67, 0, 2, -2)
        slice_684: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_683, 1, 1, -2);  slice_683 = slice_684 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        _tensor_constant15: "f32[1]" = self._tensor_constant15;  _tensor_constant15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        _tensor_constant16: "f32[1]" = self._tensor_constant16;  _tensor_constant16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_685: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_67, 0, 2, -2)
        slice_681: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_67, 0, 2, -2)
        slice_682: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_681, 1, 1, -2);  slice_681 = None
        slice_scatter_68: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_685, slice_682, 1, 1, -2);  slice_685 = slice_682 = None
        slice_scatter_69: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_67, slice_scatter_68, 0, 2, -2);  slice_scatter_67 = slice_scatter_68 = None
        slice_740: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_69, 0, 2, -2)
        slice_738: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_69, 0, 2, -2)
        slice_739: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_738, 1, 1, -2);  slice_738 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        unsqueeze_32: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0);  arg6_1 = None
        unsqueeze_33: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 1);  unsqueeze_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:201 in isoneutral_diffusion_pre, code: * maskV[2:-2, 1:-2, ki:]
        slice_731: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_732: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_731, 1, 1, -2);  slice_731 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_96: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_33, slice_732);  unsqueeze_33 = slice_732 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:203 in isoneutral_diffusion_pre, code: torch.tensor([K_iso_steep], device=device),
        full_default_17: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        slice_736: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2);  slice_scatter_47 = None
        slice_737: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_736, 1, 1, -2);  slice_736 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        slice_703: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_704: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_703, 1, 2, -1);  slice_703 = None
        slice_708: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2)
        slice_709: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_708, 1, 1, -2);  slice_708 = None
        mul_91: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_704, slice_709);  slice_704 = slice_709 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:189 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
        slice_710: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_711: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_710, 1, 2, -1);  slice_710 = None
        slice_715: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_716: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_715, 1, 1, -2);  slice_715 = None
        mul_92: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_711, slice_716);  slice_711 = slice_716 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:188 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
        add_44: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        neg_16: "f64[200, 201, 26]" = torch.ops.aten.neg.default(add_44);  add_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        full_default_16: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_717: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_718: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_717, 1, 2, -1);  slice_717 = None
        slice_722: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -2)
        slice_723: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_722, 1, 2, -1);  slice_722 = None
        mul_93: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_718, slice_723);  slice_718 = slice_723 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:193 in isoneutral_diffusion_pre, code: + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdz[2:-2, 1 + jp : -2 + jp, :su]
        slice_724: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_725: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_724, 1, 2, -1);  slice_724 = None
        slice_729: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -2)
        slice_730: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_729, 1, 2, -1);  slice_729 = None
        mul_94: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_725, slice_730);  slice_725 = slice_730 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:192 in isoneutral_diffusion_pre, code: drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdz[2:-2, 1 + jp : -2 + jp, :su]
        add_45: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:196 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzn) - epsln
        minimum_7: "f64[200, 201, 26]" = torch.ops.aten.minimum.default(full_default_16, add_45);  full_default_16 = add_45 = None
        sub_16: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(minimum_7, 1e-20);  minimum_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:195 in isoneutral_diffusion_pre, code: syn = -drodyn / (
        div_21: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(neg_16, sub_16);  neg_16 = sub_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_10: "f64[200, 201, 26]" = torch.ops.aten.abs.default(div_21)
        neg_17: "f64[200, 201, 26]" = torch.ops.aten.neg.default(abs_10);  abs_10 = None
        add_46: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(neg_17, 0.001);  neg_17 = None
        div_22: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(add_46, 0.001);  add_46 = None
        tanh_7: "f64[200, 201, 26]" = torch.ops.aten.tanh.default(div_22);  div_22 = None
        add_47: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_95: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(add_47, 0.5);  add_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:204 in isoneutral_diffusion_pre, code: diffloc[2:-2, 1:-2, ki:] * taper,
        mul_97: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_737, mul_95);  slice_737 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:202 in isoneutral_diffusion_pre, code: * torch.max(
        maximum_7: "f64[200, 201, 26]" = torch.ops.aten.maximum.default(full_default_17, mul_97);  full_default_17 = mul_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:200 in isoneutral_diffusion_pre, code: dzw[None, None, :su]
        mul_98: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_96, maximum_7);  mul_96 = maximum_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        add_48: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_739, mul_98);  slice_739 = mul_98 = None
        slice_scatter_72: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_740, add_48, 1, 1, -2);  slice_740 = add_48 = None
        slice_scatter_73: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_69, slice_scatter_72, 0, 2, -2);  slice_scatter_69 = slice_scatter_72 = None
        slice_744: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_73, 0, 2, -2)
        slice_745: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_744, 1, 1, -2);  slice_744 = slice_745 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:207 in isoneutral_diffusion_pre, code: Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * syn * maskV[2:-2, 1:-2, ki:]
        slice_527: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_528: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_527, 1, 1, -2)
        slice_529: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_528, 2, 1, 9223372036854775807)
        select_55: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_529, 3, 0)
        slice_524: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_525: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_524, 1, 1, -2);  slice_524 = None
        slice_526: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_525, 2, 1, 9223372036854775807);  slice_525 = None
        select_53: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_526, 3, 0);  slice_526 = None
        select_54: "f64[200, 201, 25]" = torch.ops.aten.select.int(select_53, 3, 0);  select_53 = None
        mul_69: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_65, div_15);  mul_65 = div_15 = None
        slice_521: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_522: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_521, 1, 1, -2);  slice_521 = None
        slice_523: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_522, 2, 1, 9223372036854775807);  slice_522 = None
        mul_70: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_69, slice_523);  mul_69 = slice_523 = None
        copy_21: "f64[200, 201, 25]" = torch.ops.aten.copy.default(select_54, mul_70);  select_54 = mul_70 = None
        select_scatter_10: "f64[200, 201, 25, 2]" = torch.ops.aten.select_scatter.default(select_55, copy_21, 3, 0);  select_55 = copy_21 = None
        select_scatter_11: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_529, select_scatter_10, 3, 0);  slice_529 = select_scatter_10 = None
        slice_scatter_54: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_528, select_scatter_11, 2, 1, 9223372036854775807);  slice_528 = select_scatter_11 = None
        slice_scatter_55: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_527, slice_scatter_54, 1, 1, -2);  slice_527 = slice_scatter_54 = None
        slice_scatter_56: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg15_1, slice_scatter_55, 0, 2, -2);  slice_scatter_55 = None
        slice_634: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 2, -2)
        slice_635: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_634, 1, 1, -2)
        slice_636: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_635, 2, 1, 9223372036854775807)
        select_63: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_636, 3, 1)
        slice_631: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 2, -2)
        slice_632: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_631, 1, 1, -2);  slice_631 = None
        slice_633: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_632, 2, 1, 9223372036854775807);  slice_632 = None
        select_61: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_633, 3, 1);  slice_633 = None
        select_62: "f64[200, 201, 25]" = torch.ops.aten.select.int(select_61, 3, 0);  select_61 = None
        mul_79: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_75, div_17);  mul_75 = div_17 = None
        slice_616: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_617: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_616, 1, 1, -2);  slice_616 = None
        slice_618: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_617, 2, 1, 9223372036854775807);  slice_617 = None
        mul_80: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_79, slice_618);  mul_79 = slice_618 = None
        copy_23: "f64[200, 201, 25]" = torch.ops.aten.copy.default(select_62, mul_80);  select_62 = mul_80 = None
        select_scatter_12: "f64[200, 201, 25, 2]" = torch.ops.aten.select_scatter.default(select_63, copy_23, 3, 0);  select_63 = copy_23 = None
        select_scatter_13: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_636, select_scatter_12, 3, 1);  slice_636 = select_scatter_12 = None
        slice_scatter_63: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_635, select_scatter_13, 2, 1, 9223372036854775807);  slice_635 = select_scatter_13 = None
        slice_scatter_64: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_634, slice_scatter_63, 1, 1, -2);  slice_634 = slice_scatter_63 = None
        slice_scatter_65: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_56, slice_scatter_64, 0, 2, -2);  slice_scatter_56 = slice_scatter_64 = None
        slice_699: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 2, -2)
        slice_700: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_699, 1, 1, -2)
        select_71: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_700, 3, 0)
        slice_697: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 2, -2)
        slice_698: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_697, 1, 1, -2);  slice_697 = None
        select_69: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_698, 3, 0);  slice_698 = None
        select_70: "f64[200, 201, 26]" = torch.ops.aten.select.int(select_69, 3, 1);  select_69 = None
        mul_89: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_85, div_19);  mul_85 = div_19 = None
        slice_688: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_689: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_688, 1, 1, -2);  slice_688 = None
        mul_90: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_89, slice_689);  mul_89 = slice_689 = None
        copy_25: "f64[200, 201, 26]" = torch.ops.aten.copy.default(select_70, mul_90);  select_70 = mul_90 = None
        select_scatter_14: "f64[200, 201, 26, 2]" = torch.ops.aten.select_scatter.default(select_71, copy_25, 3, 1);  select_71 = copy_25 = None
        select_scatter_15: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_700, select_scatter_14, 3, 0);  slice_700 = select_scatter_14 = None
        slice_scatter_70: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_699, select_scatter_15, 1, 1, -2);  slice_699 = select_scatter_15 = None
        slice_scatter_71: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_65, slice_scatter_70, 0, 2, -2);  slice_scatter_65 = slice_scatter_70 = None
        slice_760: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_71, 0, 2, -2)
        slice_761: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_760, 1, 1, -2)
        select_79: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_761, 3, 1)
        slice_758: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_71, 0, 2, -2)
        slice_759: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_758, 1, 1, -2);  slice_758 = None
        select_77: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_759, 3, 1);  slice_759 = None
        select_78: "f64[200, 201, 26]" = torch.ops.aten.select.int(select_77, 3, 1);  select_77 = None
        mul_99: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_95, div_21);  mul_95 = div_21 = None
        slice_749: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2);  arg10_1 = None
        slice_750: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_749, 1, 1, -2);  slice_749 = None
        mul_100: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_99, slice_750);  mul_99 = slice_750 = None
        copy_27: "f64[200, 201, 26]" = torch.ops.aten.copy.default(select_78, mul_100);  select_78 = mul_100 = None
        select_scatter_16: "f64[200, 201, 26, 2]" = torch.ops.aten.select_scatter.default(select_79, copy_27, 3, 1);  select_79 = copy_27 = None
        select_scatter_17: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_761, select_scatter_16, 3, 1);  slice_761 = select_scatter_16 = None
        slice_scatter_76: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_760, select_scatter_17, 1, 1, -2);  slice_760 = select_scatter_17 = None
        slice_scatter_77: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_71, slice_scatter_76, 0, 2, -2);  slice_scatter_71 = slice_scatter_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:208 in isoneutral_diffusion_pre, code: K_22[2:-2, 1:-2, :] = sumz / (4.0 * dzt[None, None, :])
        slice_766: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        slice_764: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        slice_765: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_764, 1, 1, -2);  slice_764 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:199 in isoneutral_diffusion_pre, code: sumz[:, :, ki:] += (
        slice_746: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_73, 0, 2, -2)
        slice_742: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_73, 0, 2, -2)
        slice_743: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_742, 1, 1, -2);  slice_742 = None
        slice_scatter_74: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_746, slice_743, 1, 1, -2);  slice_746 = slice_743 = None
        slice_scatter_75: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_73, slice_scatter_74, 0, 2, -2);  slice_scatter_73 = slice_scatter_74 = None
        slice_747: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_75, 0, 2, -2);  slice_scatter_75 = None
        slice_748: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_747, 1, 1, -2);  slice_747 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:208 in isoneutral_diffusion_pre, code: K_22[2:-2, 1:-2, :] = sumz / (4.0 * dzt[None, None, :])
        unsqueeze_34: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        unsqueeze_35: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 1);  unsqueeze_34 = None
        mul_101: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_35, 4.0);  unsqueeze_35 = None
        div_23: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(slice_748, mul_101);  slice_748 = mul_101 = None
        copy_28: "f64[200, 201, 26]" = torch.ops.aten.copy.default(slice_765, div_23);  slice_765 = div_23 = None
        slice_scatter_78: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_766, copy_28, 1, 1, -2);  slice_766 = copy_28 = None
        slice_scatter_79: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg16_1, slice_scatter_78, 0, 2, -2);  slice_scatter_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        _tensor_constant17: "f32[1]" = self._tensor_constant17;  _tensor_constant17 = None
        _tensor_constant18: "f32[1]" = self._tensor_constant18;  _tensor_constant18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:256 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        _tensor_constant19: "f32[1]" = self._tensor_constant19;  _tensor_constant19 = None
        _tensor_constant20: "f32[1]" = self._tensor_constant20;  _tensor_constant20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        _tensor_constant21: "f32[1]" = self._tensor_constant21;  _tensor_constant21 = None
        _tensor_constant22: "f32[1]" = self._tensor_constant22;  _tensor_constant22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:246 in isoneutral_diffusion_pre, code: Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]
        slice_841: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_842: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_841, 1, 2, -2)
        slice_843: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_842, 2, 0, -1)
        select_84: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_843, 3, 0)
        slice_838: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_839: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_838, 1, 2, -2);  slice_838 = None
        slice_840: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_839, 2, 0, -1);  slice_839 = None
        select_82: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_840, 3, 0);  slice_840 = None
        select_83: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_82, 3, 0);  select_82 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_799: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_800: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_799, 1, 2, -2);  slice_799 = None
        slice_801: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_800, 2, 0, 25);  slice_800 = None
        slice_808: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -3)
        slice_809: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_808, 1, 2, -2);  slice_808 = None
        slice_810: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_809, 2, 0, 25);  slice_809 = None
        mul_104: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_801, slice_810);  slice_801 = slice_810 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:233 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_811: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_812: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_811, 1, 2, -2);  slice_811 = None
        slice_813: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_812, 2, 0, 25);  slice_812 = None
        slice_820: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 1, -3)
        slice_821: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_820, 1, 2, -2);  slice_820 = None
        slice_822: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_821, 2, 0, 25);  slice_821 = None
        mul_105: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_813, slice_822);  slice_813 = slice_822 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        add_50: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        neg_18: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_50);  add_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_18: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:225 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdz[2:-2, 2:-2, :-1]
        slice_775: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_776: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_775, 1, 2, -2);  slice_775 = None
        slice_777: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_776, 2, 0, 25);  slice_776 = None
        slice_784: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -2)
        slice_785: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_784, 1, 2, -2);  slice_784 = None
        slice_786: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_785, 2, 0, -1);  slice_785 = None
        mul_102: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_777, slice_786);  slice_777 = slice_786 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:226 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdz[2:-2, 2:-2, :-1]
        slice_787: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_788: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_787, 1, 2, -2);  slice_787 = None
        slice_789: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_788, 2, 0, 25);  slice_788 = None
        slice_796: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -2)
        slice_797: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_796, 1, 2, -2);  slice_796 = None
        slice_798: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_797, 2, 0, -1);  slice_797 = None
        mul_103: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_789, slice_798);  slice_789 = slice_798 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:225 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdz[2:-2, 2:-2, :-1]
        add_49: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        minimum_8: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_18, add_49);  full_default_18 = None
        sub_17: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_8, 1e-20);  minimum_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        div_24: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_18, sub_17);  neg_18 = sub_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_11: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_24)
        neg_19: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_11);  abs_11 = None
        add_51: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_19, 0.001);  neg_19 = None
        div_25: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_51, 0.001);  add_51 = None
        tanh_8: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_25);  div_25 = None
        add_52: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_106: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_52, 0.5);  add_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:246 in isoneutral_diffusion_pre, code: Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]
        mul_111: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_106, div_24)
        slice_835: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_836: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_835, 1, 2, -2);  slice_835 = None
        slice_837: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_836, 2, 0, -1);  slice_836 = None
        mul_112: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_111, slice_837);  mul_111 = slice_837 = None
        copy_29: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_83, mul_112);  select_83 = mul_112 = None
        select_scatter_18: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_84, copy_29, 3, 0);  select_84 = copy_29 = None
        select_scatter_19: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_843, select_scatter_18, 3, 0);  slice_843 = select_scatter_18 = None
        slice_scatter_83: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_842, select_scatter_19, 2, 0, -1);  slice_842 = select_scatter_19 = None
        slice_scatter_84: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_841, slice_scatter_83, 1, 2, -2);  slice_841 = slice_scatter_83 = None
        slice_scatter_85: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg17_1, slice_scatter_84, 0, 2, -2);  slice_scatter_84 = None
        slice_901: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_85, 0, 2, -2)
        slice_902: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_901, 1, 2, -2)
        slice_903: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_902, 2, 0, -1)
        select_92: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_903, 3, 1)
        slice_898: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_85, 0, 2, -2)
        slice_899: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_898, 1, 2, -2);  slice_898 = None
        slice_900: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_899, 2, 0, -1);  slice_899 = None
        select_90: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_900, 3, 1);  slice_900 = None
        select_91: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_90, 3, 0);  select_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_847: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_848: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_847, 1, 2, -2);  slice_847 = None
        slice_849: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_848, 2, 0, 25);  slice_848 = None
        slice_856: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 2, -2)
        slice_857: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_856, 1, 2, -2);  slice_856 = None
        slice_858: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_857, 2, 0, 25);  slice_857 = None
        mul_113: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_849, slice_858);  slice_849 = slice_858 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:233 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_859: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_860: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_859, 1, 2, -2);  slice_859 = None
        slice_861: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_860, 2, 0, 25);  slice_860 = None
        slice_868: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 2, -2)
        slice_869: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_868, 1, 2, -2);  slice_868 = None
        slice_870: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_869, 2, 0, 25);  slice_869 = None
        mul_114: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_861, slice_870);  slice_861 = slice_870 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        add_54: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        neg_20: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_54);  add_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_19: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_9: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_19, add_49);  full_default_19 = None
        sub_18: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_9, 1e-20);  minimum_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        div_26: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_20, sub_18);  neg_20 = sub_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_12: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_26)
        neg_21: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_12);  abs_12 = None
        add_55: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_21, 0.001);  neg_21 = None
        div_27: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_55, 0.001);  add_55 = None
        tanh_9: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_27);  div_27 = None
        add_56: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_115: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:246 in isoneutral_diffusion_pre, code: Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]
        mul_120: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_115, div_26)
        slice_883: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_884: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_883, 1, 2, -2);  slice_883 = None
        slice_885: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_884, 2, 0, -1);  slice_884 = None
        mul_121: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_120, slice_885);  mul_120 = slice_885 = None
        copy_30: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_91, mul_121);  select_91 = mul_121 = None
        select_scatter_20: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_92, copy_30, 3, 0);  select_92 = copy_30 = None
        select_scatter_21: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_903, select_scatter_20, 3, 1);  slice_903 = select_scatter_20 = None
        slice_scatter_89: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_902, select_scatter_21, 2, 0, -1);  slice_902 = select_scatter_21 = None
        slice_scatter_90: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_901, slice_scatter_89, 1, 2, -2);  slice_901 = slice_scatter_89 = None
        slice_scatter_91: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_85, slice_scatter_90, 0, 2, -2);  slice_scatter_85 = slice_scatter_90 = None
        slice_1095: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_91, 0, 2, -2)
        slice_1096: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1095, 1, 2, -2)
        slice_1097: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1096, 2, 0, -1)
        select_113: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1097, 3, 0)
        slice_1092: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_91, 0, 2, -2)
        slice_1093: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1092, 1, 2, -2);  slice_1092 = None
        slice_1094: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1093, 2, 0, -1);  slice_1093 = None
        select_111: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1094, 3, 0);  slice_1094 = None
        select_112: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_111, 3, 1);  select_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_1041: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_1042: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1041, 1, 2, -2);  slice_1041 = None
        slice_1043: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1042, 2, 1, 26);  slice_1042 = None
        slice_1050: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -3)
        slice_1051: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1050, 1, 2, -2);  slice_1050 = None
        slice_1052: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1051, 2, 1, 26);  slice_1051 = None
        mul_144: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1043, slice_1052);  slice_1043 = slice_1052 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:233 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_1053: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_1054: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1053, 1, 2, -2);  slice_1053 = None
        slice_1055: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1054, 2, 1, 26);  slice_1054 = None
        slice_1062: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 1, -3)
        slice_1063: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1062, 1, 2, -2);  slice_1062 = None
        slice_1064: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1063, 2, 1, 26);  slice_1063 = None
        mul_145: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1055, slice_1064);  slice_1055 = slice_1064 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        add_67: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        neg_26: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_67);  add_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_22: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:225 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdz[2:-2, 2:-2, :-1]
        slice_1017: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_1018: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1017, 1, 2, -2);  slice_1017 = None
        slice_1019: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1018, 2, 1, 26);  slice_1018 = None
        slice_1026: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter, 0, 2, -2);  slice_scatter = None
        slice_1027: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1026, 1, 2, -2);  slice_1026 = None
        slice_1028: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1027, 2, 0, -1);  slice_1027 = None
        mul_142: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1019, slice_1028);  slice_1019 = slice_1028 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:226 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdz[2:-2, 2:-2, :-1]
        slice_1029: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_1030: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1029, 1, 2, -2);  slice_1029 = None
        slice_1031: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1030, 2, 1, 26);  slice_1030 = None
        slice_1038: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 2, -2);  slice_scatter_1 = None
        slice_1039: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1038, 1, 2, -2);  slice_1038 = None
        slice_1040: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1039, 2, 0, -1);  slice_1039 = None
        mul_143: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1031, slice_1040);  slice_1031 = slice_1040 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:225 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdz[2:-2, 2:-2, :-1]
        add_66: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        minimum_12: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_22, add_66);  full_default_22 = None
        sub_21: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_12, 1e-20);  minimum_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        div_32: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_26, sub_21);  neg_26 = sub_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_15: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_32)
        neg_27: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_15);  abs_15 = None
        add_68: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_27, 0.001);  neg_27 = None
        div_33: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_68, 0.001);  add_68 = None
        tanh_12: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_33);  div_33 = None
        add_69: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_146: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:246 in isoneutral_diffusion_pre, code: Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]
        mul_151: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_146, div_32)
        slice_1077: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_1078: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1077, 1, 2, -2);  slice_1077 = None
        slice_1079: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1078, 2, 0, -1);  slice_1078 = None
        mul_152: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_151, slice_1079);  mul_151 = slice_1079 = None
        copy_33: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_112, mul_152);  select_112 = mul_152 = None
        select_scatter_26: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_113, copy_33, 3, 1);  select_113 = copy_33 = None
        select_scatter_27: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_1097, select_scatter_26, 3, 0);  slice_1097 = select_scatter_26 = None
        slice_scatter_107: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1096, select_scatter_27, 2, 0, -1);  slice_1096 = select_scatter_27 = None
        slice_scatter_108: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1095, slice_scatter_107, 1, 2, -2);  slice_1095 = slice_scatter_107 = None
        slice_scatter_109: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_91, slice_scatter_108, 0, 2, -2);  slice_scatter_91 = slice_scatter_108 = None
        slice_1155: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_109, 0, 2, -2)
        slice_1156: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1155, 1, 2, -2)
        slice_1157: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1156, 2, 0, -1)
        select_121: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1157, 3, 1)
        slice_1152: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_109, 0, 2, -2)
        slice_1153: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1152, 1, 2, -2);  slice_1152 = None
        slice_1154: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1153, 2, 0, -1);  slice_1153 = None
        select_119: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1154, 3, 1);  slice_1154 = None
        select_120: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_119, 3, 1);  select_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_1101: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_1102: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1101, 1, 2, -2);  slice_1101 = None
        slice_1103: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1102, 2, 1, 26);  slice_1102 = None
        slice_1110: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 2, -2);  slice_scatter_2 = None
        slice_1111: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1110, 1, 2, -2);  slice_1110 = None
        slice_1112: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1111, 2, 1, 26);  slice_1111 = None
        mul_153: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1103, slice_1112);  slice_1103 = slice_1112 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:233 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdx[1 + ip : -3 + ip, 2:-2, sl:su]
        slice_1113: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_1114: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1113, 1, 2, -2);  slice_1113 = None
        slice_1115: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1114, 2, 1, 26);  slice_1114 = None
        slice_1122: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 2, -2);  slice_scatter_3 = None
        slice_1123: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1122, 1, 2, -2);  slice_1122 = None
        slice_1124: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1123, 2, 1, 26);  slice_1123 = None
        mul_154: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1115, slice_1124);  slice_1115 = slice_1124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:232 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdx[1 + ip : -3 + ip, 2:-2, sl:su]
        add_71: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        neg_28: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_71);  add_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:236 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_23: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_13: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_23, add_66);  full_default_23 = None
        sub_22: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_13, 1e-20);  minimum_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:235 in isoneutral_diffusion_pre, code: sxb = -drodxb / (
        div_34: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_28, sub_22);  neg_28 = sub_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_16: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_34)
        neg_29: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_16);  abs_16 = None
        add_72: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_29, 0.001);  neg_29 = None
        div_35: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_72, 0.001);  add_72 = None
        tanh_13: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_35);  div_35 = None
        add_73: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_155: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_73, 0.5);  add_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:246 in isoneutral_diffusion_pre, code: Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]
        mul_160: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_155, div_34)
        slice_1137: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_1138: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1137, 1, 2, -2);  slice_1137 = None
        slice_1139: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1138, 2, 0, -1);  slice_1138 = None
        mul_161: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_160, slice_1139);  mul_160 = slice_1139 = None
        copy_34: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_120, mul_161);  select_120 = mul_161 = None
        select_scatter_28: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_121, copy_34, 3, 1);  select_121 = copy_34 = None
        select_scatter_29: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_1157, select_scatter_28, 3, 1);  slice_1157 = select_scatter_28 = None
        slice_scatter_113: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1156, select_scatter_29, 2, 0, -1);  slice_1156 = select_scatter_29 = None
        slice_scatter_114: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1155, slice_scatter_113, 1, 2, -2);  slice_1155 = slice_scatter_113 = None
        slice_scatter_115: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_109, slice_scatter_114, 0, 2, -2);  slice_scatter_109 = slice_scatter_114 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:256 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        _tensor_constant23: "f32[1]" = self._tensor_constant23;  _tensor_constant23 = None
        _tensor_constant24: "f32[1]" = self._tensor_constant24;  _tensor_constant24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:266 in isoneutral_diffusion_pre, code: Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * maskW[2:-2, 2:-2, :-1]
        slice_950: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 2, -2)
        slice_951: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_950, 1, 2, -2)
        slice_952: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_951, 2, 0, -1)
        select_97: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_952, 3, 0)
        slice_947: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 2, -2)
        slice_948: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_947, 1, 2, -2);  slice_947 = None
        slice_949: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_948, 2, 0, -1);  slice_948 = None
        select_95: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_949, 3, 0);  slice_949 = None
        select_96: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_95, 3, 0);  select_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_909: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_910: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_909, 1, 2, -2);  slice_909 = None
        slice_911: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_910, 2, 0, 25);  slice_910 = None
        slice_918: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2)
        slice_919: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_918, 1, 1, -3);  slice_918 = None
        slice_920: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_919, 2, 0, 25);  slice_919 = None
        mul_123: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_911, slice_920);  slice_911 = slice_920 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:253 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_921: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_922: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_921, 1, 2, -2);  slice_921 = None
        slice_923: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_922, 2, 0, 25);  slice_922 = None
        slice_930: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_931: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_930, 1, 1, -3);  slice_930 = None
        slice_932: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_931, 2, 0, 25);  slice_931 = None
        mul_124: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_923, slice_932);  slice_923 = slice_932 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        add_58: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        neg_22: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_58);  add_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:256 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_20: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_10: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_20, add_49);  full_default_20 = None
        sub_19: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_10, 1e-20);  minimum_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        div_28: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_22, sub_19);  neg_22 = sub_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_13: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_28)
        neg_23: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_13);  abs_13 = None
        add_59: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_23, 0.001);  neg_23 = None
        div_29: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_59, 0.001);  add_59 = None
        tanh_10: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_29);  div_29 = None
        add_60: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_125: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_60, 0.5);  add_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:266 in isoneutral_diffusion_pre, code: Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * maskW[2:-2, 2:-2, :-1]
        mul_130: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_125, div_28)
        slice_944: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_945: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_944, 1, 2, -2);  slice_944 = None
        slice_946: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_945, 2, 0, -1);  slice_945 = None
        mul_131: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_130, slice_946);  mul_130 = slice_946 = None
        copy_31: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_96, mul_131);  select_96 = mul_131 = None
        select_scatter_22: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_97, copy_31, 3, 0);  select_97 = copy_31 = None
        select_scatter_23: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_952, select_scatter_22, 3, 0);  slice_952 = select_scatter_22 = None
        slice_scatter_95: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_951, select_scatter_23, 2, 0, -1);  slice_951 = select_scatter_23 = None
        slice_scatter_96: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_950, slice_scatter_95, 1, 2, -2);  slice_950 = slice_scatter_95 = None
        slice_scatter_97: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg19_1, slice_scatter_96, 0, 2, -2);  slice_scatter_96 = None
        slice_1011: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_97, 0, 2, -2)
        slice_1012: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1011, 1, 2, -2)
        slice_1013: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1012, 2, 0, -1)
        select_105: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1013, 3, 1)
        slice_1008: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_97, 0, 2, -2)
        slice_1009: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1008, 1, 2, -2);  slice_1008 = None
        slice_1010: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1009, 2, 0, -1);  slice_1009 = None
        select_103: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1010, 3, 1);  slice_1010 = None
        select_104: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_103, 3, 0);  select_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_958: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_959: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_958, 1, 2, -2);  slice_958 = None
        slice_960: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_959, 2, 0, 25);  slice_959 = None
        slice_967: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2)
        slice_968: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_967, 1, 2, -2);  slice_967 = None
        slice_969: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_968, 2, 0, 25);  slice_968 = None
        mul_133: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_960, slice_969);  slice_960 = slice_969 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:253 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_970: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_971: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_970, 1, 2, -2);  slice_970 = None
        slice_972: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_971, 2, 0, 25);  slice_971 = None
        slice_979: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_980: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_979, 1, 2, -2);  slice_979 = None
        slice_981: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_980, 2, 0, 25);  slice_980 = None
        mul_134: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_972, slice_981);  slice_972 = slice_981 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        add_62: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        neg_24: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_62);  add_62 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:256 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_21: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_11: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_21, add_49);  full_default_21 = add_49 = None
        sub_20: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_11, 1e-20);  minimum_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        div_30: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_24, sub_20);  neg_24 = sub_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_14: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_30)
        neg_25: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_14);  abs_14 = None
        add_63: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_25, 0.001);  neg_25 = None
        div_31: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_63, 0.001);  add_63 = None
        tanh_11: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_31);  div_31 = None
        add_64: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_135: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:266 in isoneutral_diffusion_pre, code: Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * maskW[2:-2, 2:-2, :-1]
        mul_140: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_135, div_30)
        slice_993: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_994: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_993, 1, 2, -2);  slice_993 = None
        slice_995: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_994, 2, 0, -1);  slice_994 = None
        mul_141: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_140, slice_995);  mul_140 = slice_995 = None
        copy_32: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_104, mul_141);  select_104 = mul_141 = None
        select_scatter_24: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_105, copy_32, 3, 0);  select_105 = copy_32 = None
        select_scatter_25: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_1013, select_scatter_24, 3, 1);  slice_1013 = select_scatter_24 = None
        slice_scatter_101: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1012, select_scatter_25, 2, 0, -1);  slice_1012 = select_scatter_25 = None
        slice_scatter_102: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1011, slice_scatter_101, 1, 2, -2);  slice_1011 = slice_scatter_101 = None
        slice_scatter_103: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_97, slice_scatter_102, 0, 2, -2);  slice_scatter_97 = slice_scatter_102 = None
        slice_1216: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_103, 0, 2, -2)
        slice_1217: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1216, 1, 2, -2)
        slice_1218: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1217, 2, 0, -1)
        select_129: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1218, 3, 0)
        slice_1213: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_103, 0, 2, -2)
        slice_1214: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1213, 1, 2, -2);  slice_1213 = None
        slice_1215: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1214, 2, 0, -1);  slice_1214 = None
        select_127: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1215, 3, 0);  slice_1215 = None
        select_128: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_127, 3, 1);  select_127 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_1163: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2)
        slice_1164: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1163, 1, 2, -2);  slice_1163 = None
        slice_1165: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1164, 2, 1, 26);  slice_1164 = None
        slice_1172: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2)
        slice_1173: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1172, 1, 1, -3);  slice_1172 = None
        slice_1174: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1173, 2, 1, 26);  slice_1173 = None
        mul_163: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1165, slice_1174);  slice_1165 = slice_1174 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:253 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_1175: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2)
        slice_1176: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1175, 1, 2, -2);  slice_1175 = None
        slice_1177: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1176, 2, 1, 26);  slice_1176 = None
        slice_1184: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_1185: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1184, 1, 1, -3);  slice_1184 = None
        slice_1186: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1185, 2, 1, 26);  slice_1185 = None
        mul_164: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1177, slice_1186);  slice_1177 = slice_1186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        add_75: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        neg_30: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_75);  add_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:256 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_24: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_14: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_24, add_66);  full_default_24 = None
        sub_23: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_14, 1e-20);  minimum_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        div_36: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_30, sub_23);  neg_30 = sub_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_17: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_36)
        neg_31: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_17);  abs_17 = None
        add_76: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_31, 0.001);  neg_31 = None
        div_37: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_76, 0.001);  add_76 = None
        tanh_14: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_37);  div_37 = None
        add_77: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_165: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:266 in isoneutral_diffusion_pre, code: Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * maskW[2:-2, 2:-2, :-1]
        mul_170: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_165, div_36)
        slice_1198: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_1199: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1198, 1, 2, -2);  slice_1198 = None
        slice_1200: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1199, 2, 0, -1);  slice_1199 = None
        mul_171: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_170, slice_1200);  mul_170 = slice_1200 = None
        copy_35: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_128, mul_171);  select_128 = mul_171 = None
        select_scatter_30: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_129, copy_35, 3, 1);  select_129 = copy_35 = None
        select_scatter_31: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_1218, select_scatter_30, 3, 0);  slice_1218 = select_scatter_30 = None
        slice_scatter_119: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1217, select_scatter_31, 2, 0, -1);  slice_1217 = select_scatter_31 = None
        slice_scatter_120: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1216, slice_scatter_119, 1, 2, -2);  slice_1216 = slice_scatter_119 = None
        slice_scatter_121: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_103, slice_scatter_120, 0, 2, -2);  slice_scatter_103 = slice_scatter_120 = None
        slice_1277: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_121, 0, 2, -2)
        slice_1278: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1277, 1, 2, -2)
        slice_1279: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1278, 2, 0, -1)
        select_137: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1279, 3, 1)
        slice_1274: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_121, 0, 2, -2)
        slice_1275: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1274, 1, 2, -2);  slice_1274 = None
        slice_1276: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_1275, 2, 0, -1);  slice_1275 = None
        select_135: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_1276, 3, 1);  slice_1276 = None
        select_136: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_135, 3, 1);  select_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_1224: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_5, 0, 2, -2);  mul_5 = None
        slice_1225: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1224, 1, 2, -2);  slice_1224 = None
        slice_1226: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1225, 2, 1, 26);  slice_1225 = None
        slice_1233: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 2, -2);  slice_scatter_4 = None
        slice_1234: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1233, 1, 2, -2);  slice_1233 = None
        slice_1235: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1234, 2, 1, 26);  slice_1234 = None
        mul_173: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1226, slice_1235);  slice_1226 = slice_1235 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:253 in isoneutral_diffusion_pre, code: + drdS[2:-2, 2:-2, sl:su] * dSdy[2:-2, 1 + jp : -3 + jp, sl:su]
        slice_1236: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_7, 0, 2, -2);  mul_7 = None
        slice_1237: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1236, 1, 2, -2);  slice_1236 = None
        slice_1238: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1237, 2, 1, 26);  slice_1237 = None
        slice_1245: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2);  slice_scatter_5 = None
        slice_1246: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1245, 1, 2, -2);  slice_1245 = None
        slice_1247: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1246, 2, 1, 26);  slice_1246 = None
        mul_174: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_1238, slice_1247);  slice_1238 = slice_1247 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:252 in isoneutral_diffusion_pre, code: drdT[2:-2, 2:-2, sl:su] * dTdy[2:-2, 1 + jp : -3 + jp, sl:su]
        add_79: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        neg_32: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_79);  add_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:256 in isoneutral_diffusion_pre, code: torch.min(torch.tensor([0.0], device=device), drodzb) - epsln
        full_default_25: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_15: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_25, add_66);  full_default_25 = add_66 = None
        sub_24: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_15, 1e-20);  minimum_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:255 in isoneutral_diffusion_pre, code: syb = -drodyb / (
        div_38: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_32, sub_24);  neg_32 = sub_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:30 in dm_taper, code: return 0.5 * (1.0 + torch.tanh((-torch.abs(sx) + iso_slopec) / iso_dslope))
        abs_18: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_38)
        neg_33: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_18);  abs_18 = None
        add_80: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_33, 0.001);  neg_33 = None
        div_39: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_80, 0.001);  add_80 = None
        tanh_15: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_39);  div_39 = None
        add_81: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_175: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_81, 0.5);  add_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:266 in isoneutral_diffusion_pre, code: Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * maskW[2:-2, 2:-2, :-1]
        mul_180: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_175, div_38)
        slice_1259: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_1260: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1259, 1, 2, -2);  slice_1259 = None
        slice_1261: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1260, 2, 0, -1);  slice_1260 = None
        mul_181: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_180, slice_1261);  mul_180 = slice_1261 = None
        copy_36: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_136, mul_181);  select_136 = mul_181 = None
        select_scatter_32: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_137, copy_36, 3, 1);  select_137 = copy_36 = None
        select_scatter_33: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_1279, select_scatter_32, 3, 1);  slice_1279 = select_scatter_32 = None
        slice_scatter_125: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1278, select_scatter_33, 2, 0, -1);  slice_1278 = select_scatter_33 = None
        slice_scatter_126: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_1277, slice_scatter_125, 1, 2, -2);  slice_1277 = slice_scatter_125 = None
        slice_scatter_127: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_121, slice_scatter_126, 0, 2, -2);  slice_scatter_121 = slice_scatter_126 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:271 in isoneutral_diffusion_pre, code: K_33[2:-2, 2:-2, -1] = 0.0
        _tensor_constant25: "f64[]" = self._tensor_constant25;  _tensor_constant25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:268 in isoneutral_diffusion_pre, code: K_33[2:-2, 2:-2, :-1] = sumx / (4 * dxt[2:-2, None, None]) + sumy / (
        slice_1289: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg22_1, 0, 2, -2)
        slice_1290: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1289, 1, 2, -2)
        slice_1286: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg22_1, 0, 2, -2)
        slice_1287: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1286, 1, 2, -2);  slice_1286 = None
        slice_1288: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1287, 2, 0, -1);  slice_1287 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:213 in isoneutral_diffusion_pre, code: sumx = torch.zeros_like(K_11)[2:-2, 2:-2, :-1]
        full_10: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:239 in isoneutral_diffusion_pre, code: sumx += (
        slice_830: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_10, 0, 2, -2)
        slice_831: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_830, 1, 2, -2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:213 in isoneutral_diffusion_pre, code: sumx = torch.zeros_like(K_11)[2:-2, 2:-2, :-1]
        slice_769: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_10, 0, 2, -2)
        slice_770: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_769, 1, 2, -2);  slice_769 = None
        slice_771: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_770, 2, 0, -1);  slice_770 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        slice_823: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 1, -3)
        unsqueeze_36: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_823, 1);  slice_823 = None
        unsqueeze_37: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:241 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_824: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_825: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_824, 1, 2, -2);  slice_824 = None
        slice_826: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_825, 2, 0, -1);  slice_825 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_107: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_37, slice_826);  unsqueeze_37 = slice_826 = None
        mul_108: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_107, mul_106);  mul_107 = mul_106 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:243 in isoneutral_diffusion_pre, code: * sxb**2
        pow_1: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_24, 2);  div_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_109: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_108, pow_1);  mul_108 = pow_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:244 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_827: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_828: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_827, 1, 2, -2);  slice_827 = None
        slice_829: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_828, 2, 0, -1);  slice_828 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_110: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_109, slice_829);  mul_109 = slice_829 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:239 in isoneutral_diffusion_pre, code: sumx += (
        add_53: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_771, mul_110);  slice_771 = mul_110 = None
        slice_scatter_80: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_831, add_53, 2, 0, -1);  slice_831 = add_53 = None
        slice_scatter_81: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_830, slice_scatter_80, 1, 2, -2);  slice_830 = slice_scatter_80 = None
        slice_scatter_82: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_81, 0, 2, -2);  full_10 = slice_scatter_81 = None
        slice_878: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_82, 0, 2, -2)
        slice_879: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_878, 1, 2, -2)
        slice_832: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_82, 0, 2, -2)
        slice_833: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_832, 1, 2, -2);  slice_832 = None
        slice_834: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_833, 2, 0, -1);  slice_833 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        slice_871: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2)
        unsqueeze_38: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_871, 1);  slice_871 = None
        unsqueeze_39: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 2);  unsqueeze_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:241 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_872: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_873: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_872, 1, 2, -2);  slice_872 = None
        slice_874: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_873, 2, 0, -1);  slice_873 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_116: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_39, slice_874);  unsqueeze_39 = slice_874 = None
        mul_117: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_116, mul_115);  mul_116 = mul_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:243 in isoneutral_diffusion_pre, code: * sxb**2
        pow_2: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_26, 2);  div_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_118: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_117, pow_2);  mul_117 = pow_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:244 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_875: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_876: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_875, 1, 2, -2);  slice_875 = None
        slice_877: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_876, 2, 0, -1);  slice_876 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_119: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_118, slice_877);  mul_118 = slice_877 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:239 in isoneutral_diffusion_pre, code: sumx += (
        add_57: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_834, mul_119);  slice_834 = mul_119 = None
        slice_scatter_86: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_879, add_57, 2, 0, -1);  slice_879 = add_57 = None
        slice_scatter_87: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_878, slice_scatter_86, 1, 2, -2);  slice_878 = slice_scatter_86 = None
        slice_scatter_88: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_82, slice_scatter_87, 0, 2, -2);  slice_scatter_82 = slice_scatter_87 = None
        slice_1072: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_88, 0, 2, -2)
        slice_1073: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1072, 1, 2, -2)
        slice_880: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_88, 0, 2, -2)
        slice_881: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_880, 1, 2, -2);  slice_880 = None
        slice_882: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_881, 2, 0, -1);  slice_881 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        slice_1065: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 1, -3)
        unsqueeze_44: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_1065, 1);  slice_1065 = None
        unsqueeze_45: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, 2);  unsqueeze_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:241 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_1066: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_1067: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1066, 1, 2, -2);  slice_1066 = None
        slice_1068: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1067, 2, 0, -1);  slice_1067 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_147: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_45, slice_1068);  unsqueeze_45 = slice_1068 = None
        mul_148: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_147, mul_146);  mul_147 = mul_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:243 in isoneutral_diffusion_pre, code: * sxb**2
        pow_5: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_32, 2);  div_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_149: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_148, pow_5);  mul_148 = pow_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:244 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_1069: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_1070: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1069, 1, 2, -2);  slice_1069 = None
        slice_1071: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1070, 2, 0, -1);  slice_1070 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_150: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_149, slice_1071);  mul_149 = slice_1071 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:239 in isoneutral_diffusion_pre, code: sumx += (
        add_70: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_882, mul_150);  slice_882 = mul_150 = None
        slice_scatter_104: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_1073, add_70, 2, 0, -1);  slice_1073 = add_70 = None
        slice_scatter_105: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_1072, slice_scatter_104, 1, 2, -2);  slice_1072 = slice_scatter_104 = None
        slice_scatter_106: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_88, slice_scatter_105, 0, 2, -2);  slice_scatter_88 = slice_scatter_105 = None
        slice_1132: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_106, 0, 2, -2)
        slice_1133: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1132, 1, 2, -2)
        slice_1074: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_106, 0, 2, -2)
        slice_1075: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1074, 1, 2, -2);  slice_1074 = None
        slice_1076: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1075, 2, 0, -1);  slice_1075 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        slice_1125: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2);  arg8_1 = None
        unsqueeze_46: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_1125, 1);  slice_1125 = None
        unsqueeze_47: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:241 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_1126: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_1127: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1126, 1, 2, -2);  slice_1126 = None
        slice_1128: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1127, 2, 0, -1);  slice_1127 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_156: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_47, slice_1128);  unsqueeze_47 = slice_1128 = None
        mul_157: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_156, mul_155);  mul_156 = mul_155 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:243 in isoneutral_diffusion_pre, code: * sxb**2
        pow_6: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_34, 2);  div_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_158: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_157, pow_6);  mul_157 = pow_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:244 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_1129: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_1130: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1129, 1, 2, -2);  slice_1129 = None
        slice_1131: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1130, 2, 0, -1);  slice_1130 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:240 in isoneutral_diffusion_pre, code: dxu[1 + ip : -3 + ip, None, None]
        mul_159: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_158, slice_1131);  mul_158 = slice_1131 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:239 in isoneutral_diffusion_pre, code: sumx += (
        add_74: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_1076, mul_159);  slice_1076 = mul_159 = None
        slice_scatter_110: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_1133, add_74, 2, 0, -1);  slice_1133 = add_74 = None
        slice_scatter_111: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_1132, slice_scatter_110, 1, 2, -2);  slice_1132 = slice_scatter_110 = None
        slice_scatter_112: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_106, slice_scatter_111, 0, 2, -2);  slice_scatter_106 = slice_scatter_111 = None
        slice_1134: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_112, 0, 2, -2);  slice_scatter_112 = None
        slice_1135: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1134, 1, 2, -2);  slice_1134 = None
        slice_1136: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1135, 2, 0, -1);  slice_1135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:268 in isoneutral_diffusion_pre, code: K_33[2:-2, 2:-2, :-1] = sumx / (4 * dxt[2:-2, None, None]) + sumy / (
        slice_1283: "f64[200]" = torch.ops.aten.slice.Tensor(arg20_1, 0, 2, -2);  arg20_1 = None
        unsqueeze_52: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_1283, 1);  slice_1283 = None
        unsqueeze_53: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 2);  unsqueeze_52 = None
        mul_182: "f64[200, 1, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_53, 4);  unsqueeze_53 = None
        div_40: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(slice_1136, mul_182);  slice_1136 = mul_182 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:214 in isoneutral_diffusion_pre, code: sumy = torch.zeros_like(K_11)[2:-2, 2:-2, :-1]
        full_11: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:259 in isoneutral_diffusion_pre, code: sumy += (
        slice_939: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_11, 0, 2, -2)
        slice_940: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_939, 1, 2, -2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:214 in isoneutral_diffusion_pre, code: sumy = torch.zeros_like(K_11)[2:-2, 2:-2, :-1]
        slice_772: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_11, 0, 2, -2)
        slice_773: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_772, 1, 2, -2);  slice_772 = None
        slice_774: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_773, 2, 0, -1);  slice_773 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:250 in isoneutral_diffusion_pre, code: facty = cosu[1 + jp : -3 + jp] * dyu[1 + jp : -3 + jp]
        slice_907: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 1, -3)
        slice_908: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 1, -3)
        mul_122: "f64[200]" = torch.ops.aten.mul.Tensor(slice_907, slice_908);  slice_907 = slice_908 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        unsqueeze_40: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_122, 0);  mul_122 = None
        unsqueeze_41: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 2);  unsqueeze_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:261 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_933: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_934: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_933, 1, 2, -2);  slice_933 = None
        slice_935: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_934, 2, 0, -1);  slice_934 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_126: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_41, slice_935);  unsqueeze_41 = slice_935 = None
        mul_127: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_126, mul_125);  mul_126 = mul_125 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:263 in isoneutral_diffusion_pre, code: * syb**2
        pow_3: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_28, 2);  div_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_128: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_127, pow_3);  mul_127 = pow_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:264 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_936: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_937: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_936, 1, 2, -2);  slice_936 = None
        slice_938: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_937, 2, 0, -1);  slice_937 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_129: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_128, slice_938);  mul_128 = slice_938 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:259 in isoneutral_diffusion_pre, code: sumy += (
        add_61: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_774, mul_129);  slice_774 = mul_129 = None
        slice_scatter_92: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_940, add_61, 2, 0, -1);  slice_940 = add_61 = None
        slice_scatter_93: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_939, slice_scatter_92, 1, 2, -2);  slice_939 = slice_scatter_92 = None
        slice_scatter_94: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_11, slice_scatter_93, 0, 2, -2);  full_11 = slice_scatter_93 = None
        slice_988: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 0, 2, -2)
        slice_989: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_988, 1, 2, -2)
        slice_941: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 0, 2, -2)
        slice_942: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_941, 1, 2, -2);  slice_941 = None
        slice_943: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_942, 2, 0, -1);  slice_942 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:250 in isoneutral_diffusion_pre, code: facty = cosu[1 + jp : -3 + jp] * dyu[1 + jp : -3 + jp]
        slice_956: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 2, -2)
        slice_957: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 2, -2)
        mul_132: "f64[200]" = torch.ops.aten.mul.Tensor(slice_956, slice_957);  slice_956 = slice_957 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        unsqueeze_42: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_132, 0);  mul_132 = None
        unsqueeze_43: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:261 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_982: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_983: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_982, 1, 2, -2);  slice_982 = None
        slice_984: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_983, 2, 0, -1);  slice_983 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_136: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_43, slice_984);  unsqueeze_43 = slice_984 = None
        mul_137: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_136, mul_135);  mul_136 = mul_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:263 in isoneutral_diffusion_pre, code: * syb**2
        pow_4: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_30, 2);  div_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_138: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_137, pow_4);  mul_137 = pow_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:264 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_985: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_986: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_985, 1, 2, -2);  slice_985 = None
        slice_987: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_986, 2, 0, -1);  slice_986 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_139: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_138, slice_987);  mul_138 = slice_987 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:259 in isoneutral_diffusion_pre, code: sumy += (
        add_65: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_943, mul_139);  slice_943 = mul_139 = None
        slice_scatter_98: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_989, add_65, 2, 0, -1);  slice_989 = add_65 = None
        slice_scatter_99: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_988, slice_scatter_98, 1, 2, -2);  slice_988 = slice_scatter_98 = None
        slice_scatter_100: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_94, slice_scatter_99, 0, 2, -2);  slice_scatter_94 = slice_scatter_99 = None
        slice_1193: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_100, 0, 2, -2)
        slice_1194: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1193, 1, 2, -2)
        slice_990: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_100, 0, 2, -2)
        slice_991: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_990, 1, 2, -2);  slice_990 = None
        slice_992: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_991, 2, 0, -1);  slice_991 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:250 in isoneutral_diffusion_pre, code: facty = cosu[1 + jp : -3 + jp] * dyu[1 + jp : -3 + jp]
        slice_1161: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 1, -3)
        slice_1162: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 1, -3)
        mul_162: "f64[200]" = torch.ops.aten.mul.Tensor(slice_1161, slice_1162);  slice_1161 = slice_1162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        unsqueeze_48: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_162, 0);  mul_162 = None
        unsqueeze_49: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:261 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_1187: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_1188: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1187, 1, 2, -2);  slice_1187 = None
        slice_1189: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1188, 2, 0, -1);  slice_1188 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_166: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_49, slice_1189);  unsqueeze_49 = slice_1189 = None
        mul_167: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_166, mul_165);  mul_166 = mul_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:263 in isoneutral_diffusion_pre, code: * syb**2
        pow_7: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_36, 2);  div_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_168: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_167, pow_7);  mul_167 = pow_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:264 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_1190: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_1191: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1190, 1, 2, -2);  slice_1190 = None
        slice_1192: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1191, 2, 0, -1);  slice_1191 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_169: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_168, slice_1192);  mul_168 = slice_1192 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:259 in isoneutral_diffusion_pre, code: sumy += (
        add_78: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_992, mul_169);  slice_992 = mul_169 = None
        slice_scatter_116: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_1194, add_78, 2, 0, -1);  slice_1194 = add_78 = None
        slice_scatter_117: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_1193, slice_scatter_116, 1, 2, -2);  slice_1193 = slice_scatter_116 = None
        slice_scatter_118: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_100, slice_scatter_117, 0, 2, -2);  slice_scatter_100 = slice_scatter_117 = None
        slice_1254: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_118, 0, 2, -2)
        slice_1255: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1254, 1, 2, -2)
        slice_1195: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_118, 0, 2, -2)
        slice_1196: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1195, 1, 2, -2);  slice_1195 = None
        slice_1197: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1196, 2, 0, -1);  slice_1196 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:250 in isoneutral_diffusion_pre, code: facty = cosu[1 + jp : -3 + jp] * dyu[1 + jp : -3 + jp]
        slice_1222: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 2, -2);  arg18_1 = None
        slice_1223: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 2, -2);  arg11_1 = None
        mul_172: "f64[200]" = torch.ops.aten.mul.Tensor(slice_1222, slice_1223);  slice_1222 = slice_1223 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        unsqueeze_50: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
        unsqueeze_51: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 2);  unsqueeze_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:261 in isoneutral_diffusion_pre, code: * K_iso[2:-2, 2:-2, :-1]
        slice_1248: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2);  arg12_1 = None
        slice_1249: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1248, 1, 2, -2);  slice_1248 = None
        slice_1250: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1249, 2, 0, -1);  slice_1249 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_176: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_51, slice_1250);  unsqueeze_51 = slice_1250 = None
        mul_177: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_176, mul_175);  mul_176 = mul_175 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:263 in isoneutral_diffusion_pre, code: * syb**2
        pow_8: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_38, 2);  div_38 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_178: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_177, pow_8);  mul_177 = pow_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:264 in isoneutral_diffusion_pre, code: * maskW[2:-2, 2:-2, :-1]
        slice_1251: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2);  arg5_1 = None
        slice_1252: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1251, 1, 2, -2);  slice_1251 = None
        slice_1253: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1252, 2, 0, -1);  slice_1252 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:260 in isoneutral_diffusion_pre, code: facty[None, :, None]
        mul_179: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_178, slice_1253);  mul_178 = slice_1253 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:259 in isoneutral_diffusion_pre, code: sumy += (
        add_82: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_1197, mul_179);  slice_1197 = mul_179 = None
        slice_scatter_122: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_1255, add_82, 2, 0, -1);  slice_1255 = add_82 = None
        slice_scatter_123: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_1254, slice_scatter_122, 1, 2, -2);  slice_1254 = slice_scatter_122 = None
        slice_scatter_124: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_118, slice_scatter_123, 0, 2, -2);  slice_scatter_118 = slice_scatter_123 = None
        slice_1256: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_124, 0, 2, -2);  slice_scatter_124 = None
        slice_1257: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1256, 1, 2, -2);  slice_1256 = None
        slice_1258: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_1257, 2, 0, -1);  slice_1257 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:269 in isoneutral_diffusion_pre, code: 4 * dyt[None, 2:-2, None] * cost[None, 2:-2, None]
        unsqueeze_54: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg21_1, 0);  arg21_1 = None
        slice_1284: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_54, 1, 2, -2);  unsqueeze_54 = None
        unsqueeze_55: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_1284, 2);  slice_1284 = None
        mul_183: "f64[1, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_55, 4);  unsqueeze_55 = None
        unsqueeze_56: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
        slice_1285: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_56, 1, 2, -2);  unsqueeze_56 = None
        unsqueeze_57: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_1285, 2);  slice_1285 = None
        mul_184: "f64[1, 200, 1]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_57);  mul_183 = unsqueeze_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:268 in isoneutral_diffusion_pre, code: K_33[2:-2, 2:-2, :-1] = sumx / (4 * dxt[2:-2, None, None]) + sumy / (
        div_41: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(slice_1258, mul_184);  slice_1258 = mul_184 = None
        add_83: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(div_40, div_41);  div_40 = div_41 = None
        copy_37: "f64[200, 200, 25]" = torch.ops.aten.copy.default(slice_1288, add_83);  slice_1288 = add_83 = None
        slice_scatter_128: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_1290, copy_37, 2, 0, -1);  slice_1290 = copy_37 = None
        slice_scatter_129: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_1289, slice_scatter_128, 1, 2, -2);  slice_1289 = slice_scatter_128 = None
        slice_scatter_130: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg22_1, slice_scatter_129, 0, 2, -2);  slice_scatter_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:271 in isoneutral_diffusion_pre, code: K_33[2:-2, 2:-2, -1] = 0.0
        slice_1301: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_130, 0, 2, -2)
        slice_1302: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1301, 1, 2, -2)
        slice_1299: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_130, 0, 2, -2)
        slice_1300: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_1299, 1, 2, -2);  slice_1299 = None
        select_141: "f64[200, 200]" = torch.ops.aten.select.int(slice_1300, 2, -1);  slice_1300 = None
        full_default_26: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_38: "f64[200, 200]" = torch.ops.aten.copy.default(select_141, full_default_26);  select_141 = full_default_26 = None
        select_scatter_34: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(slice_1302, copy_38, 2, -1);  slice_1302 = copy_38 = None
        slice_scatter_131: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_1301, select_scatter_34, 1, 2, -2);  slice_1301 = select_scatter_34 = None
        slice_scatter_132: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_130, slice_scatter_131, 0, 2, -2);  slice_scatter_130 = slice_scatter_131 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:165 in isoneutral_diffusion_pre, code: K_11[1:-2, 2:-2, :] = sumz / (4.0 * dzt[None, None, :])
        copy_: "f64[204, 204, 26]" = torch.ops.aten.copy_.default(arg0_1, slice_scatter_42);  arg0_1 = slice_scatter_42 = copy_ = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:164 in isoneutral_diffusion_pre, code: Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * sxe * maskU[1:-2, 2:-2, ki:]
        copy__1: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg13_1, slice_scatter_40);  arg13_1 = slice_scatter_40 = copy__1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:207 in isoneutral_diffusion_pre, code: Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * syn * maskV[2:-2, 1:-2, ki:]
        copy__2: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg15_1, slice_scatter_77);  arg15_1 = slice_scatter_77 = copy__2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:208 in isoneutral_diffusion_pre, code: K_22[2:-2, 1:-2, :] = sumz / (4.0 * dzt[None, None, :])
        copy__3: "f64[204, 204, 26]" = torch.ops.aten.copy_.default(arg16_1, slice_scatter_79);  arg16_1 = slice_scatter_79 = copy__3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:246 in isoneutral_diffusion_pre, code: Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]
        copy__4: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg17_1, slice_scatter_115);  arg17_1 = slice_scatter_115 = copy__4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:266 in isoneutral_diffusion_pre, code: Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * syb * maskW[2:-2, 2:-2, :-1]
        copy__5: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg19_1, slice_scatter_127);  arg19_1 = slice_scatter_127 = copy__5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pyhpc_isoneutral_mixing/isoneutral_pytorch.py:271 in isoneutral_diffusion_pre, code: K_33[2:-2, 2:-2, -1] = 0.0
        copy__6: "f64[204, 204, 26]" = torch.ops.aten.copy_.default(arg22_1, slice_scatter_132);  arg22_1 = slice_scatter_132 = copy__6 = None
        return ()

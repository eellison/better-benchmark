import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[512, 197951]", primals_2: "f32[512]", primals_3: "f32[1024, 197951]", primals_4: "f32[512, 512]", primals_5: "f32[512]", primals_6: "f32[1024, 512]", primals_7: "f32[1024]", primals_8: "f32[512, 1024]", primals_9: "f32[512]", primals_10: "f32[512, 512]", primals_11: "f32[512]", primals_12: "f32[197951, 512]", primals_13: "f32[197951]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute: "f32[197951, 512]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[1024, 512]" = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt: "b8[1024, 512]" = torch.ops.aten.gt.Scalar(addmm, 0)
        mul: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm, 1.0507009873554805)
        mul_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm, 1.0)
        expm1: "f32[1024, 512]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[1024, 512]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[1024, 512]" = torch.ops.aten.addmm.default(primals_5, where, permute_1);  primals_5 = permute_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_1: "b8[1024, 512]" = torch.ops.aten.gt.Scalar(addmm_1, 0)
        mul_3: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_1, 1.0507009873554805)
        mul_4: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_1, 1.0)
        expm1_1: "f32[1024, 512]" = torch.ops.aten.expm1.default(mul_4);  mul_4 = None
        mul_5: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(expm1_1, 1.7580993408473766);  expm1_1 = None
        where_1: "f32[1024, 512]" = torch.ops.aten.where.self(gt_1, mul_3, mul_5);  gt_1 = mul_3 = mul_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_2: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_2: "f32[1024, 1024]" = torch.ops.aten.addmm.default(primals_7, where_1, permute_2);  primals_7 = permute_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_2: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(addmm_2, 0)
        mul_6: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0507009873554805)
        mul_7: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0)
        expm1_2: "f32[1024, 1024]" = torch.ops.aten.expm1.default(mul_7);  mul_7 = None
        mul_8: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(expm1_2, 1.7580993408473766);  expm1_2 = None
        where_2: "f32[1024, 1024]" = torch.ops.aten.where.self(gt_2, mul_6, mul_8);  gt_2 = mul_6 = mul_8 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:136 in encode, code: x = self.drop(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[1024, 1024]" = torch.ops.prims.inductor_random.default([1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_3: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.8);  inductor_random_default = None
        mul_9: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(gt_3, where_2);  where_2 = None
        mul_10: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_9, 5.000000000000001);  mul_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_3: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_3: "f32[1024, 512]" = torch.ops.aten.addmm.default(primals_9, mul_10, permute_3);  primals_9 = permute_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_4: "b8[1024, 512]" = torch.ops.aten.gt.Scalar(addmm_3, 0)
        mul_11: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_3, 1.0507009873554805)
        mul_12: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_3, 1.0)
        expm1_3: "f32[1024, 512]" = torch.ops.aten.expm1.default(mul_12);  mul_12 = None
        mul_13: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(expm1_3, 1.7580993408473766);  expm1_3 = None
        where_3: "f32[1024, 512]" = torch.ops.aten.where.self(gt_4, mul_11, mul_13);  gt_4 = mul_11 = mul_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm_4: "f32[1024, 512]" = torch.ops.aten.addmm.default(primals_11, where_3, permute_4);  primals_11 = permute_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_5: "b8[1024, 512]" = torch.ops.aten.gt.Scalar(addmm_4, 0)
        mul_14: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1.0507009873554805)
        mul_15: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1.0)
        expm1_4: "f32[1024, 512]" = torch.ops.aten.expm1.default(mul_15);  mul_15 = None
        mul_16: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(expm1_4, 1.7580993408473766);  expm1_4 = None
        where_4: "f32[1024, 512]" = torch.ops.aten.where.self(gt_5, mul_14, mul_16);  gt_5 = mul_14 = mul_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_5: "f32[512, 197951]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_5: "f32[1024, 197951]" = torch.ops.aten.addmm.default(primals_13, where_4, permute_5);  primals_13 = permute_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_6: "b8[1024, 197951]" = torch.ops.aten.gt.Scalar(addmm_5, 0)
        mul_17: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1.0507009873554805)
        mul_18: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1.0)
        expm1_5: "f32[1024, 197951]" = torch.ops.aten.expm1.default(mul_18);  mul_18 = None
        mul_19: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(expm1_5, 1.7580993408473766);  expm1_5 = None
        where_5: "f32[1024, 197951]" = torch.ops.aten.where.self(gt_6, mul_17, mul_19);  gt_6 = mul_17 = mul_19 = None
        return (where_5, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, addmm, where, addmm_1, where_1, addmm_2, gt_3, mul_10, addmm_3, where_3, addmm_4, where_4, addmm_5)

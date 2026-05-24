import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[64]", primals_3: "f32[512, 3, 224, 224]", primals_4: "f32[16, 64, 1, 1]", primals_5: "f32[16]", primals_6: "f32[64, 16, 1, 1]", primals_7: "f32[64]", primals_8: "f32[64, 16, 3, 3]", primals_9: "f32[64]", primals_10: "f32[16, 128, 1, 1]", primals_11: "f32[16]", primals_12: "f32[64, 16, 1, 1]", primals_13: "f32[64]", primals_14: "f32[64, 16, 3, 3]", primals_15: "f32[64]", primals_16: "f32[32, 128, 1, 1]", primals_17: "f32[32]", primals_18: "f32[128, 32, 1, 1]", primals_19: "f32[128]", primals_20: "f32[128, 32, 3, 3]", primals_21: "f32[128]", primals_22: "f32[32, 256, 1, 1]", primals_23: "f32[32]", primals_24: "f32[128, 32, 1, 1]", primals_25: "f32[128]", primals_26: "f32[128, 32, 3, 3]", primals_27: "f32[128]", primals_28: "f32[48, 256, 1, 1]", primals_29: "f32[48]", primals_30: "f32[192, 48, 1, 1]", primals_31: "f32[192]", primals_32: "f32[192, 48, 3, 3]", primals_33: "f32[192]", primals_34: "f32[48, 384, 1, 1]", primals_35: "f32[48]", primals_36: "f32[192, 48, 1, 1]", primals_37: "f32[192]", primals_38: "f32[192, 48, 3, 3]", primals_39: "f32[192]", primals_40: "f32[64, 384, 1, 1]", primals_41: "f32[64]", primals_42: "f32[256, 64, 1, 1]", primals_43: "f32[256]", primals_44: "f32[256, 64, 3, 3]", primals_45: "f32[256]", primals_46: "f32[64, 512, 1, 1]", primals_47: "f32[64]", primals_48: "f32[256, 64, 1, 1]", primals_49: "f32[256]", primals_50: "f32[256, 64, 3, 3]", primals_51: "f32[256]", primals_52: "f32[1000, 512, 1, 1]", primals_53: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        convolution: "f32[512, 64, 111, 111]" = torch.ops.aten.convolution.default(primals_3, primals_1, primals_2, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
        relu: "f32[512, 64, 111, 111]" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True)
        getitem: "f32[512, 64, 55, 55]" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[512, 64, 55, 55]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_1: "f32[512, 16, 55, 55]" = torch.ops.aten.convolution.default(getitem, primals_4, primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_5 = None
        relu_1: "f32[512, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_2: "f32[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, primals_6, primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_7 = None
        relu_2: "f32[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f32[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, primals_8, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
        relu_3: "f32[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat: "f32[512, 128, 55, 55]" = torch.ops.aten.cat.default([relu_2, relu_3], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_4: "f32[512, 16, 55, 55]" = torch.ops.aten.convolution.default(cat, primals_10, primals_11, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_11 = None
        relu_4: "f32[512, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_5: "f32[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, primals_12, primals_13, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_13 = None
        relu_5: "f32[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6: "f32[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, primals_14, primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_15 = None
        relu_6: "f32[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_1: "f32[512, 128, 55, 55]" = torch.ops.aten.cat.default([relu_5, relu_6], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_1 = None
        getitem_2: "f32[512, 128, 27, 27]" = _low_memory_max_pool_with_offsets_1[0]
        getitem_3: "i8[512, 128, 27, 27]" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_7: "f32[512, 32, 27, 27]" = torch.ops.aten.convolution.default(getitem_2, primals_16, primals_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_17 = None
        relu_7: "f32[512, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_8: "f32[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, primals_18, primals_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_19 = None
        relu_8: "f32[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "f32[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, primals_20, primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_21 = None
        relu_9: "f32[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_2: "f32[512, 256, 27, 27]" = torch.ops.aten.cat.default([relu_8, relu_9], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_10: "f32[512, 32, 27, 27]" = torch.ops.aten.convolution.default(cat_2, primals_22, primals_23, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_23 = None
        relu_10: "f32[512, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_11: "f32[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, primals_24, primals_25, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_25 = None
        relu_11: "f32[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12: "f32[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, primals_26, primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_27 = None
        relu_12: "f32[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_3: "f32[512, 256, 27, 27]" = torch.ops.aten.cat.default([relu_11, relu_12], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_3 = None
        getitem_4: "f32[512, 256, 13, 13]" = _low_memory_max_pool_with_offsets_2[0]
        getitem_5: "i8[512, 256, 13, 13]" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_13: "f32[512, 48, 13, 13]" = torch.ops.aten.convolution.default(getitem_4, primals_28, primals_29, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_29 = None
        relu_13: "f32[512, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_14: "f32[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, primals_30, primals_31, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_31 = None
        relu_14: "f32[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        convolution_15: "f32[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, primals_32, primals_33, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_33 = None
        relu_15: "f32[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_4: "f32[512, 384, 13, 13]" = torch.ops.aten.cat.default([relu_14, relu_15], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_16: "f32[512, 48, 13, 13]" = torch.ops.aten.convolution.default(cat_4, primals_34, primals_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_35 = None
        relu_16: "f32[512, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_17: "f32[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, primals_36, primals_37, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_37 = None
        relu_17: "f32[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convolution_18: "f32[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, primals_38, primals_39, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_39 = None
        relu_18: "f32[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_5: "f32[512, 384, 13, 13]" = torch.ops.aten.cat.default([relu_17, relu_18], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_19: "f32[512, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_5, primals_40, primals_41, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_41 = None
        relu_19: "f32[512, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_20: "f32[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, primals_42, primals_43, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_43 = None
        relu_20: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        convolution_21: "f32[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, primals_44, primals_45, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_45 = None
        relu_21: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_6: "f32[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_20, relu_21], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_22: "f32[512, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_6, primals_46, primals_47, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_47 = None
        relu_22: "f32[512, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_23: "f32[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, primals_48, primals_49, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_49 = None
        relu_23: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        convolution_24: "f32[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, primals_50, primals_51, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_51 = None
        relu_24: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_7: "f32[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_23, relu_24], 1)

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 512, 13, 13]" = torch.ops.prims.inductor_random.default([512, 512, 13, 13], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[512, 512, 13, 13]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(gt, cat_7);  cat_7 = None
        mul_1: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        convolution_25: "f32[512, 1000, 13, 13]" = torch.ops.aten.convolution.default(mul_1, primals_52, primals_53, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_53 = None
        relu_25: "f32[512, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean: "f32[512, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_25, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        view: "f32[512, 1000]" = torch.ops.aten.reshape.default(mean, [512, 1000]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        le: "b8[512, 1000, 13, 13]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        le_1: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        le_2: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        le_4: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        le_5: "b8[512, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        le_7: "b8[512, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        le_8: "b8[512, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        le_10: "b8[512, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        le_11: "b8[512, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        le_13: "b8[512, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        le_14: "b8[512, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        le_16: "b8[512, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_17: "b8[512, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        le_19: "b8[512, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_20: "b8[512, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        le_22: "b8[512, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_23: "b8[512, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        le_25: "b8[512, 64, 111, 111]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (view, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, getitem, getitem_1, relu_1, cat, relu_4, getitem_2, getitem_3, relu_7, cat_2, relu_10, getitem_4, getitem_5, relu_13, cat_4, relu_16, cat_5, relu_19, cat_6, relu_22, gt, mul_1, le, le_1, le_2, le_4, le_5, le_7, le_8, le_10, le_11, le_13, le_14, le_16, le_17, le_19, le_20, le_22, le_23, le_25)

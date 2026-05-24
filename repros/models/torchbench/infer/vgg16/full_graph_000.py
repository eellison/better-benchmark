import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[64, 3, 3, 3]", arg1_1: "f16[64]", arg2_1: "f16[128, 3, 224, 224]", arg3_1: "f16[64, 64, 3, 3]", arg4_1: "f16[64]", arg5_1: "f16[128, 64, 3, 3]", arg6_1: "f16[128]", arg7_1: "f16[128, 128, 3, 3]", arg8_1: "f16[128]", arg9_1: "f16[256, 128, 3, 3]", arg10_1: "f16[256]", arg11_1: "f16[256, 256, 3, 3]", arg12_1: "f16[256]", arg13_1: "f16[256, 256, 3, 3]", arg14_1: "f16[256]", arg15_1: "f16[512, 256, 3, 3]", arg16_1: "f16[512]", arg17_1: "f16[512, 512, 3, 3]", arg18_1: "f16[512]", arg19_1: "f16[512, 512, 3, 3]", arg20_1: "f16[512]", arg21_1: "f16[512, 512, 3, 3]", arg22_1: "f16[512]", arg23_1: "f16[512, 512, 3, 3]", arg24_1: "f16[512]", arg25_1: "f16[512, 512, 3, 3]", arg26_1: "f16[512]", arg27_1: "f16[4096, 25088]", arg28_1: "f16[4096]", arg29_1: "f16[4096, 4096]", arg30_1: "f16[4096]", arg31_1: "f16[1000, 4096]", arg32_1: "f16[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        convolution: "f16[128, 64, 224, 224]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "f16[128, 64, 224, 224]" = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1: "f16[128, 64, 224, 224]" = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = arg4_1 = None
        relu_1: "f16[128, 64, 224, 224]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem: "f16[128, 64, 112, 112]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_2: "f16[128, 128, 112, 112]" = torch.ops.aten.convolution.default(getitem, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg5_1 = arg6_1 = None
        relu_2: "f16[128, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f16[128, 128, 112, 112]" = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3: "f16[128, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_3 = None
        getitem_2: "f16[128, 128, 56, 56]" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_4: "f16[128, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg9_1 = arg10_1 = None
        relu_4: "f16[128, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5: "f16[128, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg11_1 = arg12_1 = None
        relu_5: "f16[128, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6: "f16[128, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg13_1 = arg14_1 = None
        relu_6: "f16[128, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_6 = None
        getitem_4: "f16[128, 256, 28, 28]" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        convolution_7: "f16[128, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_4, arg15_1, arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg15_1 = arg16_1 = None
        relu_7: "f16[128, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convolution_8: "f16[128, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_7, arg17_1, arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg17_1 = arg18_1 = None
        relu_8: "f16[128, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "f16[128, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_8, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg19_1 = arg20_1 = None
        relu_9: "f16[128, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_9 = None
        getitem_6: "f16[128, 512, 14, 14]" = _low_memory_max_pool_with_offsets_3[0];  _low_memory_max_pool_with_offsets_3 = None
        convolution_10: "f16[128, 512, 14, 14]" = torch.ops.aten.convolution.default(getitem_6, arg21_1, arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg21_1 = arg22_1 = None
        relu_10: "f16[128, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11: "f16[128, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg23_1 = arg24_1 = None
        relu_11: "f16[128, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12: "f16[128, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_11, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg25_1 = arg26_1 = None
        relu_12: "f16[128, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        _low_memory_max_pool_with_offsets_4 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_12 = None
        getitem_8: "f16[128, 512, 7, 7]" = _low_memory_max_pool_with_offsets_4[0];  _low_memory_max_pool_with_offsets_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "f16[128, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7]);  getitem_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        view: "f16[128, 25088]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [128, 25088]);  _adaptive_avg_pool2d = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        permute: "f16[25088, 4096]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm: "f16[128, 4096]" = torch.ops.aten.addmm.default(arg28_1, view, permute);  arg28_1 = view = permute = None
        relu_13: "f16[128, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_1: "f16[128, 4096]" = torch.ops.aten.addmm.default(arg30_1, relu_13, permute_1);  arg30_1 = relu_13 = permute_1 = None
        relu_14: "f16[128, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2: "f16[4096, 1000]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_2: "f16[128, 1000]" = torch.ops.aten.addmm.default(arg32_1, relu_14, permute_2);  arg32_1 = relu_14 = permute_2 = None
        return (addmm_2,)

# SOL Gap Investigation Summary

Baseline source: `/tmp/scratch_space/better_benchmark/sweep_pr184905_baseline.json`
Interleaved source: `/tmp/scratch_space/better_benchmark/sweep_3config_interleaved.json`
Baseline rows: 1482
Rows with best compile / SOL > 1.1x: 1090
Rows with best compile / SOL > 2.0x: 544
Rows with best compile / SOL > 5.0x: 67
Rows with best compile / (SOL + 3us * n_kernels) > 1.1x: 778
Rows with best compile / (SOL + 3us * n_kernels) > 2.0x: 328
Grouped prefixes above 1.1x: 29
Interleaved repros currently covered: 1482

## 3-config winners so far
- combo_looped: 645
- combo_persistent: 569
- default: 268

## Top SOL gaps

- 61.88x `sum_sum_sum_45f02142ecfd`: best=coord_descent 14377.1us, SOL=232.4us, kernels=11, ops=aten.unsqueeze.default:16;aten.mul.Tensor:14;aten.add.Tensor:7;aten.index_put.default:4;aten.sub.Tensor:4;aten.neg.default:3;aten.sum.dim_IntList:3;aten.squeeze
- 36.33x `sum_sum_sum_f90d684d32cb`: best=compiled 4347.0us, SOL=119.6us, kernels=6, ops=aten.unsqueeze.default:16;aten.mul.Tensor:14;aten.add.Tensor:7;aten.index_put.default:4;aten.sub.Tensor:4;aten.neg.default:3;aten.sum.dim_IntList:3;aten.squeeze
- 21.59x `sum_sum_sum_dadf6aa035dd`: best=coord_descent 1370.2us, SOL=63.5us, kernels=6, ops=aten.unsqueeze.default:16;aten.mul.Tensor:14;aten.add.Tensor:7;aten.index_put.default:4;aten.sub.Tensor:4;aten.neg.default:3;aten.sum.dim_IntList:3;aten.squeeze
- 15.16x `sum_adeaebad93f7`: best=coord_descent 2434.0us, SOL=160.6us, kernels=3, ops=aten.gather.default:4;prims.inductor_lookup_seed.default:4;aten.expand.default:3;prims.inductor_randint.default:3;<built-in function getitem>:2;aten.mul.Tensor:
- 14.89x `sum_sum_sum_9e60972dd685`: best=coord_descent 132.0us, SOL=8.9us, kernels=3, ops=aten.mul.Tensor:7;aten.add.Tensor:4;aten.sum.dim_IntList:4;aten.view.default:4;aten.sub.Tensor:3;aten.div.Tensor:1;aten.eq.Scalar:1;aten.full.default:1;aten.ind
- 14.56x `pointwise_531d72f1b34a`: best=coord_descent 548.8us, SOL=37.7us, kernels=116, ops=aten.select.int:463;aten.slice.Tensor:253;aten.mul.Tensor:153;aten.select_scatter.default:150;aten.div.Tensor:75;aten.add.Tensor:72;aten.neg.default:58;aten.sub
- 13.68x `sum_sum_sum_2c925f59efff`: best=coord_descent 477.2us, SOL=34.9us, kernels=6, ops=aten.unsqueeze.default:16;aten.mul.Tensor:14;aten.add.Tensor:7;aten.index_put.default:4;aten.sub.Tensor:4;aten.neg.default:3;aten.sum.dim_IntList:3;aten.squeeze
- 12.62x `pointwise_70c0751eb408`: best=coord_descent 939.8us, SOL=74.5us, kernels=67, ops=aten.slice.Tensor:748;aten.mul.Tensor:184;aten.slice_scatter.default:133;aten.add.Tensor:84;aten.select.int:68;aten.unsqueeze.default:58;aten.div.Tensor:42;aten
- 11.15x `sum_sum_8bcd6e12dcd4`: best=coord_descent 782.4us, SOL=70.2us, kernels=6, ops=aten.view.default:3;aten.slice.Tensor:2;aten.sum.dim_IntList:2;aten.where.self:2;aten.full.default:1;aten.scatter_add.default:1;prims._low_memory_max_pool_offse
- 10.81x `amax_sum_9940b361e5b4`: best=coord_descent 340.9us, SOL=31.5us, kernels=14, ops=aten.slice.Tensor:58;aten.permute.default:28;aten.view.default:27;aten.slice_scatter.default:22;aten.unsqueeze.default:15;aten.full.default:13;aten.copy.default
- 9.74x `sum_sum_sum_95dac16d4328`: best=coord_descent 501.6us, SOL=51.5us, kernels=2, ops=aten.mul.Tensor:18;aten.unsqueeze.default:18;aten.sub.Tensor:6;aten.sum.dim_IntList:4;aten.copy.default:2;aten.add.Tensor:1;aten.clone.default:1;aten.new_empty_
- 9.73x `sum_sum_sum_afd118ca907d`: best=coord_descent 6364.2us, SOL=654.2us, kernels=6, ops=aten.add.Tensor:12;aten.view.default:9;aten.mul.Tensor:6;aten.sum.dim_IntList:4;aten.permute.default:3;aten.unsqueeze.default:3;aten.where.self:3;aten.clone.def
- 9.60x `sum_18262b26934c`: best=coord_descent 1304.5us, SOL=135.8us, kernels=4, ops=aten.view.default:3;aten.full.default:1;aten.scatter_add.default:1;aten.sum.dim_IntList:1;aten.where.self:1;prims._low_memory_max_pool_offsets_to_indices.defaul
- 9.47x `sum_3ee4028cab37`: best=coord_descent 651.1us, SOL=68.7us, kernels=4, ops=aten.view.default:3;aten.full.default:1;aten.scatter_add.default:1;aten.sum.dim_IntList:1;aten.where.self:1;prims._low_memory_max_pool_offsets_to_indices.defaul
- 9.29x `amax_sum_4c524f75213e`: best=coord_descent 867.6us, SOL=93.3us, kernels=16, ops=aten.slice.Tensor:40;aten.slice_scatter.default:22;aten.permute.default:20;aten.view.default:19;aten.copy.default:12;aten.unsqueeze.default:11;aten.full.default
- 9.14x `sum_7df61c52c7f8`: best=coord_descent 637.0us, SOL=69.7us, kernels=4, ops=aten.view.default:3;aten.full.default:1;aten.scatter_add.default:1;aten.sum.dim_IntList:1;aten.where.self:1;prims._low_memory_max_pool_offsets_to_indices.defaul
- 9.09x `pointwise_a14dcfc06344`: best=coord_descent 353.1us, SOL=38.8us, kernels=2, ops=aten.unsqueeze.default:32;aten.mul.Tensor:12;aten.add.Tensor:8;aten.reciprocal.default:4;aten.relu.default:4;aten.sqrt.default:4;aten.sub.Tensor:4;aten.avg_pool
- 8.88x `sum_sum_3219a09ab96a`: best=coord_descent 326.4us, SOL=36.8us, kernels=4, ops=aten.view.default:3;aten.slice.Tensor:2;aten.sum.dim_IntList:2;aten.where.self:2;aten.full.default:1;aten.scatter_add.default:1;prims._low_memory_max_pool_offse
- 8.84x `sum_sum_sum_cd47d7ca4703`: best=coord_descent 153.4us, SOL=17.3us, kernels=5, ops=aten.mul.Tensor:8;aten.full.default:5;aten.sum.dim_IntList:4;aten.index_put.default:3;aten.where.self:3;aten.eq.Scalar:2;aten.sub.Tensor:2;aten.unsqueeze.defaul
- 8.63x `amax_sum_67d7c2666a5c`: best=coord_descent 246.8us, SOL=28.6us, kernels=16, ops=aten.slice.Tensor:40;aten.slice_scatter.default:22;aten.permute.default:20;aten.view.default:19;aten.copy.default:12;aten.unsqueeze.default:11;aten.full.default
- 8.59x `sum_7ee057acd9bc`: best=coord_descent 446.4us, SOL=52.0us, kernels=4, ops=aten.view.default:3;aten.full.default:1;aten.scatter_add.default:1;aten.sum.dim_IntList:1;aten.where.self:1;prims._low_memory_max_pool_offsets_to_indices.defaul
- 8.30x `var_mean_var_mean_6d7a29cb97f1`: best=coord_descent 442.4us, SOL=53.3us, kernels=4, ops=aten.mul.Tensor:14;aten.unsqueeze.default:11;aten.add.Tensor:8;<built-in function getitem>:6;aten.squeeze.dims:5;aten.copy_.default:4;aten.sub.Tensor:3;aten.rel
- 8.14x `amax_sum_7e55413cb344`: best=coord_descent 140.4us, SOL=17.2us, kernels=3, ops=<built-in function getitem>:4;prims.convert_element_type.default:2;aten.amax.default:1;aten.div.Tensor:1;aten.div.Tensor_mode:1;aten.exp.default:1;aten.full.def
- 8.09x `pointwise_a2382a85ee99`: best=coord_descent 83.9us, SOL=10.4us, kernels=1, ops=aten.mul.Tensor:203;aten.add.Tensor:115;aten.sub.Tensor:31;aten.div.Tensor:8;aten.reciprocal.default:6;aten.pow.Tensor_Scalar:5;aten.log.default:2;aten.sqrt.def
- 7.57x `sum_34d3dca078b3`: best=coord_descent 1060.8us, SOL=140.1us, kernels=4, ops=aten.view.default:3;aten.full.default:1;aten.scatter_add.default:1;aten.sum.dim_IntList:1;aten.where.self:1;prims._low_memory_max_pool_offsets_to_indices.defaul

## Largest grouped families

- var_mean: count=224, max=7.52x, geomean=1.67x, reps=var_mean_765fb8f2c85e;var_mean_f0d7c08a0622;var_mean_598830735cf6;var_mean_6d9f14e83fbb;var_mean_21efaf61742a;var_mean_2f7685c1567b;var_mean_83a48287f334;var_mean_cf2474e318ae;var_mean_0a0eec8ae477;var_mean_b0e2cade591e;var_mean_6f5e73770101;var_mean_5b0c15caa439
- sum_sum_sum: count=205, max=61.88x, geomean=2.84x, reps=sum_sum_sum_45f02142ecfd;sum_sum_sum_f90d684d32cb;sum_sum_sum_dadf6aa035dd;sum_sum_sum_9e60972dd685;sum_sum_sum_2c925f59efff;sum_sum_sum_95dac16d4328;sum_sum_sum_afd118ca907d;sum_sum_sum_cd47d7ca4703;sum_sum_sum_82a3f0084247;sum_sum_sum_72d5ffa26b42;sum_sum_sum_6a2ad206248c;sum_sum_sum_8e6cde42e572
- pointwise: count=178, max=14.56x, geomean=1.92x, reps=pointwise_531d72f1b34a;pointwise_70c0751eb408;pointwise_a14dcfc06344;pointwise_a2382a85ee99;pointwise_f9c1d1b08ddb;pointwise_6d842b54b40d;pointwise_13f26d420251;pointwise_437415a3398d;pointwise_cf3acd87ba9e;pointwise_55a3cc897c83;pointwise_71e3a6c09140;pointwise_f2df04089ff4
- sum_sum: count=135, max=11.15x, geomean=2.27x, reps=sum_sum_8bcd6e12dcd4;sum_sum_3219a09ab96a;sum_sum_6a14a9c9ba88;sum_sum_e9338369070e;sum_sum_a7c7ab7bfcf5;sum_sum_0930bd38b7d1;sum_sum_67d7103962e7;sum_sum_d73ef74614dc;sum_sum_4bd81dea302d;sum_sum_3c44d89daf3a;sum_sum_addc448a9236;sum_sum_ced249279c9d
- sum: count=109, max=15.16x, geomean=1.96x, reps=sum_adeaebad93f7;sum_18262b26934c;sum_3ee4028cab37;sum_7df61c52c7f8;sum_7ee057acd9bc;sum_34d3dca078b3;sum_bc4a942a8c4f;sum_14fe7b321763;sum_8a66186d1ffe;sum_617cd87647d6;sum_5a4992885bd6;sum_0becf9609ad7
- mean: count=51, max=4.45x, geomean=1.85x, reps=mean_5c93e9826aa8;mean_d0fc206717a8;mean_f7170c220032;mean_b22538109e2c;mean_148ad2fc17ca;mean_e9b5d856a37d;mean_eccc5dff9f55;mean_b1d8f8a4704b;mean_f362a8f86f2e;mean_7874ce26c7f3;mean_cf12f53df7dd;mean_feaf8e3a5f0b
- amax_sum: count=38, max=10.81x, geomean=3.09x, reps=amax_sum_9940b361e5b4;amax_sum_4c524f75213e;amax_sum_67d7c2666a5c;amax_sum_7e55413cb344;amax_sum_5f0c26b7e967;amax_sum_68fe981b18dd;amax_sum_87e1fb077f24;amax_sum_4bd27b112605;amax_sum_93cb2fd0355b;amax_sum_35490be2986b;amax_sum_0f87f6568d0f;amax_sum_f5253e4f250e
- amax_sum_sum: count=30, max=5.22x, geomean=2.17x, reps=amax_sum_sum_dc96a87ba8db;amax_sum_sum_9e768dff4978;amax_sum_sum_1bad0f362efd;amax_sum_sum_8b606a2377c7;amax_sum_sum_e2f518f0a274;amax_sum_sum_b9ac8700504c;amax_sum_sum_6fd07d12d98a;amax_sum_sum_86d05d6810f4;amax_sum_sum_31600750c3c4;amax_sum_sum_d85e67b00643;amax_sum_sum_f9833ae619a1;amax_sum_sum_0e37ca9164b3
- var_mean_mean: count=29, max=3.80x, geomean=2.53x, reps=var_mean_mean_1e867cdb78f5;var_mean_mean_0805988a80e9;var_mean_mean_8d6fc761298a;var_mean_mean_e2206aa32217;var_mean_mean_254b25297850;var_mean_mean_6b2793eaa579;var_mean_mean_7a446f7d0ed8;var_mean_mean_ebf25f51023b;var_mean_mean_0593cda589d5;var_mean_mean_f6d34778b88d;var_mean_mean_2fedd7a79265;var_mean_mean_3480e8831bac
- amax_sum_any: count=21, max=3.74x, geomean=2.56x, reps=amax_sum_any_24355de03713;amax_sum_any_7c109c56b046;amax_sum_any_a6daa5fac790;amax_sum_any_d16c257aa03e;amax_sum_any_64da678446f9;amax_sum_any_fd11cc31d3d3;amax_sum_any_5686b4ede363;amax_sum_any_e20fbe84102b;amax_sum_any_0986fbf8aadc;amax_sum_any_29e1adb86456;amax_sum_any_3e6e6f246ff6;amax_sum_any_e1163072ce50
- var_mean_var_mean: count=12, max=8.30x, geomean=2.57x, reps=var_mean_var_mean_6d7a29cb97f1;var_mean_var_mean_40a0055bb26e;var_mean_var_mean_73e1a842318f;var_mean_var_mean_aabd7680d8e0;var_mean_var_mean_0795d88a1e39;var_mean_var_mean_7eb8a9047c3e;var_mean_var_mean_8c8eb33a711a;var_mean_var_mean_ca75e017814e;var_mean_var_mean_de5c360021d3;var_mean_var_mean_98a1510ddc30;var_mean_var_mean_2989608c4b10;var_mean_var_mean_99567a16bff5
- any_amax_sum: count=9, max=2.00x, geomean=1.27x, reps=any_amax_sum_2da4fabb8348;any_amax_sum_56d491110e60;any_amax_sum_c1a313554420;any_amax_sum_727a4ae37fb6;any_amax_sum_70169d16c71e;any_amax_sum_c34514991dfb;any_amax_sum_0815d0a11243;any_amax_sum_97b43144672a;any_amax_sum_4a963b47b28d
- mean_var_mean: count=8, max=1.62x, geomean=1.32x, reps=mean_var_mean_3c4ea6d8f342;mean_var_mean_9d590f58ecc9;mean_var_mean_c9089e261699;mean_var_mean_cf650837b7b1;mean_var_mean_e6c3cd7f84ea;mean_var_mean_d6e91c72f94d;mean_var_mean_f69b8a5457fd;mean_var_mean_9f6d83adf6c4
- var_mean_var_mean_var_mean: count=6, max=5.67x, geomean=4.34x, reps=var_mean_var_mean_var_mean_0407b3e7c77f;var_mean_var_mean_var_mean_9267280d8024;var_mean_var_mean_var_mean_97062ec87370;var_mean_var_mean_var_mean_850152d806da;var_mean_var_mean_var_mean_5d12643e9b78;var_mean_var_mean_var_mean_850dfa63ec81
- mean_mean: count=5, max=3.10x, geomean=2.36x, reps=mean_mean_50d3de630629;mean_mean_1b98d81214e6;mean_mean_fbdbd1645f6a;mean_mean_6cb471f41ecf;mean_mean_fac851fa5f98
- any_mean: count=5, max=1.71x, geomean=1.61x, reps=any_mean_fde44ec669a9;any_mean_38ea597e5c87;any_mean_6b05ef27be02;any_mean_33b10ff8a837;any_mean_417d9efb98a0
- sum_mean: count=4, max=3.27x, geomean=2.31x, reps=sum_mean_39228c93d6a4;sum_mean_273a847a963c;sum_mean_be3cca3d51c6;sum_mean_6ea3cb4a159b
- mean_var: count=4, max=2.87x, geomean=2.00x, reps=mean_var_5f74618c3485;mean_var_e46ebb5a28fe;mean_var_d333e54cfcb6;mean_var_2a871bf52561
- amax_sum_amax: count=3, max=5.11x, geomean=4.61x, reps=amax_sum_amax_2a81770def44;amax_sum_amax_68fa105ccaf0;amax_sum_amax_9faf4a02126a
- var_mean_var_mean_mean: count=3, max=2.99x, geomean=2.84x, reps=var_mean_var_mean_mean_e12f05530897;var_mean_var_mean_mean_93065d5c677b;var_mean_var_mean_mean_b49a8ed53342
- argmax_mean: count=2, max=4.60x, geomean=3.22x, reps=argmax_mean_9cf9e9271ff1;argmax_mean_180a59791f51
- sum_sum_mean: count=2, max=2.53x, geomean=2.52x, reps=sum_sum_mean_1dfbbe76c078;sum_sum_mean_9af96955f8cc
- var_mean_any: count=1, max=2.57x, geomean=2.57x, reps=var_mean_any_e86b9ea56684
- var_mean_sum_mean: count=1, max=2.19x, geomean=2.19x, reps=var_mean_sum_mean_c177cfce7234
- var_mean_sum: count=1, max=1.71x, geomean=1.71x, reps=var_mean_sum_87c8c4a61b33

## Top launch-adjusted priorities

- score=83983.4, launch_gap=54.18x, excess=14111.7us, raw=61.88x, kernels=11 `sum_sum_sum_45f02142ecfd`
- score=21817.8, launch_gap=31.58x, excess=4209.4us, raw=36.33x, kernels=6 `sum_sum_sum_f90d684d32cb`
- score=18682.0, launch_gap=9.47x, excess=5692.0us, raw=9.73x, kernels=6 `sum_sum_sum_afd118ca907d`
- score=12274.3, launch_gap=4.38x, excess=5721.5us, raw=4.42x, kernels=6 `sum_sum_sum_7b24a3457260`
- score=8880.5, launch_gap=14.35x, excess=2264.4us, raw=15.16x, kernels=3 `sum_adeaebad93f7`
- score=6511.3, launch_gap=6.30x, excess=2380.5us, raw=6.66x, kernels=8 `sum_sum_sum_8e6cde42e572`
- score=6266.8, launch_gap=5.17x, excess=2627.7us, raw=5.22x, kernels=2 `amax_sum_sum_dc96a87ba8db`
- score=5712.2, launch_gap=16.82x, excess=1288.7us, raw=21.59x, kernels=6 `sum_sum_sum_dadf6aa035dd`
- score=3774.7, launch_gap=8.82x, excess=1156.6us, raw=9.60x, kernels=4 `sum_18262b26934c`
- score=3752.1, launch_gap=3.93x, excess=1853.9us, raw=4.07x, kernels=7 `sum_sum_cdaed89f373c`
- score=3731.9, launch_gap=6.10x, excess=1359.6us, raw=6.70x, kernels=8 `sum_sum_6a14a9c9ba88`
- score=3351.4, launch_gap=2.80x, excess=2247.9us, raw=2.81x, kernels=1 `amax_sum_3ed297ef02cd`
- score=2985.4, launch_gap=3.56x, excess=1619.2us, raw=3.59x, kernels=2 `amax_sum_sum_1bad0f362efd`
- score=2922.9, launch_gap=3.54x, excess=1596.6us, raw=3.56x, kernels=1 `amax_sum_f0661488d68c`
- score=2706.5, launch_gap=5.14x, excess=1090.4us, raw=5.59x, kernels=7 `sum_sum_a7c7ab7bfcf5`
- score=2654.0, launch_gap=6.97x, excess=908.7us, raw=7.57x, kernels=4 `sum_34d3dca078b3`
- score=2429.3, launch_gap=3.41x, excess=664.3us, raw=12.62x, kernels=67 `pointwise_70c0751eb408`
- score=2415.3, launch_gap=8.87x, excess=694.3us, raw=11.15x, kernels=6 `sum_sum_8bcd6e12dcd4`
- score=2389.8, launch_gap=5.02x, excess=964.1us, raw=5.57x, kernels=8 `sum_sum_sum_1627b1a3a6f6`
- score=2351.1, launch_gap=4.69x, excess=986.2us, raw=5.22x, kernels=9 `sum_sum_0930bd38b7d1`
- score=2336.0, launch_gap=6.14x, excess=726.3us, raw=9.29x, kernels=16 `amax_sum_4c524f75213e`
- score=2294.2, launch_gap=4.99x, excess=974.8us, raw=5.11x, kernels=2 `amax_sum_amax_2a81770def44`
- score=2246.8, launch_gap=3.02x, excess=1398.8us, raw=3.04x, kernels=2 `sum_b691b8dad90a`
- score=2180.3, launch_gap=2.39x, excess=1733.6us, raw=2.39x, kernels=1 `sum_e00c7291b6ee`
- score=2043.3, launch_gap=5.98x, excess=757.1us, raw=6.49x, kernels=4 `var_mean_f0d7c08a0622`

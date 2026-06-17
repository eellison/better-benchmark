# Matmul partitioning views

Built on the **existing** `get_fusion_partitions` (default partitioner, unchanged). Matmuls are non-fusible extern ops, so their pointwise neighbors sit in separate partitions; this view re-attaches them.

## Totals

- Graphs analyzed: **343** (load failures: 21)
- Matmuls seen: **17575**
- Matmuls with a strictly-pointwise epilogue: **8566** (clean = every consumer is pointwise: **7943**)
- `mm -> pointwise -> mm` chains (SEQUENTIAL, >=1 pointwise op between): **6290**
- `mm -> mm` with only views between (no pointwise): **380**
- **>=2 matmuls converging to ONE output** (fan-in / reduce: `mm(a,b) (+|*) mm(c,d)`): **3230** (through a strictly-pointwise bridge: **248**)
  - of which **memory-eliminating** (every contributing matmul's ONLY consumer is the bridge -> intermediate never hits DRAM): **198**, eliminating **10755.5 MB** of matmul-output reads

## Matmul op breakdown (epilogue hits)

- `mm`: 4606
- `addmm`: 2651
- `bmm`: 1309

## Top epilogue op-signatures

| count | epilogue ops (distinct, sorted) |
|------:|---------------------------------|
| 2963 | `convert_element_type` |
| 2669 | `clone` |
| 603 | `add, convert_element_type, erf, mul` |
| 484 | `clone, mul` |
| 288 | `clone, convert_element_type` |
| 192 | `add, mul` |
| 192 | `add, convert_element_type, mul` |
| 190 | `add, mul, pow, tanh` |
| 162 | `relu` |
| 134 | `add, convert_element_type, mul, pow, tanh` |
| 120 | `add, convert_element_type, div, exp, mul, neg` |
| 120 | `add, convert_element_type, mul, tanh` |
| 112 | `add, convert_element_type` |
| 98 | `le, relu` |
| 30 | `add` |
| 27 | `add, convert_element_type, mul, pow, sub, tanh` |
| 23 | `clone, div` |
| 22 | `convert_element_type, mul, where` |
| 21 | `mul` |
| 18 | `add, convert_element_type, div, exp, neg` |
| 12 | `add, clone, div` |
| 12 | `add, clone` |
| 12 | `add, clone, mul` |
| 12 | `add, clone, convert_element_type, div` |
| 12 | `add, clone, convert_element_type` |

## Top `mm -> pointwise -> mm` bridge signatures

| count | bridge pointwise ops |
|------:|----------------------|
| 3096 | `clone` |
| 603 | `add, convert_element_type, erf, mul` |
| 502 | `clone, mul` |
| 384 | `clone, convert_element_type` |
| 262 | `add, mul` |
| 262 | `add, convert_element_type, mul` |
| 190 | `add, mul, pow, tanh` |
| 162 | `relu` |
| 134 | `add, convert_element_type, mul, pow, tanh` |
| 120 | `add, convert_element_type, div, exp, mul, neg` |
| 120 | `add, convert_element_type, mul, tanh` |
| 98 | `le, relu` |
| 86 | `add, convert_element_type, mul, pow, sub, tanh` |
| 50 | `add` |
| 48 | `add, convert_element_type` |
| 44 | `convert_element_type, mul, where` |
| 23 | `clone, div` |
| 18 | `add, convert_element_type, div, exp, neg` |
| 12 | `add, clone, div` |
| 12 | `add, clone` |
| 12 | `add, clone, mul` |
| 12 | `add, clone, convert_element_type, div` |
| 12 | `add, clone, convert_element_type` |
| 12 | `add, clone, convert_element_type, mul` |
| 9 | `convert_element_type, expm1, gt, mul, where` |

## Matmul fan-in (>=2 matmuls -> 1 output)

The 'two matmuls reduce to one output' pattern: independent sibling matmuls combined through a pointwise bridge into a single value (`mm(a,b)+mm(c,d)` residual/parallel-proj; `silu(mm(x,Wg))*mm(x,Wu)` gated MLP). Distinct from the sequential chain above.

**The memory win requires EXCLUSIVITY.** Fusing the combine into the GEMM epilogue only eliminates a DRAM read when each contributing matmul's output feeds *nothing but* the bridge. If a matmul output is reused elsewhere (or escapes to a graph output) it must be materialized regardless, so that matmul is not counted as eliminable. `all_exclusive` rows are the clean targets.

**Combine op** (the pointwise op joining the matmuls) -- all strict-pw vs memory-eliminating subset:

| combine | strict-pw | memory-eliminating |
|---------|----------:|-------------------:|
| `mul` | 136 | 136 |
| `add` | 112 | 62 |

**Arity** (how many matmuls converge, strict-pw):

- 2 matmuls: 238
- 3 matmuls: 4
- 12 matmuls: 6

**Top memory-eliminating fan-in bridge signatures:**

| count | bridge ops |
|------:|------------|
| 60 | `add, convert_element_type, div, exp, mul, neg` |
| 60 | `add, convert_element_type, mul, tanh` |
| 48 | `add, mul` |
| 16 | `add, mul, pow, tanh` |
| 12 | `add, convert_element_type` |
| 2 | `add, div` |

## Example clean epilogues (one per model)

- `AlbertForMaskedLM` / full_graph_000: `addmm` [4096, 4096] -> epilogue `mul+clone`
- `AllenaiLongformerBase` / full_graph_002: `mm` [8192, 768] -> epilogue `add+div+clone`
- `BartForCausalLM` / full_graph_000: `addmm` [8192, 4096] -> epilogue `convert_element_type+mul+mul+erf+add+mul+convert_element_type`
- `BertForMaskedLM` / full_graph_000: `addmm` [16384, 768] -> epilogue `mul+clone`
- `BlenderbotForCausalLM` / full_graph_000: `addmm` [4096, 10240] -> epilogue `convert_element_type+mul+mul+erf+add+mul+convert_element_type`
- `BlenderbotForConditionalGeneration` / full_graph_000: `addmm` [2048, 2560] -> epilogue `mul+clone`
- `DebertaV2ForMaskedLM` / full_graph_000: `addmm` [4096, 1536] -> epilogue `clone`
- `DistilBertForMaskedLM` / full_graph_000: `addmm` [32768, 768] -> epilogue `mul+clone`
- `DistillGPT2` / full_graph_000: `addmm` [16384, 3072] -> epilogue `mul+pow+mul+add+mul+tanh+add+mul`
- `ElectraForCausalLM` / full_graph_000: `addmm` [32768, 256] -> epilogue `mul+clone`
- `GPT2ForSequenceClassification` / full_graph_000: `addmm` [8192, 3072] -> epilogue `mul+pow+mul+add+mul+tanh+add+mul`
- `GPTJForCausalLM` / full_graph_000: `bmm` [16, 128, 256] -> epilogue `clone`
- `GPTJForQuestionAnswering` / full_graph_000: `bmm` [16, 128, 256] -> epilogue `clone`
- `GPTNeoForCausalLM` / full_graph_000: `mm` [4096, 2048] -> epilogue `convert_element_type+clone`
- `GPTNeoForSequenceClassification` / full_graph_000: `mm` [4096, 2048] -> epilogue `convert_element_type+clone`
- `GoogleFnet` / full_graph_000: `addmm` [16384, 3072] -> epilogue `mul+pow+mul+add+mul+tanh+add+mul`
- `LayoutLMForMaskedLM` / full_graph_000: `addmm` [16384, 768] -> epilogue `clone`
- `M2M100ForConditionalGeneration` / full_graph_000: `addmm` [8192, 1024] -> epilogue `mul+clone`
- `MBartForCausalLM` / full_graph_000: `addmm` [8192, 4096] -> epilogue `convert_element_type+mul+mul+erf+add+mul+convert_element_type`
- `MT5ForConditionalGeneration` / full_graph_000: `mm` [4096, 384] -> epilogue `clone`
- `MegatronBertForCausalLM` / full_graph_000: `addmm` [8192, 4096] -> epilogue `convert_element_type+mul+mul+erf+add+mul+convert_element_type`
- `MobileBertForMaskedLM` / full_graph_000: `addmm` [32768, 128] -> epilogue `mul+add`
- `OPTForCausalLM` / full_graph_000: `addmm` [8192, 768] -> epilogue `mul`
- `PLBartForCausalLM` / full_graph_000: `addmm` [16384, 3072] -> epilogue `convert_element_type+mul+mul+erf+add+mul+convert_element_type`
- `PegasusForCausalLM` / full_graph_000: `addmm` [16384, 4096] -> epilogue `convert_element_type+mul+mul+erf+add+mul+convert_element_type`

## Example mm -> pointwise -> mm chains (one per model)

- `AlbertForMaskedLM` / full_graph_000: `addmm` [4096, 4096] -> `mul+clone` -> `bmm` [512, 512, 512]  (bridge in corpus: a18f3869022f)
- `AllenaiLongformerBase` / full_graph_002: `mm` [8192, 768] -> `add+div+clone` -> `bmm` [288, 512, 512]  (bridge in corpus: 2b567494bf14)
- `BartForCausalLM` / full_graph_000: `addmm` [8192, 4096] -> `convert_element_type+mul+mul+erf+add+mul+convert_element_type` -> `addmm` [8192, 1024]  (bridge in corpus: 826ea0b5f6e4)
- `BertForMaskedLM` / full_graph_000: `addmm` [16384, 768] -> `mul+clone` -> `bmm` [384, 512, 512]  (bridge in corpus: a18f3869022f)
- `BlenderbotForCausalLM` / full_graph_000: `addmm` [4096, 10240] -> `convert_element_type+mul+mul+erf+add+mul+convert_element_type` -> `addmm` [4096, 2560]  (bridge in corpus: 826ea0b5f6e4)
- `BlenderbotForConditionalGeneration` / full_graph_000: `addmm` [2048, 2560] -> `mul+clone` -> `bmm` [512, 128, 128]  (bridge in corpus: a18f3869022f)
- `DebertaV2ForMaskedLM` / full_graph_000: `addmm` [4096, 1536] -> `clone` -> `bmm` [192, 512, 512]  (bridge in corpus: c0d19d490a2f)
- `DistilBertForMaskedLM` / full_graph_000: `addmm` [32768, 768] -> `mul+clone` -> `bmm` [3072, 128, 128]  (bridge in corpus: a18f3869022f)
- `DistillGPT2` / full_graph_000: `addmm` [16384, 3072] -> `mul+pow+mul+add+mul+tanh+add+mul` -> `addmm` [16384, 768]  (bridge in corpus: 2d969ac9ad59)
- `ElectraForCausalLM` / full_graph_000: `addmm` [32768, 256] -> `mul+clone` -> `bmm` [256, 512, 512]  (bridge in corpus: a18f3869022f)
- `GPT2ForSequenceClassification` / full_graph_000: `addmm` [8192, 3072] -> `mul+pow+mul+add+mul+tanh+add+mul` -> `addmm` [8192, 768]  (bridge in corpus: 2d969ac9ad59)
- `GPTJForCausalLM` / full_graph_000: `bmm` [16, 128, 256] -> `clone` -> `mm` [128, 4096]  (bridge in corpus: e87d6ebc9ded)
- `GPTJForQuestionAnswering` / full_graph_000: `bmm` [16, 128, 256] -> `clone` -> `mm` [128, 4096]  (bridge in corpus: e87d6ebc9ded)
- `GPTNeoForCausalLM` / full_graph_000: `mm` [4096, 2048] -> `convert_element_type+clone` -> `bmm` [512, 128, 128]  (bridge in corpus: 03afed5ec8e9)
- `GPTNeoForSequenceClassification` / full_graph_000: `mm` [4096, 2048] -> `convert_element_type+clone` -> `bmm` [512, 128, 128]  (bridge in corpus: 03afed5ec8e9)
- `GoogleFnet` / full_graph_000: `addmm` [16384, 3072] -> `mul+pow+mul+add+mul+tanh+add+mul` -> `addmm` [16384, 768]  (bridge in corpus: 2d969ac9ad59)
- `LayoutLMForMaskedLM` / full_graph_000: `addmm` [16384, 3072] -> `convert_element_type+mul+mul+erf+add+mul+convert_element_type` -> `addmm` [16384, 768]  (bridge in corpus: 826ea0b5f6e4)
- `M2M100ForConditionalGeneration` / full_graph_000: `addmm` [8192, 1024] -> `mul+clone` -> `bmm` [1024, 128, 128]  (bridge in corpus: a18f3869022f)
- `MBartForCausalLM` / full_graph_000: `addmm` [8192, 4096] -> `convert_element_type+mul+mul+erf+add+mul+convert_element_type` -> `addmm` [8192, 1024]  (bridge in corpus: 826ea0b5f6e4)
- `MT5ForConditionalGeneration` / full_graph_000: `mm` [4096, 384] -> `clone` -> `bmm` [192, 128, 128]  (bridge in corpus: 301cf5a13527)
- `MegatronBertForCausalLM` / full_graph_000: `addmm` [8192, 4096] -> `convert_element_type+mul+mul+erf+add+mul+convert_element_type` -> `addmm` [8192, 1024]  (bridge in corpus: 826ea0b5f6e4)
- `MobileBertForMaskedLM` / full_graph_000: `addmm` [32768, 128] -> `mul+add` -> `addmm` [32768, 128]  (bridge in corpus: b2b66f70d626)
- `OPTForCausalLM` / full_graph_000: `addmm` [8192, 3072] -> `relu` -> `addmm` [8192, 768]  (bridge in corpus: 3abc926270f6)
- `PLBartForCausalLM` / full_graph_000: `addmm` [16384, 3072] -> `convert_element_type+mul+mul+erf+add+mul+convert_element_type` -> `addmm` [16384, 768]  (bridge in corpus: 826ea0b5f6e4)
- `PegasusForCausalLM` / full_graph_000: `addmm` [16384, 4096] -> `convert_element_type+mul+mul+erf+add+mul+convert_element_type` -> `addmm` [16384, 1024]  (bridge in corpus: 826ea0b5f6e4)

## Example memory-eliminating fan-in (one per model)

(`all_exclusive` -- every contributing matmul feeds only the bridge, so its output never hits DRAM.)

- `MT5ForConditionalGeneration` / full_graph_000: 2x `mm+mm` ([4096, 1024] , [4096, 1024]) `mul`-combined via `mul+pow+mul+add+mul+tanh+add+mul+mul` -> [4096, 1024], elim=16777.2KB  (bridge in corpus: a07e50c2fb69)
- `MobileBertForMaskedLM` / full_graph_000: 2x `addmm+addmm` ([32768, 128] , [32768, 128]) `add`-combined via `mul+add+add+mul+add` -> [256, 128, 128], elim=16777.2KB  (bridge in corpus: 5c3762c611b3)
- `Qwen3-0.6B` / full_graph_000: 2x `mm+mm` ([1000, 3072] , [1000, 3072]) `mul`-combined via `convert_element_type+neg+exp+add+div+convert_element_type+mul` -> [1000, 3072], elim=12288.0KB  (bridge in corpus: 7c39bacfff54)
- `gemma-2-2b` / full_graph_000: 2x `mm+mm` ([1000, 9216] , [1000, 9216]) `mul`-combined via `convert_element_type+mul+mul+mul+mul+add+mul+tanh+add+mul+convert_element_type+mul` -> [1000, 9216], elim=36864.0KB  (bridge in corpus: aa77af8fcd54)
- `gemma-3-4b-it` / full_graph_000: 2x `mm+mm` ([1000, 10240] , [1000, 10240]) `mul`-combined via `convert_element_type+mul+mul+mul+mul+add+mul+tanh+add+mul+convert_element_type+mul` -> [1000, 10240], elim=40960.0KB  (bridge in corpus: aa77af8fcd54)
- `Mistral-7B-Instruct-v0.3` / full_graph_000: 2x `mm+mm` ([1000, 14336] , [1000, 14336]) `mul`-combined via `convert_element_type+neg+exp+add+div+convert_element_type+mul` -> [1000, 14336], elim=57344.0KB  (bridge in corpus: 7c39bacfff54)
- `AlbertForMaskedLM` / full_graph_001: 12x `mm+mm+mm+mm+mm+mm+mm+mm+mm+mm+mm+mm` ([4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096] , [4096, 4096]) `add`-combined via `convert_element_type+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add+convert_element_type+add` -> [4096, 4096], elim=402653.2KB  (bridge in corpus: b09beca3a7bd)
- `AllenaiLongformerBase` / full_graph_005: 3x `mm+mm+mm` ([8192, 768] , [8192, 768] , [8192, 768]) `add`-combined via `convert_element_type+convert_element_type+add+convert_element_type+add+add` -> [8, 1024, 768], elim=37748.7KB  (bridge in corpus: f2ae34e9b8f1)
- `BartForCausalLM` / full_graph_007: 3x `mm+mm+mm` ([8192, 1024] , [8192, 1024] , [8192, 1024]) `add`-combined via `convert_element_type+add+convert_element_type+add+convert_element_type+add` -> [8, 1024, 1024], elim=50331.6KB  (bridge in corpus: ee05f315ed84)
- `BlenderbotForConditionalGeneration` / full_graph_010: 2x `mm+mm` ([2048, 2560] , [2048, 2560]) `add`-combined via `convert_element_type+convert_element_type+add` -> [16, 128, 2560], elim=20971.5KB  (bridge in corpus: 514cc391d8e4)
- `M2M100ForConditionalGeneration` / full_graph_009: 2x `mm+mm` ([8192, 1024] , [8192, 1024]) `add`-combined via `convert_element_type+convert_element_type+add` -> [64, 128, 1024], elim=33554.4KB  (bridge in corpus: 514cc391d8e4)
- `PLBartForCausalLM` / full_graph_007: 3x `mm+mm+mm` ([16384, 768] , [16384, 768] , [16384, 768]) `add`-combined via `convert_element_type+add+convert_element_type+add+convert_element_type+add` -> [16, 1024, 768], elim=75497.5KB  (bridge in corpus: ee05f315ed84)
- `TrOCRForCausalLM` / full_graph_006: 3x `mm+mm+mm` ([16384, 1024] , [16384, 1024] , [16384, 1024]) `add`-combined via `convert_element_type+add+convert_element_type+add+convert_element_type+add` -> [64, 256, 1024], elim=100663.3KB  (bridge in corpus: ee05f315ed84)
- `deit_base_distilled_patch16_224` / full_graph_000: 2x `addmm+addmm` ([128, 1000] , [128, 1000]) `add`-combined via `add+div` -> [128, 1000], elim=512.0KB  (bridge in corpus: 7314654d3afc)

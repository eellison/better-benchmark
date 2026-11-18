# Gemma 4 31B text inference

Identity: `hf/infer/Gemma4_31B_text_6l_prefill_b1_s2048_sdpa_bf16`.

Captured from the local Hugging Face `Gemma4ForCausalLM` implementation using
`Gemma4TextConfig`. The pinned `google/gemma-4-31B` config revision is
`5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`; original config SHA-256:
`6a81841cad2b6ba06841e23c6afb5f0a27827bc12c64328ffa6338831a21267e`.

The workload retains original layers 0–5 of 60: five sliding-attention layers
and one full-attention layer. Depth was reduced before model allocation. All
other released text geometry was preserved: hidden size 5376, intermediate
size 21504, vocabulary 262144, 32 query heads, sliding head dimension 256 with
16 KV heads, global head dimension 512 with 4 KV heads, and sliding window
1024. The global layer reuses its K projection for V. Embedding and output
weights are tied. This variant has no MoE, cross-layer KV sharing, or
per-layer embeddings.

Random config-initialized weights and legal integer tokens use seed 0.
Parameters are BF16; explicit internal FP32 operations and FP32 RoPE buffers
are preserved. Execution uses batch 1, sequence 2048, eval/no-grad,
`use_cache=False`, no past KV, and `logits_to_keep=1`.
`torch.compile(fullgraph=True, dynamic=False)` completed with finite BF16
logits shaped `[1, 1, 262144]`. The reduced model has
4,338,096,896 unique parameters (8,676,193,792 bytes).

HF attention implementation is `sdpa`, with native/default backend dispatch.
All six attention calls resolved to
`aten._scaled_dot_product_efficient_attention.default` on the capture GPU.
Flash, efficient, and math SDPA backends were enabled; no backend was forced.

Capture used `install_capture_hook(capture_only=False,
constant_folding=False)` before the first compiled invocation, uninstalled
in `finally`, and merged only after successful output checks using
`temporary_capture_for_merge(..., suite="hf", mode="infer")`. The temporary
raw capture was removed on context exit.

The corpus contains [one full post-grad graph](full_graph_000.py), its
[input/output constraints and round-trip stamp](full_graph_000.meta.json),
and a [pattern manifest](manifest.json). Fourteen pattern/shape points
represent 37 partition occurrences: ten reduction patterns and four
pointwise patterns. Twelve canonical patterns are new; two reuse existing
Gemma patterns with additional shape points. The graph records 963 function
calls: 847 in fusible partitions, 66 unpartitioned transparent/fusible nodes,
and 50 external calls (42 matrix multiplies, six efficient-attention calls,
and two cumulative sums).

| Canonical pattern | Shape hash | Occurrences | Corpus change |
|---|---|---:|---|
| [pointwise_aa77af8fcd54](../../../../canonical/pointwise_aa77af8fcd54/repro.py) | a50c1259 | 6 | Reused |
| [mean_mean_d3c8742cb719](../../../../canonical/mean_mean_d3c8742cb719/repro.py) | 7873d1fc | 6 | New |
| [mean_58e9696dce58](../../../../canonical/mean_58e9696dce58/repro.py) | dde61cf0 | 5 | New |
| [mean_54e99b6b5074](../../../../canonical/mean_54e99b6b5074/repro.py) | 3fb7bf4d | 4 | New |
| [mean_c2de4b54d531](../../../../canonical/mean_c2de4b54d531/repro.py) | 37bea588 | 4 | New |
| [mean_mean_d1c6ce701eb2](../../../../canonical/mean_mean_d1c6ce701eb2/repro.py) | 1747c9ec | 4 | New |
| [mean_510438572607](../../../../canonical/mean_510438572607/repro.py) | 249302f1 | 1 | New |
| [pointwise_801dc7a5bcd4](../../../../canonical/pointwise_801dc7a5bcd4/repro.py) | 9c77a429 | 1 | New |
| [mean_mean_a0ab10b310aa](../../../../canonical/mean_mean_a0ab10b310aa/repro.py) | 1747c9ec | 1 | New |
| [mean_mean_a5a5d0354ee8](../../../../canonical/mean_mean_a5a5d0354ee8/repro.py) | 1747c9ec | 1 | New |
| [mean_mean_mean_b7a1035756e6](../../../../canonical/mean_mean_mean_b7a1035756e6/repro.py) | 2739cc8a | 1 | New |
| [pointwise_e5d8b71c9267](../../../../canonical/pointwise_e5d8b71c9267/repro.py) | 9c77a429 | 1 | New |
| [mean_mean_f3095e931799](../../../../canonical/mean_mean_f3095e931799/repro.py) | f6c36e9d | 1 | New |
| [pointwise_f7d2043cd67e](../../../../canonical/pointwise_f7d2043cd67e/repro.py) | 0e59119a | 1 | Reused |

Validation consumed the written artifacts through the shared repository APIs:

- Zero dropped regions and `roundtrip: ok` on the saved full graph.
- Input round-trip, partition determinism, and exact manifest pattern match;
  no missing, extra, or spelling-equivalent-only matches.
- Saved full graph executed with generated inputs: finite BF16 output
  `[1, 1, 262144]`.
- Exact pattern/shape/occurrence equality among the capture index, merged
  `shapes.json` records, and shared graph-accounting results from live FX.
- `scripts/validate_eager.py --all-shapes` passed all 17 affected points,
  including the three older points of the two reused patterns.
- `scripts/validate_corpus_invariants.py` passed all hard invariants and
  reported no soft warnings after stale local cache remnants were removed.

The baseline invariant run identified 40 directories containing only stale
Python caches. Twelve directory names were populated by this extraction;
the remaining 28 cache-only directory trees were deleted. No tracked corpus
files were missing, and all older shape records were preserved.

Runtime and source provenance:

| Component | Version/revision |
|---|---|
| better-benchmark base | `824c970f25f052de6cf66a4c634df246af00bb1e` |
| PyTorch | `2.16.0a0+gitfcc81df` / `fcc81df8cffbbff1ca2f0033add4441b7c81ebd9` |
| Transformers | `5.21.0.dev0` / `6a74bb9094493a513037f0c9812d5a97d76f4f5e` |
| GPU | `NVIDIA H200 MIG 3g.71gb` |
| Python | `/home/devuser/miniconda/bin/python` |

Source construction follows
`/workspaces/pytorch-devcontainers/specs/benchmarks/gemma/README.md` and its
`workload.py`/`original_config.json`. Local Transformers `src` and the
better-benchmark root were on `PYTHONPATH`. Missing `safetensors` and
`tokenizers` dependencies were installed without reinstalling existing
packages. Reports, the integrated workload, validation drivers, effective
config, capture index, and logs are under
`/home/devuser/.cache/better-benchmark/gemma4-31b-text/`.

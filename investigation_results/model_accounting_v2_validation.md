# Model Graph Accounting v2 — Validation

Date: 2026-06-10
Branch: investigations-june-2026

## What changed

`scripts/model_graph_accounting.py` previously reimplemented fusible/non-fusible
classification by hand (a hard-coded `NON_FUSIBLE_OP_NAMES` set + union-find on
"fusible" nodes). Its partitions were NOT guaranteed to match how the canonical
repros were cut, so pattern hashes could not be mapped back to
`repros/canonical/` and occurrence-weighted projections were unreliable.

It now REUSES the capture pipeline's partitioner. The partitioning + hashing
logic was factored out of `capture_hook._CaptureState.process_graph` into
module-level helpers in `capture_hook.py` that both the capture path and the
accounting script call:

- `get_fusion_partitions(gm)` — `CapabilityBasedPartitioner` +
  `is_fusible_node` (from `torch._inductor.fx_passes.fusion_regions`) +
  `create_op_support`, transparent view ops allowed inside partitions,
  `skip_horizontal_fusion=True` (with connected-component fallback for older
  torch), partitions without real compute dropped. Identical to the capture
  hook's setup — it IS the capture hook's setup, now shared.
- `extract_partition_subgraph(comp, gm)` — the subgraph extraction previously
  in `_CaptureState._extract_subgraph` (now a thin delegate).
- `compute_dag_signature(sub_gm)` / `pattern_hash_for_subgraph(sub_gm)` — the
  content-addressed 12-hex pattern hash (ops + wiring, ignoring shapes/scalar
  constants) that names `repros/canonical/<family>_<hash>` directories.
- `shape_hash_for_placeholders(placeholder_info)` — the 8-hex shape-config
  hash.
- `compute_partition_pattern(comp, gm)` — convenience wrapper returning both
  hashes plus the extracted sub-GraphModule.

Refactor equivalence check: ran the OLD `_CaptureState.process_graph`
(pre-refactor, via `git stash`) and the NEW one on both convnextv2 train
full graphs and compared the full (pattern_hash, shape_hash) multisets:
**IDENTICAL** (16 + 19 regions, byte-for-byte same hash pairs).
`scripts/test_adversarial.py` also passes (chain-splitting, stride
preservation, etc.).

The accounting script now:

- Loads full graphs via `full_graph_harness.load_full_graph` +
  `make_inputs_from_full_graph_specs`, traces with `make_fx` (fake mode first,
  real fallback) to get `node.meta['val']` everywhere.
- Partitions with `get_fusion_partitions` and hashes each partition with
  `compute_partition_pattern` — every partition maps directly to a
  `repros/canonical/<family>_<hash>` dir (via meta.json `pattern_hash`).
- **Counts occurrences**: each pattern row reports how many times the hash
  appears across the model's graphs (this was the missing piece — the capture
  pipeline dedups by hash, so manifests only record the unique set).
- Lists non-fusible ops (anything `is_fusible_node` rejects that the
  partitioner left out of every partition) with actual shapes from
  `node.meta['val']`.
- Flags UNMATCHED partitions (hash not in `repros/canonical/`) and annotates
  them with trace-equivalent canonical variants when the op multiset matches
  modulo reshape→view / clone canonicalization (see below).
- Aggregates time as **SUM of (per-partition oracle_us × occurrence count)**
  from `results/all_oracle_timings_b200_v2.json` — never averaged ratios.

## Key correctness check: convnextv2 manifest cross-check

Manifest claims: 6 patterns (infer), 18 patterns (train).

Re-partitioning the same saved full graphs with the SAME partitioner finds
**5 unique patterns (infer) and 13 (train)**. Every discrepancy is explained:

### The discrepancy mechanism: reshape→view retrace canonicalization

The saved `full_graph_*.py` files contain `aten.reshape.default` calls (they
were printed from the post-grad graph). The accounting path (and any path that
re-traces a saved graph through `make_fx`, e.g. `repartition_from_graphs.py`)
re-traces them, and tracing canonicalizes `aten.reshape` → `aten.view` (and
elides no-op `aten.clone`). The DAG-signature pattern hash encodes op names,
so the same partition gets TWO different hashes depending on whether it was
hashed at original capture time (reshape) or after a retrace (view).

The manifests were accumulated across both capture generations (original
capture + later repartition runs merge into the same manifest), so they
double-count these partitions — one hash per variant:

| reshape-variant (manifest-only) | view-variant (re-found) | same partition? |
|---|---|---|
| 26d1711c064d (sum_sum_sum) | f68c9f1fa09b | yes — op multiset identical mod reshape→view/clone |
| 70d71fcb0d68 (sum_sum_sum) | 431633879271 | yes |
| 9d590f58ecc9 (mean_var_mean) | c9089e261699 | yes |
| be3cca3d51c6 (sum_mean) | 39228c93d6a4 | yes |
| 6b7ff2251678 (sum_sum_sum) | dbc10f22635c | yes |
| e262d057f3c9 (sum_e262...) | 22b5cd24890b | yes |

Train manifest arithmetic: 18 hashes = 12 view-variant hashes + 6
reshape-variant hashes, of which 5 view-variants are ALSO in the manifest
(double-counted) and 1 (22b5cd24890b) is not. Collapsing variants: **13 unique
partitions — exactly what the accounting finds.** Both members of each pair
exist as separate `repros/canonical/` dirs (a known corpus-dedup issue, now
detectable with this tool).

Infer: 6 manifest hashes = 5 unique partitions after collapsing the
cf650837b7b1/3c4ea6d8f342 pair — exactly the 5 the accounting finds. The one
UNMATCHED infer partition (d742fc89ffad) is the view-variant of
sum_mean_273a847a963c (its canonical dir exists under the reshape hash from a
capture generation whose hash isn't reproducible from the saved graph).

Conclusion: the accounting partition count and hash set are consistent with
the manifests once the reshape/view hash aliasing is collapsed. There are NO
partitioner disagreements — the partition boundaries are identical (op
multisets match 1:1); only the hash spelling differs for 6-of-13 (train) /
2-of-5 (infer) patterns.

### Occurrence counts (the previously-missing piece)

convnextv2 train (90 partition occurrences across fwd+bwd, vs the manifest's
flat list of 18 deduped hashes):

| pattern | canonical dir | occurrences | oracle_us | sum_us |
|---|---|---:|---:|---:|
| 2f98fd23fbea | var_mean_2f98fd23fbea | 15 | — | — |
| 39228c93d6a4 | sum_mean_39228c93d6a4 | 14 | 73.6 | 1030.0 |
| e58ca1d0ccf4 | sum_sum_sum_e58ca1d0ccf4 | 14 | 78.0 | 1092.7 |
| 431633879271 | sum_sum_sum_431633879271 | 13 | — | — |
| 2e0e9617102b | sum_2e0e9617102b | 13 | 57.0 | 741.3 |
| 8e1dc74da682 | pointwise_8e1dc74da682 | 10 | 13.3 | 133.1 |
| 4df459ab207d | var_mean_4df459ab207d | 3 | 30.5 | 91.4 |
| d138e6df4d0c | sum_sum_sum_d138e6df4d0c | 3 | 191.4 | 574.1 |
| c9089e261699 | mean_var_mean_c9089e261699 | 1 | — | — |
| 508eb468b8d9 | sum_sum_sum_508eb468b8d9 | 1 | 113.6 | 113.6 |
| f68c9f1fa09b | sum_sum_sum_f68c9f1fa09b | 1 | — | — |
| dbc10f22635c | sum_sum_sum_dbc10f22635c | 1 | 9.7 | 9.7 |
| 22b5cd24890b | sum_22b5cd24890b | 1 | 6.2 | 6.2 |

Fusible time projection (SUM of us × occurrences, 9/13 patterns timed):
**3792 us oracle vs 6448 us compile** per train step — i.e. a flat per-pattern
analysis would weight var_mean_2f98fd23fbea (15 occurrences) the same as
sum_22b5cd24890b (1 occurrence), off by >10x in aggregate weighting.

Non-fusible (train): 46 convolution + 46 convolution_backward + 1 addmm +
2 mm, each with concrete shapes (e.g. `aten::convolution.default ->
[128, 320, 14, 14] f32 x17`).

## 5-model validation run

| model (suite) | graphs | partition occurrences | unique patterns | UNMATCHED | unexplained UNMATCHED | non-fusible ops | oracle sum_us | compile sum_us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| timm_convnextv2_nano infer (timm) | 1 | 43 | 5 | 1 | 0 | 47 | 913.7 (3/5 timed) | 536.0 |
| timm_convnextv2_nano train (timm) | 2 | 90 | 13 | 0 | 0 | 95 | 3792.1 (9/13) | 6447.9 |
| BERT_pytorch infer (torchbench) | 1 | 99 | 12 | 4 | 4 | 98 | 2293.3 (8/12) | 2262.9 |
| BERT_pytorch train (torchbench) | 2 | 198 | 26 | 1 | 1 | 294 | 20589.2 (24/26) | 11878.5 |
| DistilBertForMaskedLM infer (hf) | 1 | 53 | 14 | 0 | 0 | 50 | 2102.9 (13/14) | 2282.0 |
| DistilBertForMaskedLM train (hf) | 3 | 157 | 28 | 0 | 0 | 250 | 16178.4 (27/28) | 13171.5 |
| RMSNormForward (genai) | 1 | 1 | 1 | 0 | 0 | 0 | 342.9 (1/1) | 343.8 |
| facebook_opt-125m (vllm) | 5 | 10 | 10 | 0 | 0 | 8 | 227.5 (9/10) | 316.4 |

Manifest cross-checks: DistilBertForMaskedLM (infer + train), RMSNormForward,
and convnextv2 (after variant collapse) are EXACT MATCHES. facebook_opt-125m
has no manifest (vllm capture predates manifest writing). BERT_pytorch
manifests are subsets of what the partitioner finds (8 of 12 infer / 25 of 26
train) — the manifests were written before validation-gating was relaxed, so
patterns whose generated repros failed eager validation were never merged.

## True UNMATCHED partitions (corpus coverage gaps)

BERT_pytorch is the only validated model with genuinely unmatched partitions
(no exact hash AND no trace-equivalent canonical dir):

| pattern | occurrences | kind | what it is |
|---|---:|---|---|
| c56e1bde1517 | 11 (infer) | reduction | masked softmax: eq/where + amax/exp/sum on [1536,128,128] with bool mask |
| d4e3a818f94c | 1 (infer) | reduction | same softmax but with mask built inline from int64 ids (gt/repeat/unsqueeze) |
| 79f21f639a3a | 1 (infer) + 1 (train) | reduction | log_softmax over [16384, 20005] vocab |
| d5f17b931ab1 | 1 (infer) | pointwise | residual add + select on [128,128,768] |

These four patterns (14 partition occurrences total in BERT_pytorch infer)
never made it into `repros/canonical/` — likely eager-validation failures
during the original merge (bool-mask and int64-id inputs are exactly the
input classes the validator used to choke on). They are real corpus coverage
gaps worth recapturing; the masked-softmax one runs 11x per inference pass.

## Caveats

- Pattern hashes are spelling-sensitive to reshape/view: corpus dirs exist
  under both spellings for at least 8 patterns (6 convnextv2-train + 2
  convnextv2-infer pairs found here). A corpus dedup pass keyed on
  trace-normalized op multisets + wiring would collapse these.
- Time projection covers only patterns with `oracle_us` in
  `results/all_oracle_timings_b200_v2.json`; rows show timed/total counts.
  Missing rows are either BAD_ORACLE-filtered or untimed, NOT zero-cost.
- Non-fusible op timings (conv/mm/sdpa) are out of scope here; the table
  reports their shapes so flop-based estimates can be layered on.

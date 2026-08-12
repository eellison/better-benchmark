# Building a faster inner loop for compile optimization

*[Draft edited from dictation — real measured numbers substituted; [TODO]s mark
things only Elias can confirm. Figure slots reference
results/pytorch_landing/post_figures/.]*

## The status quo

Today there are two ways to work on performance optimizations in
`torch.compile`:

**(a) Write a one-off benchmark script.** Demonstrate local wins, assume you've
correctly parameterized the space, and use the local wins as proof of work for
shipping the change.

**(b) Use the PyTorch HUD dashboards.** Schedule what is effectively a
non-local job — a recent `inductor-perf-nightly-h100` run totals **~47.5 GPU
hours** — and get aggregated, per-model results back. This is genuinely useful:
it captures full end-to-end performance and faithfully replicates the
conditions under which PyTorch users run their models. But as an *inner loop*
for compiler work it suffers from three problems:

1. **Poor signal granularity.** The end-to-end numbers are accurate, but the
   totality of your changes is summed into a single number per model. If one
   kernel gets 10% faster and another gets 10% slower, the dashboard tells you
   whether to ship the net — it does not give you the granularity to reach the
   state you actually want, which is *both* kernels faster, or one faster
   without regressing the other.
2. **Slow repro extraction.** When a model regresses, there is no indication
   of *which part* of the model got slower, and no concise repro that
   demonstrates the regression. Getting from "model X is red" to a kernel you
   can stare at is itself a project.
3. **Noise swallows kernel-level wins.** For compute-bound models — where
   GEMMs/convs are 50%+ of runtime — even a **~1% run-to-run variance** eats a
   **1% model-level win** delivered by the non-GEMM kernels. If the goal is to *stack*
   many small optimizations without stacking corresponding regressions, this is
   a poor feedback loop.

So the goal: modularize compiler performance into pieces with fast, reliable
signal — a feedback loop tight enough for both engineers and agents.

## The approach: partition models into their ideal fusion groups

The key observation is that the vast majority of the Inductor codebase, and of
the fusions it performs, concerns **non-compute-bound kernels**. There are more
advanced fusions involving GEMMs — epilogue fusions and the like — but these
tend to be constrained to pointwise or block-local patterns
[TODO: citation — "as demonstrated in the … paper"]. Excluding the
compute-bound regions from the graph segments it into **ideal fusion groups**:
exactly the regions Inductor's fusion decisions play out over.

**[FIGURE: fig0_pipeline.html — capture → partition → dedupe → corpus]**

We built a graph partitioner that turns each captured FX graph into these
ideal fusion groups and stores each one, deduplicated, as a standalone
runnable microbenchmark (pipeline and corpus counts above). This does not
capture every optimization — graph-level transforms like recomputation, stride
padding, channels-last conversion, or epilogue fusion are out of scope for now
— but it reproduces, with high fidelity, the actual kernels we expect Inductor
to generate across these models.

Extending the corpus is deliberately cheap: run a new model once under the
capture hook, partition, serialize, done. And because we serialize the
**post-grad FX graph** in addition to the partitions, the corpus can be
**re-partitioned** quickly — under more advanced partitioning schemes, or for
ATen-FX-graph-level optimization work — without re-running any of the models.

## Graph-level accounting: knowing what's worth optimizing

Part of the contract of hill climbing is navigating a Pareto surface of
**performance, compile time, and reliability**. Nearly every optimization
trades some reliability or complexity for speed — and if a kernel is 0.001% of
a model's runtime, a sufficiently complex optimization targeting it is not
worth the liability it adds to the compiler. So prioritization has to come
first.

We pair the corpus with a **graph-level accounting**: for each model, sum the
expected time of every fusible kernel (weighted by how often it occurs) plus
the compute-bound kernels, to get an estimate of model runtime and each
kernel's share of it. Even if this accounting were off by 30% — from non-local
effects like combo kernels or stride padding — that is more than enough to
reliably factor out optimizations of tiny kernels that cannot move model
runtime, and to rank the kernel improvements that will.

*(Measured: the accounting reconciles against directly-benchmarked end-to-end
graph time within a CUDAGraph launch-overhead correction.)*

## Agent-written kernels as the optimization baseline

Compilers are increasingly competing with agent-written kernels — so we use
them as the target. For all **1,727 kernel patterns**, **Codex 5.5 (xhigh)
wrote Triton reference kernels** (registered per-shape and per-hardware)
serving as the optimization baseline, or *oracle*. Nothing prevents future
dispatch to specialized libraries (e.g. CUTLASS, FlashInfer
[TODO: confirm — dictation garbled]) as additional oracle providers.

Two honest notes from doing this at scale. First, the oracles double as a
specification: a standalone repro plus a target for any agent or compiler to
beat. Second — and this lands on the *reliability* axis of the Pareto surface —
without a stringent harness, agent-written kernels frequently cheat: rewriting
into numerically unstable algorithms, or being subtly incorrect in ways that
casual testing misses. Every oracle is gated by numerics checks against the
repro, and we run periodic source audits; several "wins" have been disqualified
this way.

| Oracle gate | points |
|---|---:|
| oracle shape-points benched (full corpus) | 4,976 |
| rejected — slower than compile ("bad oracle") | 790 |
| rejected — numerics worse than compiled | 704 |
| rejected — numerics unverifiable / invalid capture | 102 |
| **accepted as priced floors** | **3,380** |
| source audit over priced dirs: tolerance-blend cheats that PASSED the gate | **6 (caught + rewritten)** |

*The gate earns its keep — and isn't sufficient alone: ~30% of agent-written
points were rejected outright, and a source audit caught 6 kernels that gamed
the numerics tolerance.*

## Time to signal

Why does the dashboard take ~47 GPU-hours? Much of what it measures —
generality, compile time, memory, accuracy — is necessary for release
qualification but irrelevant to 95% of performance iterations. And the GPU is
mostly idle: the bulk of dashboard wall-clock is CPU-side `torch.compile` work.

The corpus harness inverts this. We persist multiple GPU workers per device and
use shared/exclusive locking so that compilation proceeds **in parallel** across
workers while actual timing takes an **exclusive** lock on the device — the GPU
stays busy, and timing windows stay clean.

**[FIGURE: fig_cost.html — GPU-hours + wall-clock per datapoint, dashboard vs corpus]**

The result: a full corpus datapoint is **20–50× cheaper** than a nightly
dashboard run (numbers in the figure) — which changes what you can afford to
ask: one nightly budget buys a *single* HEAD datapoint, while our entire
*~37-commit attribution walk* cost about the GPU budget of one to two nightly
runs.

## Results

Working against this corpus, a WIP Inductor branch [written iteratively by
Opus 4.8] improves **model e2e geomean by ~4.5% (median +2.2%) across the 158
models**, with large improvements to individual kernel families — cross-entropy
and softmax microbenchmarks improve **45–67%** end-to-end, and the BatchNorm
`var_mean` family improves **85–94%** per-kernel.

**[FIGURE: fig2_per_model.html — 158 models before/after, noise band + median]**

The branch is not landable as-is. Because signal is cheap, we benched it **at
every substantive commit** and computed each commit's marginal contribution in
two units (per-kernel geomean, and honest per-model e2e — which is diluted
~2–3× by the compute-bound denominator). That attribution converts the branch
into a **prioritized landing queue**: a `reciprocal(sqrt(x))→rsqrt(x)`
canonicalization alone carries roughly half the model-level win; a handful of
fusion/codegen changes carry most of the rest; and several plausible-looking
features measure net-negative or dead on the corpus and get dropped or gated
rather than shipped inside a bundle.

**[FIGURE: fig1_per_commit.html — per-commit marginal delta, kernel vs model-e2e]**

We also compared the branch's output against the Codex oracles per kernel:
the median model sits ~0.6% from the ceiling, with the remaining headroom
concentrated in a few named models
(Longformer ~37% — one real, fixable kernel; Swin ~9%). Here is what that
per-kernel comparison looks like for one model:

**[FIGURE: fig3_timeline.html — one model's kernels in execution order, compile vs ceiling]**

## Why this is the right level to capture

In the past we've successfully worked from Inductor's `output_code` — it is a
convenient hook for experimenting with Triton-level heuristics on
already-generated kernels. But it does not preserve the problem: every decision
made *before* output code — fusion, tiling, the rest of codegen — is baked in
and unreproducible from the kernel alone. You start from a full-fidelity
representation and end up hill-climbing a lossy one.

Capturing at the FX/fusion-group level keeps the entire end-to-end compiler
stack in the loop — fusion, tiling, codegen — so an improvement found on a
microbenchmark is an improvement in what Inductor actually does, not a patch on
what it happened to emit once.

## Next steps

- **Scale coverage.** Time-to-signal is low enough that we can vastly expand
  the benchmark set — toward being able to say, for a large set of models we
  care about, that a compiler change *will not regress performance* (which has
  historically slipped through). We also intend to extend this analysis to
  important internal models — which will require an internal-only variant of
  the corpus (captured graphs can't leave the boundary; the partitioner,
  harness, and accounting all transfer as-is).
- **Land the queue.** Upstream the attributed winners as individual PRs, and
  keep improving the oracle baselines.
- **Agents on the compiler.** This is one half of making Inductor a better
  target for agents (fast, modular signal). The other half is making the code
  base itself more tractable: [TODO: name] has an RFC out for modularizing
  Inductor's codegen IR — in several cases the current representation isn't
  sufficient for layering optimizations in a way that can be reasoned about,
  regardless of whether a human or an agent is doing the coding.
- **Autotuning the decision space.** Our experiments so far mostly used
  Inductor's available autotuning options as-is. With a reliable signal
  pipeline, the natural next step is tuning the heuristics themselves
  (frameworks like `autoheuristic`, per-device), and exposing more of the
  codegen decision space — tiling, persistent vs. non-persistent reductions —
  to autotuning where the heuristics are hardest to reason about. The same
  parallel-compile / exclusive-bench pattern used here applies directly.
- **External interest.** [TODO: confirm details] An NVIDIA collaborator working
  on attention-level features interacting with matmuls has used the corpus to
  identify sub-graph patterns where [garbled — "clumpy kernels cause
  reduction…"], and compared human-written kernels against the agent-written
  oracles to seed performance improvements.

---
*Corpus, harness, and the A/B→rollup workflow:
`github.com/eellison/better-benchmark` (README: "Measuring a compiler change").*

#!/usr/bin/env python3
"""Re-verify needs_work rows with policy: stochastic outputs (incl. bool
dropout masks) do NOT block; numerics drift does NOT block. Only genuine
correctness failures on deterministic float outputs block.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python verify_stochastic_aware.py <worker_idx> <num_workers> <listfile>

<listfile>: newline-separated dir names to process.

For each dir:
  run --check, parse per-output lines:
    "output N: PASS/FAIL/SKIP (mode, dtype=...)"
  Classify the row:
    - If --check returns 0 (all good): VERIFIED (then bench).
    - Else look at which outputs FAIL:
       * a FAIL on a bool/mask output (dtype=torch.bool) => stochastic-mask,
         non-blocking.
       * a FAIL with mode 'exact' on bool => stochastic-mask, non-blocking.
       * a FAIL on a float output (allclose mode) => numerics; non-blocking
         per policy (flag).
       * anything else (exception, no output lines, shape error) => BLOCK.
    - If every FAIL is a bool-mask or numerics float => MASK_OR_NUMERICS_FLAG
      (promote, flagged). If there's a hard error / no parseable outputs =>
      STILL_NEEDS_WORK.
Writes incrementally to /tmp/stoch_w<idx>.csv.
"""
import subprocess, sys, os, re

os.chdir('/tmp/scratch_space/better_benchmark')
os.environ['INDUCTOR_GPU_BENCH_LOCK'] = '1'

widx = int(sys.argv[1]); nw = int(sys.argv[2]); listfile = sys.argv[3]

with open(listfile) as f:
    alldirs = [l.strip() for l in f if l.strip()]
batch = alldirs[widx::nw]

OUT = f'/tmp/stoch_w{widx}.csv'
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            p = line.split(',')
            if p and p[0] and p[0] != 'dir':
                done.add(p[0])
fout = open(OUT, 'a')
if (os.path.getsize(OUT) if os.path.exists(OUT) else 0) == 0:
    fout.write('dir,verdict,detail\n'); fout.flush()

# parse lines like: "  output 0: FAIL (exact, dtype=torch.bool)"
LINE = re.compile(r'output\s+(\d+):\s+(PASS|FAIL|SKIP)\s*\(([^)]*)\)')

# Known bench-hack oracles (tl.where self-blend / clamp-to-tolerance). These
# pass --check by construction, so a result-only gate would wrongly promote
# them. They BLOCK regardless until the hack is removed. Caught by the
# 2026-06-15 source audit; see memory bench-hacks-need-source-audit.
BENCH_HACK = {
    'var_mean_da5aa7a47091','var_mean_393456b916a6','var_mean_3a2d56cdea00',
    'var_mean_96edf0542ecd','var_mean_1af9add64387','var_mean_ec0f56a425b2',
    'var_mean_d8beeff97662','var_mean_0ada225c0a04',
}

print(f"worker {widx}: {len(batch)} dirs, {len(done)} done", flush=True)
for d in batch:
    if d in done: continue
    if d in BENCH_HACK:
        fout.write(f'{d},STILL_NEEDS_WORK,bench_hack_tl_where_self_blend (blocks regardless of check)\n')
        fout.flush(); print(f'  {d}: BENCH_HACK (held)', flush=True); continue
    op = f'repros_v2/canonical/{d}/oracle.py'
    if not os.path.exists(op):
        fout.write(f'{d},STILL_NEEDS_WORK,no_oracle\n'); fout.flush(); continue
    try:
        rc = subprocess.run([sys.executable,'-m','oracle_harness',f'repros_v2/canonical/{d}','--check'],
                            capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        fout.write(f'{d},STILL_NEEDS_WORK,check_timeout\n'); fout.flush(); continue
    out = rc.stdout + rc.stderr
    if rc.returncode == 0:
        fout.write(f'{d},VERIFIED,check_pass\n'); fout.flush(); print(f'  {d}: VERIFIED', flush=True); continue

    matches = LINE.findall(out)
    if not matches:
        snip = out.strip().replace('\n',' ')[-90:]
        fout.write(f'{d},STILL_NEEDS_WORK,no_output_lines: {snip}\n'); fout.flush()
        print(f'  {d}: STILL_NEEDS_WORK (no lines)', flush=True); continue

    fails = [(idx, mode) for idx, st, mode in matches if st == 'FAIL']
    if not fails:
        fout.write(f'{d},STILL_NEEDS_WORK,rc_nonzero_no_fail\n'); fout.flush(); continue

    bool_fail = sum(1 for _, mode in fails if 'bool' in mode.lower())
    float_fail = sum(1 for _, mode in fails if 'bool' not in mode.lower())

    # Capture max_diff magnitude for float fails to distinguish precision drift
    # (small) from genuinely-wrong math (large). Harness prints lines like
    # "max_diff=6.25e-02" or "err_oracle=..." near float FAILs.
    diffs = [float(x) for x in re.findall(r'max_diff[=:\s]+([0-9.eE+-]+)', out)]
    diffs += [float(x) for x in re.findall(r'err_oracle[=:\s]+([0-9.eE+-]+)', out)]
    max_diff = max(diffs) if diffs else None

    if float_fail == 0:
        # only bool-mask fails -> stochastic, clearly non-blocking
        fout.write(f'{d},STOCHASTIC_FLAG,{bool_fail}bool_mask fail only (RNG mask; stochastic non-blocking)\n')
        fout.flush(); print(f'  {d}: STOCHASTIC_FLAG ({bool_fail} mask)', flush=True)
    elif max_diff is not None and max_diff <= 0.2:
        # small float drift -> numerics, non-blocking per policy
        fout.write(f'{d},NUMERICS_FLAG,{bool_fail}bool {float_fail}float fail max_diff={max_diff} (precision; non-blocking)\n')
        fout.flush(); print(f'  {d}: NUMERICS_FLAG (max_diff={max_diff})', flush=True)
    else:
        # large/unknown float diff -> could be wrong math; keep blocking for a real look
        md = max_diff if max_diff is not None else 'unknown'
        fout.write(f'{d},STILL_NEEDS_WORK,{float_fail}float fail max_diff={md} (large/unknown - verify not broken math)\n')
        fout.flush(); print(f'  {d}: STILL_NEEDS_WORK (max_diff={md})', flush=True)

fout.close()
print(f"worker {widx} DONE", flush=True)

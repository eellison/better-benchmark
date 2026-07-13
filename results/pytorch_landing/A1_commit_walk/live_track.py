#!/usr/bin/env python3
"""Live per-state attribution — HARDENED. Only rolls up COMPLETE sweeps
(__summary__.ok >= 1700) and SHAPE-MATCHES (a model's us uses only kernels priced
in BOTH the state and the base) so partial/mismatched coverage can't fake a delta."""
import json, glob, os, math
from pathlib import Path
from collections import defaultdict

STATES="results/pytorch_landing/A1_commit_walk/states"
OCC="results/b200/occurrences"
COARSE=[l.strip() for l in open("results/pytorch_landing/A1_commit_walk/coarse_states.txt") if l.strip()]

def complete(path):
    try:
        d=json.load(open(path)); s=d.get('__summary__',{})
        return s.get("ok",0)>=1700
    except Exception: return False

def kern_idx(path):
    d=json.load(open(path)); out=defaultdict(dict)
    for k,v in d.items():
        if k.startswith('_'): continue
        ph=Path(k).parent.name.rsplit('_',1)[-1] if '/' in k else k.rsplit('_',1)[-1]
        for inner,mv in v.items():
            if isinstance(mv,dict): out[ph][inner.rsplit('_',1)[-1]]=mv.get('coord_descent_us') or mv.get('compiled_us')
    return out

def geomean(xs):
    xs=[x for x in xs if x>0]; return math.exp(sum(map(math.log,xs))/len(xs)) if xs else float('nan')

base_p=f"{STATES}/{COARSE[0]}.json"
if not complete(base_p): print("base not complete yet"); raise SystemExit
base=kern_idx(base_p)
occs={os.path.basename(f).replace('__','/').replace('.json',''):json.load(open(f))
      for f in glob.glob(f"{OCC}/*.json") if 'genai' not in os.path.basename(f)}

def model_speedups(cur):
    """per-model base/cur geomean, SHAPE-MATCHED: only (ph,sh) priced in BOTH."""
    sp=[]
    for m,occ in occs.items():
        bt=ct=0.0; any_pt=False
        for ph,shapes in (occ.get('fusible') or {}).items():
            for sh,cnt in (shapes.items() if isinstance(shapes,dict) else []):
                bu=(base.get(ph) or {}).get(sh); cu=(cur.get(ph) or {}).get(sh)
                if bu and cu: bt+=bu*cnt; ct+=cu*cnt; any_pt=True   # BOTH priced
        if any_pt and ct>0: sp.append(bt/ct)
    return sp

print(f"base {COARSE[0][:12]}: {len(base)} patterns, {len(occs)} models\n")
print(f"{'#':>2} {'commit':13}{'state':>7}{'vs-base gm':>12}{'seg':>9}  matched-models")
prev=None
for i,sha in enumerate(COARSE,1):
    p=f"{STATES}/{sha}.json"
    if not os.path.exists(p): print(f"{i:>2} {sha[:12]}{'--':>7}"); continue
    if not complete(p):
        try: ok=json.load(open(p)).get('__summary__',{}).get('ok','?')
        except: ok='?'
        print(f"{i:>2} {sha[:12]}{'PARTIAL':>7}  (ok={ok}/1727, skip)"); continue
    cur=kern_idx(p); sp=model_speedups(cur); gm=geomean(sp); pct=(gm-1)*100
    seg="" if prev is None else f"{(gm/prev-1)*100:+.2f}pp"
    print(f"{i:>2} {sha[:12]}{'OK':>7}{pct:+11.2f}%{seg:>9}  {len(sp)}")
    prev=gm

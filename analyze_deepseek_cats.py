"""Analyze cat patterns in DeepSeek graphs to predict split opportunities."""
import re
import os
from collections import defaultdict

graph_dir = "/tmp/scratch_space/better_benchmark/deepseek"

for fname in sorted(os.listdir(graph_dir)):
    if not fname.startswith("graph") or not fname.endswith(".txt"):
        continue
    filepath = os.path.join(graph_dir, fname)
    with open(filepath) as f:
        lines = f.readlines()
    
    # Find all cat operations and their consumers
    cats = {}
    all_vars = {}  # var_name -> defining line
    
    for i, line in enumerate(lines):
        # Parse variable assignments: "varname: ..." = ...
        m = re.match(r'\s+(\w+):\s*"([^"]+)"\s*=\s*(.*)', line)
        if m:
            var_name = m.group(1)
            shape_info = m.group(2)
            rhs = m.group(3)
            all_vars[var_name] = {
                'line': i+1,
                'shape': shape_info,
                'rhs': rhs.strip(),
            }
            if 'aten.cat' in rhs:
                # Extract inputs from the cat call
                cat_args_m = re.search(r'\[([^\]]+)\]', rhs)
                inputs_str = cat_args_m.group(1) if cat_args_m else ''
                dim_m = re.search(r'],\s*(-?\d+)', rhs)
                dim = int(dim_m.group(1)) if dim_m else -1
                cats[var_name] = {
                    'line': i+1,
                    'shape': shape_info,
                    'inputs': inputs_str,
                    'dim': dim,
                    'consumers': [],
                }
    
    # Find consumers of each cat output
    for var_name, info in all_vars.items():
        for cat_name in cats:
            if cat_name in info['rhs'] and var_name != cat_name:
                cats[cat_name]['consumers'].append(f"{var_name} ({info['shape'].split(']')[0]}])")
    
    if not cats:
        continue
    
    print(f"\n{'='*60}")
    print(f"{fname}:")
    for cat_name, info in cats.items():
        n_consumers = len(info['consumers'])
        # Determine if this would be pointwise_cat
        # 2-input cat on last dim with pointwise inputs → likely pointwise_cat
        inputs = [x.strip() for x in info['inputs'].split(',')]
        n_inputs = len(inputs)
        
        would_split = "?" 
        if n_consumers == 0:
            would_split = "YES (no consumers, goes to output)"
        elif n_consumers == 1:
            # Check if consumer is pointwise-ish
            would_split = "NO (has consumer → profitability blocks)"
        else:
            would_split = "NO (multiple consumers)"
        
        print(f"  {cat_name} (line {info['line']}): {info['shape']}")
        print(f"    inputs: {info['inputs']}, dim={info['dim']}, n_inputs={n_inputs}")
        print(f"    consumers ({n_consumers}): {info['consumers'][:3]}")
        print(f"    would_split: {would_split}")

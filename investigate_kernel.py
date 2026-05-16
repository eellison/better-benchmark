"""
Investigate WHY a kernel has a gap vs SOL.
Runs the repro with TORCH_LOGS=output_code to capture the generated Triton kernel,
then analyzes the kernel structure.
"""
import json
import os
import subprocess
import sys
import tempfile


def investigate(script_path: str, label: str = ""):
    """Run a repro with output_code logging to see what Inductor generates."""
    env = os.environ.copy()
    env["TORCH_LOGS"] = "output_code"
    env.pop("TORCH_LOGS_FORMAT", None)

    # Run the script just to compile (not benchmark), capture output_code
    wrapper = f"""
import torch
import torch._inductor.config as inductor_config
import sys
sys.path.insert(0, '{os.path.dirname(script_path)}')
exec(open('{script_path}').read())

# Just compile and run once to trigger code generation
mod = Repro()
inputs = make_inputs()
compiled = torch.compile(mod)
with torch.no_grad():
    out = compiled(*inputs)
    torch.cuda.synchronize()
print("COMPILE_SUCCESS")
"""

    proc = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    # Extract the generated Triton code from stderr (TORCH_LOGS goes to stderr)
    triton_code = []
    in_triton = False
    for line in proc.stderr.split("\n"):
        if "@triton.jit" in line or "def triton_" in line:
            in_triton = True
        if in_triton:
            triton_code.append(line)
        if in_triton and line.strip() == "" and len(triton_code) > 5:
            # End of a kernel
            triton_code.append("")

    # Also look for kernel launch info
    launch_info = []
    for line in proc.stderr.split("\n"):
        if "grid" in line.lower() and ("triton" in line.lower() or "kernel" in line.lower()):
            launch_info.append(line.strip())

    # Count number of distinct kernels
    kernel_count = proc.stderr.count("@triton.jit") + proc.stderr.count("@triton_heuristics")

    result = {
        "label": label,
        "script": script_path,
        "compile_success": "COMPILE_SUCCESS" in proc.stdout,
        "num_triton_kernels": kernel_count,
        "triton_code_lines": len(triton_code),
        "stderr_lines": len(proc.stderr.split("\n")),
    }

    # Save full output for detailed analysis
    out_dir = os.path.dirname(script_path)
    base_name = os.path.basename(script_path).replace(".py", "")

    with open(os.path.join(out_dir, f"{base_name}_investigation.txt"), "w") as f:
        f.write(f"=== INVESTIGATION: {label} ===\n\n")
        f.write(f"Script: {script_path}\n")
        f.write(f"Compile success: {result['compile_success']}\n")
        f.write(f"Number of Triton kernels generated: {kernel_count}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(proc.stdout[:5000])
        f.write("\n\n=== TRITON KERNELS (from stderr) ===\n")
        f.write("\n".join(triton_code[:500]))
        f.write("\n\n=== LAUNCH INFO ===\n")
        f.write("\n".join(launch_info[:50]))
        f.write("\n\n=== FULL STDERR (last 10000 chars) ===\n")
        f.write(proc.stderr[-10000:])

    return result


def investigate_category(category: str, scripts: list[str]):
    """Investigate all scripts in a category."""
    print(f"\n{'='*80}")
    print(f"INVESTIGATING: {category}")
    print(f"{'='*80}")

    results = []
    for script in scripts:
        if not os.path.exists(script):
            print(f"  SKIP (not found): {script}")
            continue
        print(f"\n  Investigating: {os.path.basename(script)}")
        try:
            r = investigate(script, label=f"{category}/{os.path.basename(script)}")
            results.append(r)
            print(f"    Kernels generated: {r['num_triton_kernels']}")
            print(f"    Triton code lines: {r['triton_code_lines']}")
        except Exception as e:
            print(f"    ERROR: {e}")

    return results


CATEGORIES = {
    "cross_entropy_amax_sum": [
        "output/aten_repros/dynamo_OPTForCausalLM_inference/region_000_amax_sum_8134ec8b7263_6de12afb.py",
        "output/aten_repros/dynamo_DistillGPT2/region_007_amax_sum_4cc1548fbf82_da67baef.py",
        "output/aten_repros/dynamo_AlbertForMaskedLM/region_000_amax_sum_eb4fe3ac03e0_d46ab65d.py",
    ],
    "rmsnorm_mean": [
        "output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_001_mean_4bbcddf61f1a_5a50d8c5.py",
        "output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_003_mean_376234b0e316_5a50d8c5.py",
        "output/aten_repros/vllm_deepseek-ai_DeepSeek-V3_inference/region_038_mean_bf5fef91b4d7_8d7ec724.py",
        "output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_006_mean_71e6d6b18006_5a50d8c5.py",
    ],
    "layernorm_var_mean": [
        "output/aten_repros/dynamo_BertForMaskedLM/region_001_var_mean_0a1bc21703bb_e045a23d.py",
        "output/aten_repros/dynamo_OPTForCausalLM_inference/region_005_var_mean_e708b478a57b_68c6dbc3.py",
        "output/aten_repros/dynamo_AlbertForMaskedLM/region_006_var_mean_a7cbd072693b_369a51f8.py",
    ],
    "weight_grad_sum": [
        "output/aten_repros/dynamo_BertForMaskedLM/region_025_sum_c25f97f8bb7a_72858422.py",
        "output/aten_repros/dynamo_BertForMaskedLM/region_005_sum_94d9ad74b3d3_4d35693e.py",
        "output/aten_repros/dynamo_BertForMaskedLM/region_016_sum_bab7a7b0118a_a53551b1.py",
        "output/aten_repros/dynamo_DistillGPT2/region_013_sum_c15fcea82dde_66ab4406.py",
    ],
    "rope_attention_pointwise": [
        "output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_005_pointwise_108892bc6e62_2851ce74.py",
        "output/aten_repros/dynamo_meta-llama/Llama-3.2-1B_inference/region_007_pointwise_fc69a1a83866_a2fe9c92.py",
        "output/aten_repros/dynamo_OPTForCausalLM_inference/region_001_pointwise_86432a5d8e65_2381e209.py",
        "output/aten_repros/vllm_openai_gpt-oss-20b_inference/region_008_pointwise_35deec51cf28_d037e534.py",
    ],
    "moe_small_pointwise": [
        "output/aten_repros/vllm_openai_gpt-oss-20b_inference/region_006_pointwise_8206ef619f2b_dc9ab797.py",
        "output/aten_repros/vllm_deepseek-ai_DeepSeek-V3_inference/region_035_pointwise_863609bc9483_d029e068.py",
        "output/aten_repros/dynamo_DistillGPT2/region_008_pointwise_9c3e580878a3_8713b6ea.py",
        "output/aten_repros/vllm_Qwen_Qwen3-0.6B_inference/region_009_pointwise_74957135a3e5_6bfb868a.py",
    ],
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <category> [category2 ...]")
        print(f"Available: {list(CATEGORIES.keys())}")
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    categories = sys.argv[1:]
    all_results = {}
    for cat in categories:
        if cat not in CATEGORIES:
            print(f"Unknown category: {cat}")
            continue
        results = investigate_category(cat, CATEGORIES[cat])
        all_results[cat] = results

    # Save summary
    with open("investigation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nDone. Investigation files saved next to each repro.")

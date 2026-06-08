"""
Recapture all timm CI models with the fixed capture hook.

Channels-last, CI batch sizes, infer + train modes.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch._dynamo
from torch._inductor.utils import fresh_inductor_cache
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import temporary_capture_for_merge

TIMM_MODELS = {
    "adv_inception_v3": 128,
    "beit_base_patch16_224": 128,
    "convnextv2_nano.fcmae_ft_in22k_in1k": 128,
    "deit_base_distilled_patch16_224": 128,
    "deit_tiny_patch16_224.fb_in1k": 128,
    "dm_nfnet_f0": 128,
    "ghostnet_100": 512,
    "inception_v3": 128,
    "mobilenetv2_100": 128,
    "mobilenetv3_large_100": 512,
    "mobilevit_s": 128,
    "nfnet_l0": 128,
    "repvgg_a2": 128,
    "swin_base_patch4_window7_224": 128,
    "tf_efficientnet_b0": 128,
    "visformer_small": 128,
    "vit_base_patch14_dinov2.lvd142m": 128,
    "vit_base_patch16_siglip_256": 128,
}

OUTPUT_DIR = Path("/tmp/scratch_space/better_benchmark/repros")


def capture_model(model_name: str, batch_size: int, mode: str):
    """Capture one model in one mode. Returns (n_regions, time_s)."""
    import timm

    label = f"timm_{model_name}_{mode}"

    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()

    model = None
    inp = None
    with temporary_capture_for_merge(
        OUTPUT_DIR,
        model_name,
        suite="timm",
        mode=mode,
        prefix="recapture_timm_",
    ) as capture:
        cap_dir = capture.capture_dir
        try:
            model = timm.create_model(model_name, pretrained=False).cuda()
            model = model.to(memory_format=torch.channels_last)

            data_config = timm.data.resolve_model_data_config(model)
            input_size = data_config.get("input_size", (3, 224, 224))
            inp = torch.randn(batch_size, *input_size, device="cuda")
            inp = inp.to(memory_format=torch.channels_last)

            if mode == "train":
                model.train()
            else:
                model.eval()

            # Save full graphs to the same per-model directory as manifest.json.
            model_dir = OUTPUT_DIR / "models" / "timm" / mode / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            install_capture_hook(str(cap_dir), label=label, graph_dir=str(model_dir))

            t0 = time.time()
            with fresh_inductor_cache():
                compiled = torch.compile(model)
                if mode == "train":
                    out = compiled(inp)
                    if isinstance(out, torch.Tensor):
                        out.sum().backward()
                    elif isinstance(out, (tuple, list)):
                        loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
                        loss.backward()
                else:
                    with torch.no_grad():
                        compiled(inp)
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            uninstall_capture_hook()

            # Merge into canonical set
            n = capture.merge()
            return n, elapsed

        except Exception as e:
            print(f"  FAILED: {e}")
            try:
                uninstall_capture_hook()
            except Exception:
                pass
            return 0, 0.0
        finally:
            del model, inp
            torch.cuda.empty_cache()
            gc.collect()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to capture (default: all)")
    parser.add_argument("--mode", choices=["infer", "train", "both"], default="both")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Start from this model (skip earlier ones)")
    args = parser.parse_args()

    models = args.models or list(TIMM_MODELS.keys())
    if args.start_from:
        try:
            idx = models.index(args.start_from)
            models = models[idx:]
        except ValueError:
            pass

    modes = ["infer", "train"] if args.mode == "both" else [args.mode]


    total_regions = 0
    total_time = 0
    results = []

    for model_name in models:
        batch_size = TIMM_MODELS.get(model_name, 128)
        for mode in modes:
            label = f"timm_{model_name}_{mode}"
            print(f"\n{'='*60}")
            print(f"  {label} (batch={batch_size})")
            print(f"{'='*60}")

            n, elapsed = capture_model(model_name, batch_size, mode)
            total_regions += n
            total_time += elapsed
            results.append({"model": model_name, "mode": mode, "regions": n, "time": elapsed})
            print(f"  => {n} regions in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"DONE: {total_regions} total regions from {len(results)} model runs in {total_time:.0f}s")
    print(f"{'='*60}")

    # Save summary
    summary_path = OUTPUT_DIR.parent / "capture_summary_timm.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

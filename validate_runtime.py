"""Check real downloaded layers through Maestro/MMGP and the AMD GPU."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from converter import convert_file, decode_chunk, regular_hadamard, unswizzle_scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    app = Path(args.app).resolve()
    report_path = Path(args.report).resolve()
    sys.path.insert(0, str(app))
    os.chdir(app)
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
    os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "0")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    torch.set_num_threads(4)
    from accelerate import init_empty_weights
    from mmgp import offload, quant_router
    quant_router.register_handler("shared.qtypes.int8_convrot")
    from shared.kernels.quanto_int8_inject import maybe_enable_quanto_int8_kernel
    from shared.qtypes.nvfp4 import _dequantize_nvfp4_weight
    maybe_enable_quanto_int8_kernel()

    cases = [
        (app / "ckpts/minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "blocks.0.attn.out_proj"),
        (app / "ckpts/minimax_h3/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "model.layers.0.mlp.down_proj"),
        (app / "ckpts/MiniMax_Music3_transformer_int8_convrot.safetensors", "transformer_blocks.0.attn.to_k"),
    ]
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": []}
    with tempfile.TemporaryDirectory(dir=app / "ckpts", ignore_cleanup_errors=True) as tmp:
        for number, (path, base) in enumerate(cases):
            print(f"[AMD validation] {base}", flush=True)
            with safe_open(str(path), framework="pt", device="cpu") as reader:
                config_tensor = reader.get_tensor(base + ".comfy_quant")
                config = json.loads(bytes(config_tensor.tolist()))
                q = reader.get_slice(base + ".weight")[:128]
                scale = reader.get_tensor(base + ".weight_scale")
                pre = reader.get_tensor(base + ".pre_quant_scale") if base + ".pre_quant_scale" in reader.keys() else None
                scale2 = reader.get_tensor(base + ".weight_scale_2") if base + ".weight_scale_2" in reader.keys() else None
            fixture = {"layer.weight": q, "layer.comfy_quant": config_tensor}
            if config["format"] == "nvfp4":
                # The first 128 rows form one complete NVFP4 scale tile.
                fixture["layer.weight_scale"] = scale[:128].contiguous()
                fixture["layer.weight_scale_2"] = scale2
                fixture["layer.pre_quant_scale"] = pre
                reference = _dequantize_nvfp4_weight(q, scale[:128], torch.tensor(1.), scale2,
                                                     torch.float32, torch.device("cpu"), layout="tensorcore")
                reference *= pre.float()
            else:
                fixture["layer.weight_scale"] = scale if scale.numel() == 1 else scale[:128].contiguous()
                reference = q.float() * fixture["layer.weight_scale"]
                if config.get("convrot"):
                    features = reference.shape[1]
                    # Validate the observable operation using rotated activations.
                    x = torch.randn(8, features)
                    rotated_output = ((x.reshape(8, -1, 256) @ regular_hadamard(256)).reshape(8, features)
                                      @ reference.T)
                    reference = (reference.reshape(128, -1, 256) @ regular_hadamard(256)).reshape(128, features)
                    torch.testing.assert_close(x @ reference.T, rotated_output, rtol=1e-4, atol=1e-4)
            source = Path(tmp) / f"layer{number}.safetensors"
            save_file(fixture, str(source))
            for dtype in ("fp16", "bf16", "int8"):
                destination = Path(tmp) / f"layer{number}_{dtype}.safetensors"
                convert_file(source, destination, dtype, chunk_mb=1, progress=lambda _: None)
                with init_empty_weights():
                    class LayerModel(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.layer = torch.nn.Linear(reference.shape[1], 128, bias=False, dtype=torch.bfloat16)
                    model = LayerModel()
                offload.load_model_data(model, str(destination), default_dtype=torch.bfloat16,
                                        writable_tensors=False, verboseLevel=0)
                model.eval().requires_grad_(False).to("cuda")
                torch.manual_seed(17)
                x = torch.randn(8, reference.shape[1], device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    expected = torch.nn.functional.linear(x, reference.to(device="cuda", dtype=torch.bfloat16))
                    actual = model.layer(x)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(10):
                        actual = model.layer(x)
                    torch.cuda.synchronize()
                error = (actual.float() - expected.float()).norm() / expected.float().norm()
                item = {"source": str(path), "layer": base, "dtype": dtype,
                        "relative_output_error": error.item(),
                        "warm_ms_per_forward": (time.perf_counter() - start) * 100,
                        "output_device": str(actual.device), "weight_type": type(model.layer.weight).__name__}
                report["cases"].append(item)
                print(json.dumps(item), flush=True)
                if error.item() > (.04 if dtype == "int8" else .02):
                    raise AssertionError(f"Numerical error too large: {item}")
                del model, x, actual, expected
                torch.cuda.empty_cache()
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

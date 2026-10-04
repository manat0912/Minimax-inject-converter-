"""Opt-in MiniMax file-conversion injection for an existing Maestro process.

Use --launch to run launch.py with the injection. No upstream source is edited.
"""
from __future__ import annotations

import argparse
import functools
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import runpy
import socket
import sys
import time

TARGET_MODULE = "models.minimax_h3.minimax_h3_main"


def converted_file(filename, dtype):
    # A persistent .pth hook runs before sitecustomize. Keep torch imports
    # out of startup so the application's compatibility fixes run first.
    if __package__:
        from .converter import convert_file, output_name
    else:
        from converter import convert_file, output_name
    return convert_file(filename, output_name(filename, dtype), dtype)


def patch_module(module, dtype="int8", fast_memory=True):
    if getattr(module, "_amd_file_converter_installed", False):
        return
    original_transformer = module._load_transformer
    original_conditioner = module._load_conditioner
    original_video_vae = module._load_video_vae
    original_audio_vae = module._load_audio_vae

    def timed(instance, method_name, label):
        if os.environ.get("MINIMAX_AMD_TIMING") != "1":
            return instance
        original = getattr(instance, method_name)
        @functools.wraps(original)
        def measure(*args, **kwargs):
            import torch
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            print(f"[AMD timing] {label}: {time.perf_counter()-started:.3f}s", flush=True)
            return result
        setattr(instance, method_name, measure)
        return instance
    if fast_memory:
        original_profile = module.offload.profile

        @functools.wraps(original_profile)
        def profile(pipe, *args, **kwargs):
            transformer = pipe.get("transformer") if isinstance(pipe, dict) else None
            if type(transformer).__name__ == "MiniMaxH3Transformer":
                import torch
                total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                # Each stage runs separately under MMGP. Keep RAM pageable;
                # the 32GB PC cannot afford to pin a 50GB model stack.
                kwargs["pinnedMemory"] = False
                kwargs["partialPinning"] = False
                budgets = dict(kwargs.get("budgets") or {})
                budgets.update({
                    "transformer": max(1024, min(24576, total_mb - 8192)),
                    "text_encoder": max(1024, min(27648, total_mb - 5120)),
                    "vae": max(1024, min(8192, total_mb - 8192)),
                    "vision_encoder": 2048, "audio_vae": 1024,
                })
                kwargs["budgets"] = budgets
                # The generic job planner caps every stage at the transformer's
                # residency fraction, preventing the 25GB encoder from fitting
                # even though it runs separately. Explicit budgets reserve 8GB
                # for transformer activations and 5GB for encoder activations.
                if total_mb >= 30 * 1024 and dtype == "int8":
                    kwargs["vram_safety_coefficient"] = max(float(kwargs.get("vram_safety_coefficient", .8)), .85)
                print(f"[AMD converter] Pageable RAM; stage VRAM budgets (MiB): {budgets}", flush=True)
            return original_profile(pipe, *args, **kwargs)

        module.offload.profile = profile

    @functools.wraps(original_transformer)
    def transformer(filename, *args, **kwargs):
        converted = converted_file(filename, dtype)
        return timed(original_transformer(str(converted), *args, **kwargs), "forward", "diffusion forward")

    @functools.wraps(original_conditioner)
    def conditioner(filename, *args, **kwargs):
        # GGUF has a separate portable loader and is intentionally left alone.
        if Path(filename).suffix.lower() != ".safetensors":
            return original_conditioner(filename, *args, **kwargs)
        converted = converted_file(filename, dtype)
        kwargs["variant"] = "amd_" + dtype
        return timed(original_conditioner(str(converted), *args, **kwargs), "forward", "prompt encoding")

    @functools.wraps(original_video_vae)
    def video_vae(*args, **kwargs):
        return timed(original_video_vae(*args, **kwargs), "decode", "video VAE decode")

    @functools.wraps(original_audio_vae)
    def audio_vae(*args, **kwargs):
        return timed(original_audio_vae(*args, **kwargs), "decode", "audio VAE decode")

    module._load_transformer = transformer
    module._load_conditioner = conditioner
    module._load_video_vae = video_vae
    module._load_audio_vae = audio_vae
    module._amd_file_converter_installed = True
    print(f"[AMD converter] MiniMax H3 file conversion injected ({dtype}); original files preserved", flush=True)


class ConversionLoader(importlib.abc.Loader):
    def __init__(self, original, dtype, fast_memory):
        self.original, self.dtype, self.fast_memory = original, dtype, fast_memory

    def create_module(self, spec):
        return self.original.create_module(spec) if hasattr(self.original, "create_module") else None

    def exec_module(self, module):
        self.original.exec_module(module)
        patch_module(module, self.dtype, self.fast_memory)


class ConversionFinder(importlib.abc.MetaPathFinder):
    def __init__(self, dtype, fast_memory):
        self.dtype, self.fast_memory = dtype, fast_memory

    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = ConversionLoader(spec.loader, self.dtype, self.fast_memory)
        return spec


def install(dtype="int8", fast_memory=True):
    if dtype not in {"int8", "fp16", "bf16"}:
        raise ValueError(dtype)
    if TARGET_MODULE in sys.modules:
        patch_module(sys.modules[TARGET_MODULE], dtype, fast_memory)
    elif not any(isinstance(finder, ConversionFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, ConversionFinder(dtype, fast_memory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="Existing Maestro/app directory")
    parser.add_argument("--dtype", choices=("int8", "fp16", "bf16"), default="int8")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--port", type=int, default=0, help="0 chooses an available loopback port")
    parser.add_argument("--stock-memory", action="store_true", help="Keep upstream memory settings for comparison")
    parser.add_argument("--timing", action="store_true", help="Synchronize the GPU to print stage timings (diagnostic overhead)")
    args, remaining = parser.parse_known_args()
    app = Path(args.app).resolve(strict=True)
    if not (app / "launch.py").is_file():
        parser.error("--app must point to Maestro's app folder containing launch.py")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    runtime_bin = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    os.environ["PATH"] = str(runtime_bin) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONUTF8"] = "1"
    if args.timing:
        os.environ["MINIMAX_AMD_TIMING"] = "1"
    install(args.dtype, fast_memory=not args.stock_memory)
    if not args.port:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            args.port = reservation.getsockname()[1]
    os.environ["SERVER_PORT"] = str(args.port)
    for key, value in {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "TORCH_BLAS_PREFER_HIPBLASLT": "0", "ROCBLAS_USE_HIPBLASLT": "0",
        "HSA_ENABLE_SDMA": "0", "MIOPEN_FIND_MODE": "FAST",
    }.items():
        os.environ.setdefault(key, value)
    print(f"[AMD converter] Prototype URL: http://127.0.0.1:{args.port}", flush=True)
    sys.path.insert(0, str(app))
    os.chdir(app)
    if args.launch:
        sys.argv = [str(app / "launch.py"), *remaining]
        runpy.run_path(str(app / "launch.py"), run_name="__main__")
    else:
        parser.error("Specify --launch, or import install() from your test harness")


if __name__ == "__main__":
    main()

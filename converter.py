"""Convert Comfy quantized safetensors to ordinary floating weights, in chunks.

No CUDA extensions, ComfyUI imports, pickle, or whole-checkpoint allocation.
The original checkpoint is never modified. See README.md for supported layouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
import uuid

import torch
from safetensors import safe_open

VERSION = "0.2.0"
TARGETS = {"fp16": torch.float16, "bf16": torch.bfloat16, "int8": torch.int8}
SUPPORTED = {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "nvfp4"}
AUX = ("comfy_quant", "weight_scale", "scale_weight", "weight_scale_2",
       "input_scale", "pre_quant_scale")
FP4_VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)


def read_header(path):
    with open(path, "rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Not a safetensors file")
        size = struct.unpack("<Q", size_bytes)[0]
        if not 2 <= size <= 100_000_000:
            raise ValueError("Invalid safetensors header size")
        return json.loads(stream.read(size))


def regular_hadamard(size, device="cpu"):
    # ConvRot uses this symmetric, regular Hadamard, not the Sylvester H2 basis.
    if size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError(f"ConvRot group size must be a power of four: {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                       [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h * size ** -.5


def unswizzle_scales(scale, rows, columns):
    padded_rows = (rows + 127) // 128 * 128
    groups = columns // 16
    padded_groups = (groups + 3) // 4 * 4
    if scale.numel() != padded_rows * padded_groups:
        raise ValueError(f"Invalid NVFP4 block scales: {tuple(scale.shape)} for {(rows, columns)}")
    return (scale.float().reshape(padded_rows // 128, padded_groups // 4, 32, 4, 4)
            .permute(0, 3, 2, 1, 4).reshape(padded_rows, padded_groups)[:rows, :groups])


def decode_chunk(weight, config, scale, *, tensor_scale=None, pre_scale=None, hadamard=None):
    """Decode in FP32, fold AWQ input scales, then let the caller choose storage."""
    fmt = config["format"]
    if fmt == "nvfp4":
        lut = torch.tensor(FP4_VALUES, device=weight.device)
        # Comfy tensorcore checkpoint order: high nibble, then low nibble.
        indices = torch.stack((weight >> 4, weight & 15), dim=-1).long()
        out = lut[indices].reshape(weight.shape[0], -1)
        out = (out.reshape(weight.shape[0], -1, 16)
               * (scale.float() * tensor_scale.float()).unsqueeze(-1)).reshape(weight.shape[0], -1)
    else:
        out = weight.float() * scale.float()
    if hadamard is not None:
        out = (out.reshape(out.shape[0], -1, hadamard.shape[0]) @ hadamard).reshape(out.shape)
    if pre_scale is not None:
        # Linear(x * pre_scale, W) == Linear(x, W * pre_scale[None, :]).
        out = out * pre_scale.float().reshape(1, -1)
    return out


def make_plan(path, dtype="bf16"):
    if dtype not in TARGETS:
        raise ValueError(f"Unsupported output dtype: {dtype}")
    source = Path(path).resolve(strict=True)
    header = read_header(source)
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    configs, consumed = {}, set()
    with safe_open(str(source), framework="pt", device="cpu") as tensors:
        for key in entries:
            if not key.endswith(".comfy_quant"):
                continue
            base = key[:-len(".comfy_quant")]
            config = json.loads(bytes(tensors.get_tensor(key).tolist()))
            if config.get("format") not in SUPPORTED:
                raise ValueError(f"Unsupported quantization at {base}: {config}")
            unknown = set(config) - {"format", "full_precision_matrix_mult", "convrot", "convrot_groupsize", "params"}
            if unknown:
                raise ValueError(f"Unknown quantization options at {base}: {unknown}")
            params = config.get("params", {})
            if set(params) - {"convrot", "convrot_groupsize"}:
                raise ValueError(f"Unknown quantization parameters at {base}: {params}")
            config = {**params, **config}
            weight_key = base + ".weight"
            if weight_key not in entries or base + ".weight_scale" not in entries:
                raise ValueError(f"Missing weight or scale at {base}")
            item = entries[weight_key]
            expected = {"int8_tensorwise": "I8", "nvfp4": "U8",
                        "float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2"}[config["format"]]
            if item["dtype"] != expected:
                raise ValueError(f"Unexpected {item['dtype']} storage for {config['format']} at {base}")
            shape = list(item["shape"])
            if len(shape) != 2:
                raise ValueError(f"Only 2D quantized weights are supported: {base} {shape}")
            if config["format"] == "nvfp4":
                shape[1] *= 2
                if shape[1] % 16 or base + ".weight_scale_2" not in entries:
                    raise ValueError(f"Invalid NVFP4 shape or missing global scale at {base}")
                if config.get("convrot"):
                    raise ValueError("NVFP4 with ConvRot is not supported")
                unswizzle_scales(tensors.get_tensor(base + ".weight_scale"), *shape)
                if tensors.get_tensor(base + ".weight_scale_2").numel() != 1:
                    raise ValueError(f"NVFP4 global scale must be scalar: {base}")
            else:
                scale_shape = entries[base + ".weight_scale"]["shape"]
                if math.prod(scale_shape) not in {1, shape[0]}:
                    raise ValueError(f"Unsupported scale shape at {base}: {scale_shape}")
            if config.get("convrot"):
                group = int(config.get("convrot_groupsize", 256))
                regular_hadamard(group)
                if shape[1] % group:
                    raise ValueError(f"Invalid ConvRot dimensions at {base}")
            if base + ".pre_quant_scale" in entries:
                if math.prod(entries[base + ".pre_quant_scale"]["shape"]) != shape[1]:
                    raise ValueError(f"Invalid AWQ pre-scale at {base}")
            configs[weight_key] = {"base": base, "config": config, "shape": shape}
            consumed.update(base + "." + suffix for suffix in AUX if base + "." + suffix in entries)
    # Fail closed rather than cast unknown packed integer/FP8 weight formats.
    for key, item in entries.items():
        if key.endswith(".weight") and item["dtype"] in {"I8", "U8", "F8_E4M3", "F8_E5M2"} and key not in configs:
            raise ValueError(f"Quantized weight has no supported comfy_quant descriptor: {key}")
    output, offset = {}, 0
    int8_config = json.dumps({"format": "int8_tensorwise"}).encode()
    for key, item in entries.items():
        if key in consumed:
            continue
        shape = configs[key]["shape"] if key in configs else item["shape"]
        # Keep embeddings floating for ordinary nn.Embedding consumers. Linear
        # layers retain INT8 storage with a portable, per-row scale descriptor.
        to_int8 = key in configs and dtype == "int8" and not key.endswith("embed_tokens.weight")
        if key in configs:
            configs[key]["to_int8"] = to_int8
        length = math.prod(shape) * (1 if to_int8 else 2) if key in configs else item["data_offsets"][1] - item["data_offsets"][0]
        output[key] = {"dtype": ("I8" if to_int8 else "F16" if dtype == "fp16" else "BF16") if key in configs else item["dtype"],
                       "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
        if to_int8:
            base = configs[key]["base"]
            output[base + ".weight_scale"] = {"dtype": "F32", "shape": [shape[0], 1],
                                                "data_offsets": [offset, offset + shape[0] * 4]}
            offset += shape[0] * 4
            output[base + ".comfy_quant"] = {"dtype": "U8", "shape": [len(int8_config)],
                                               "data_offsets": [offset, offset + len(int8_config)]}
            offset += len(int8_config)
    stat = source.stat()
    identity = {"version": VERSION, "source": str(source), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "header": hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest(),
                "dtype": dtype}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Retain model metadata but remove directives that would re-enable quantization.
    metadata = {k: v for k, v in header.get("__metadata__", {}).items()
                if "quant" not in k.lower() and "convrot" not in k.lower()}
    metadata.update(amd_converter_version=VERSION, amd_converter_fingerprint=fingerprint,
                    amd_converter_source=str(source), amd_converter_dtype=dtype)
    output["__metadata__"] = metadata
    return {"source": source, "entries": entries, "configs": configs, "output": output,
            "bytes": offset, "fingerprint": fingerprint, "dtype": dtype}


def output_name(source, dtype="bf16"):
    source = Path(source)
    return source.with_name(source.stem + f"_amd_{dtype}.safetensors")


def cache_valid(destination, plan):
    try:
        actual = read_header(destination)
        if actual != plan["output"]:
            return False
        with open(destination, "rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
        if Path(destination).stat().st_size != 8 + header_size + plan["bytes"]:
            return False
        with safe_open(str(destination), framework="pt", device="cpu") as tensors:
            return set(tensors.keys()) == set(plan["output"]) - {"__metadata__"}
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False


@torch.inference_mode()
def convert_file(source, destination=None, dtype="bf16", chunk_mb=16, device="cpu", progress=print):
    if chunk_mb < 1:
        raise ValueError("chunk_mb must be positive")
    plan = make_plan(source, dtype)
    source = plan["source"]
    destination = Path(destination or output_name(source, dtype)).resolve()
    if source == destination:
        raise ValueError("The destination must differ from the original checkpoint")
    if destination.exists():
        if cache_valid(destination, plan):
            progress(f"[AMD converter] Cache hit: {destination}")
            return destination
        raise FileExistsError(f"Destination exists but is not a matching conversion: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < plan["bytes"] + 1024 ** 3:
        raise OSError(f"Need {plan['bytes'] / 1024**3:.2f} GiB plus 1 GiB free at {destination.parent}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU conversion requested but PyTorch cannot access a GPU")
    device = torch.device(device)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.partial")
    lock = destination.with_name(destination.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    start = time.monotonic()
    os.close(lock_fd)
    try:
        if destination.exists():
            if cache_valid(destination, plan):
                return destination
            raise FileExistsError(destination)
        encoded = json.dumps(plan["output"], separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with safe_open(str(source), framework="pt", device="cpu") as tensors, open(temporary, "xb") as stream:
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            output_keys = [k for k in plan["output"] if k != "__metadata__"]
            pending = {}
            for index, key in enumerate(output_keys):
                if key in pending:
                    stream.write(memoryview(pending.pop(key).reshape(-1).view(torch.uint8).numpy()))
                    continue
                item = plan["entries"][key]
                shape = item["shape"]
                spec = plan["configs"].get(key)
                scale = tensor_scale = pre = h = None
                if spec:
                    base, config = spec["base"], spec["config"]
                    scale = tensors.get_tensor(base + ".weight_scale")
                    if config["format"] == "nvfp4":
                        scale = unswizzle_scales(scale, *spec["shape"])
                        tensor_scale = tensors.get_tensor(base + ".weight_scale_2").float().to(device)
                    else:
                        scale = scale.float().reshape(-1, 1)
                    if base + ".pre_quant_scale" in plan["entries"]:
                        pre = tensors.get_tensor(base + ".pre_quant_scale").float().to(device)
                    if config.get("convrot"):
                        h = regular_hadamard(int(config.get("convrot_groupsize", 256)), device)
                rows = shape[0] if shape else 1
                width = math.prod(spec["shape"][1:] if spec else shape[1:])
                # Budget FP32 math, indices, output, and source buffers, not only the final FP16 tensor.
                row_bytes = max(1, width * (32 if spec and spec["config"]["format"] == "nvfp4" else 16))
                step = max(1, chunk_mb * 1024**2 // row_bytes)
                scales_out = []
                for first in range(0, rows, step):
                    last = min(rows, first + step)
                    value = tensors.get_slice(key)[first:last] if shape else tensors.get_tensor(key)
                    if spec:
                        value = value.to(device)
                        row_scale = scale if scale.shape[0] == 1 else scale[first:last]
                        value = decode_chunk(value, spec["config"], row_scale.to(device),
                                             tensor_scale=tensor_scale, pre_scale=pre, hadamard=h)
                        if not torch.isfinite(value).all().item():
                            raise ValueError(f"Non-finite decoded values at {key}")
                        if spec["to_int8"]:
                            new_scale = (value.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-30)
                            value = (value / new_scale).round().clamp(-127, 127).to(torch.int8)
                            scales_out.append(new_scale.cpu())
                        else:
                            value = value.to(torch.float16 if dtype == "fp16" else torch.bfloat16)
                        if not torch.isfinite(value).all().item():
                            raise ValueError(f"Non-finite {dtype} conversion at {key}; try bf16 if fp16 overflows")
                    raw = value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                    stream.write(memoryview(raw))
                if spec and spec["to_int8"]:
                    pending[base + ".weight_scale"] = torch.cat(scales_out)
                    pending[base + ".comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
                if spec or index % 50 == 0:
                    progress(f"[AMD converter] {index+1}/{len(output_keys)} {key}")
            stream.flush()
            os.fsync(stream.fileno())
        if not cache_valid(temporary, plan):
            raise RuntimeError("Converted checkpoint failed header/size validation")
        # Detect source replacement during the lengthy conversion.
        if make_plan(source, dtype)["fingerprint"] != plan["fingerprint"]:
            raise RuntimeError("Source checkpoint changed during conversion")
        os.rename(temporary, destination)
        progress(f"[AMD converter] Wrote {plan['bytes']/1024**3:.2f} GiB in {time.monotonic()-start:.1f}s: {destination}")
        return destination
    finally:
        temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output")
    parser.add_argument("--dtype", choices=TARGETS, default="bf16")
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.inspect:
        plan = make_plan(args.source, args.dtype)
        print(json.dumps({"source": str(plan["source"]), "quantized_layers": len(plan["configs"]),
                          "output_gib": round(plan["bytes"] / 1024**3, 3), "dtype": args.dtype,
                          "destination": str(args.output or output_name(args.source, args.dtype))}, indent=2))
    else:
        convert_file(args.source, args.output, args.dtype, args.chunk_mb, args.device)


if __name__ == "__main__":
    main()

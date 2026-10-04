# MiniMax AMD file converter

Converts Comfy-format quantized `.safetensors` files to portable INT8,
FP16, or BF16. Tested with the downloaded MiniMax H3 diffusion weights,
Qwen3-VL NVFP4/AWQ encoder, and MiniMax Music3 INT8 ConvRot layers on an
AMD Radeon AI PRO R9700 under Windows ROCm. A complete 20-step MiniMax H3
video generation has passed on that machine. Other GPUs, higher resolutions,
long clips, reference media, and quality equivalence to the original
quantization have not been validated.

## What conversion does

- Scaled FP8: applies the stored weight scale.
- Row-scaled INT8: applies the stored scale to each output row.
- INT8 ConvRot: also reverses the regular Hadamard rotation of the weights.
- Comfy NVFP4: decodes high-nibble-first E2M1 values, unswizzles the tiled
  block scales, applies the global scale, and folds AWQ input smoothing
  into the columns of the weight matrix.
- `int8` output: requantizes decoded linear weights with one FP32 scale
  per row, without ConvRot, AWQ input scaling, or NVFP4 kernels. Embeddings
  become BF16 for compatibility with ordinary embedding modules.
- `fp16`/`bf16` output: replaces quantized weights with ordinary floats.
  Existing unquantized tensors retain their precision (including FP32
  modulation tables and codec buffers).

INT8 output introduces another quantization step. Measure output quality
before adopting it. Floating conversion cannot recover precision lost
when the original file was quantized.

Only the above Comfy `comfy_quant` layouts are supported. This is not a
generic converter for GGUF, Quanto archives, GPTQ/AWQ integer packs,
Nunchaku, or every file with “int8” in its name. Unknown packed weights,
formats, options, and shapes fail explicitly.

## Use in the existing Maestro installation

Use Maestro's existing ROCm Python environment. Do not replace its torch
wheel with a regular PyPI torch install. Dependencies are `torch` and
`safetensors`, which Maestro already installs.

Pinokio's Maestro AMD installer fetches release `v0.2.0` as its last
installation stage and installs a venv-local startup hook. Update restores
that hook and keeps the same release pin. For a manual installation, run
from the Maestro AMD launcher root (the clone folder name is required):

```powershell
git clone --depth 1 --branch v0.2.0 https://github.com/manat0912/Minimax-inject-converter-.git Maestro/app/amd_extensions/minimax_amd
& .\Maestro\app\env-amd\Scripts\python.exe Maestro/app/amd_extensions/minimax_amd/install_injection.py --dtype int8
```

Restart Maestro, select the usual MiniMax H3 model and NVFP4 AWQ encoder,
and generate. Conversion runs before the model loader, once per source
file and target format. Model downloads remain Maestro's responsibility.

Standalone conversion and process-local injection are also available:

```powershell
# Inspect size/layout without writing output.
& .\Maestro\app\env-amd\Scripts\python.exe Maestro/app/amd_extensions/minimax_amd/converter.py Maestro/app/ckpts/minimax_h3_fl2va_pruned_fp8_scaled.safetensors --inspect --dtype int8

# Convert one file. The original remains intact.
& .\Maestro\app\env-amd\Scripts\python.exe -u Maestro/app/amd_extensions/minimax_amd/converter.py Maestro/app/ckpts/minimax_h3/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors --dtype int8

# Run Maestro with opt-in, process-local loader injection.
& .\Maestro\app\env-amd\Scripts\python.exe -u Maestro/app/amd_extensions/minimax_amd/maestro.py --app Maestro/app --dtype int8 --launch
```

The injected process prints its available loopback URL. Select the usual
MiniMax H3 model and NVFP4 AWQ encoder in that process's UI: the injection
converts the selected safetensors before loading, then uses the compatible
non-NVFP4 conditioner path. GGUF encoder selections keep their normal loader.
The persistent hook activates conversion in normal Pinokio launches after
a restart. The process-local command also works without installing a hook.

For INT8 generation, Maestro's existing MMGP custom INT8 handler and its
AMD-compatible Triton kernels provide inference. The optional memory policy
applies only to MiniMax H3: it disables RAM pinning and assigns separate
VRAM budgets to the transformer, text encoder, and VAEs. On a 32 GiB GPU
the budgets are up to 24 GiB, 27 GiB, and 8 GiB respectively. MMGP manages
stage offloading; these are not simultaneous allocations. Smaller GPUs
receive smaller budgets and will need more offloading. Use `--stock-memory`
to compare the existing memory policy.

For the INT8 prototype on GPUs with at least 30 GiB VRAM, the global
residency cap is raised to at least 85% so the separate text-encoder stage
can fit. Per-stage budgets still reserve activation space. This policy
needs validation for longer clips, larger resolutions, and reference media;
the current test workload is 640x640 at 124 frames.

Process-local injection edits neither Maestro source nor its environment;
closing that process removes the injection. The persistent installer creates
one owned `minimax_amd_converter.pth` file in the active app venv, without
editing `launch.py` or `sitecustomize.py`. Running the helper with `--uninstall`
removes that owned hook. The package folder must be named `minimax_amd` and
remain at its installed location. Other model loaders keep their behavior.

Stage timing is optional: add `--timing` to the standalone launch command,
or set `MINIMAX_AMD_TIMING=1` before starting Maestro. It synchronizes GPU
work to print timings and adds diagnostic overhead; normal launches leave
it disabled.

## Files, memory, and cache

On this machine, floating output would occupy **37.46 GiB** for the pruned
diffusion model and **47.97 GiB** for the truncated Qwen encoder. Portable
INT8 copies occupy approximately **19.53 GiB** and **25.28 GiB**.

Conversion writes row chunks (default 16 MiB work budget), not a whole
decoded checkpoint in RAM. NVFP4 block scales are processed per layer.
First-time conversion costs disk space and time; subsequent loads reuse
the converted file. Cache identity includes source path, size, modification
time, header hash, converter version, and target format. It does not hash
all source payload bytes. Output header/layout and file size are checked.

Output is published only after a temporary file is flushed and validated.
The source is never overwritten. Existing mismatched destinations are
rejected. Concurrent conversion to the same destination is rejected by an
exclusive lock. An interrupted process can leave its `.lock`/`.partial`
files; remove those only after verifying no conversion is running.

Full video generation still requires model memory, activation headroom,
and stage transfers. A file converter alone cannot promise faster sampling
or VAE decoding. FP16 overflow aborts conversion; use BF16 instead.

## Programmatic use

Python, from the launcher root:

```python
import sys
sys.path.insert(0, "Maestro/app/amd_extensions")
from minimax_amd.converter import convert_file
result = convert_file("input.safetensors", "output.safetensors", dtype="bf16")
```

The converter is a local CLI/library, not an HTTP service. JavaScript can
invoke it with the ROCm Python executable:

```javascript
const { execFileSync } = require("node:child_process");
execFileSync("Maestro/app/env-amd/Scripts/python.exe", [
  "Maestro/app/amd_extensions/minimax_amd/converter.py", "input.safetensors", "--dtype", "fp16"
], { stdio: "inherit" });
```

After launching the injected Maestro process, its existing generation API
can be called with Python, JavaScript, or curl. Replace `PORT` with the
printed port and use your chosen settings:

```bash
curl http://127.0.0.1:PORT/api/v1/generate -H "Content-Type: application/json" -d '{"model_type":"minimax_h3","prompt":"A red ball on a wooden table.","resolution":"864x480","video_length":124,"num_inference_steps":20,"minimax_h3_text_encoder":"nvfp4_awq"}'
```

```python
import requests
job = requests.post("http://127.0.0.1:PORT/api/v1/generate", json={
    "model_type": "minimax_h3", "prompt": "A red ball on a wooden table.",
    "resolution": "864x480", "video_length": 124, "num_inference_steps": 20,
    "minimax_h3_text_encoder": "nvfp4_awq"
}).json()
```

```javascript
const job = await fetch("http://127.0.0.1:PORT/api/v1/generate", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model_type: "minimax_h3",
    prompt: "A red ball on a wooden table.", resolution: "864x480",
    video_length: 124, num_inference_steps: 20,
    minimax_h3_text_encoder: "nvfp4_awq" })
}).then(response => response.json());
```

Poll `GET /api/v1/status/<job_id>` to observe completion and output files.

## Verification and performance

Local test: Radeon AI PRO R9700 (32 GiB VRAM), 32 GiB RAM, Windows,
PyTorch `2.14.0a0+rocm7.14.0a20260612`. MiniMax H3 pruned FP8 diffusion
and Qwen3-VL NVFP4/AWQ sources were converted to INT8. Conversion took
64.8s and 110.8s respectively; these costs were excluded from the generation
below because the converted files were cached.

At 640x640, 124 frames, 24 fps, 20 inference steps, seed 123, guidance 1,
and SDPA attention:

| Stage | Observed time |
|---|---:|
| Prompt encoding | 24.3s |
| Warm diffusion forward | approximately 13.4s per step |
| Video VAE decode (FP16) | 23.0s |
| Audio VAE decode (FP32) | 4.9s |
| Complete queue, including loading/offloading/export | 5m 57s |

The export was verified as H.264 video plus AAC audio, 124 frames, 5.17s.
Sampled frames showed a coherent ball/table/window scene and camera movement.
This is a functional visual smoke test, not a quantitative quality comparison
against an unconverted checkpoint. No universal speedup is claimed.

Performance changes keep linear weights INT8 under Maestro's existing Triton
handler, retain its FP16 video VAE, reuse file conversions, disable RAM
pinning, and allow whole stages to reside in VRAM on the tested 32 GiB GPU.

The same prompt, seed, resolution, frame count, and 20 steps also passed
with Maestro's bundled **Sol Engine** on this ROCm runtime. Warm diffusion
forwards took approximately **9.5s**, video VAE decode **23.6s**, and the
complete queue **4m 58s**. Sol produced a coherent scene with different
composition from SDPA. It is an approximate attention path, not a promise
of identical output. The runs had different process/model residency and
compilation state, so these are observations rather than a controlled
end-to-end speedup benchmark.

For the tested faster path, choose **Sol Engine** in **Override Attention
Mode** (available in the Classic UI), or pass `"override_attention":"sol"`
to Maestro's generation API. Start with 640x640, 124 frames, 20 steps,
guidance 1, and the NVFP4 AWQ encoder. Sol requires the host's supported
Triton backend; the converter does not install or replace GPU kernels.
SDPA remains the compatibility option. The injection does not silently
override your attention selection.

```powershell
& .\Maestro\app\env-amd\Scripts\python.exe Maestro/app/amd_extensions/minimax_amd/test_converter.py
& .\Maestro\app\env-amd\Scripts\python.exe Maestro/app/amd_extensions/minimax_amd/test_injection.py
& .\Maestro\app\env-amd\Scripts\python.exe -u Maestro/app/amd_extensions/minimax_amd/validate_runtime.py --app Maestro/app --report Maestro/app/amd_extensions/minimax_amd/runtime_validation.json
```

Tests cover scales, NVFP4 nibble and scale ordering, AWQ folding, ConvRot
linear equivalence, source preservation, caching, truncated output rejection,
unknown-format rejection, and failed FP16 overflow cleanup. Runtime validation
loads actual checkpoint slices through Maestro/MMGP and runs GPU forwards.
Slice timings are diagnostics, not full-model throughput benchmarks.

Eight converter tests and seven injection/installer tests pass. Nine actual
checkpoint-slice GPU comparisons pass across FP16, BF16, and INT8 targets.

The requested ComfyUI node is deferred while Maestro is the first target.
A node after a MODEL loader cannot fix a loader that already failed;
ComfyUI will also need a pre-load conversion entry point. This repository
contains the converter, Maestro injection, installer, and validation scripts;
it contains no Pinokio launcher or downloaded model weights.

## Format references

The layouts follow Comfy's primary implementations:

- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py
- https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/eager/quantization.py
- https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/float_utils.py
- https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py

This package distributes no model weights. Model licenses still apply to
downloaded and converted checkpoints.

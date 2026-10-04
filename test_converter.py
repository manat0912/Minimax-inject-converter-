"""Numerical, file-integrity, and cache tests. Run with the app's Python."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from converter import convert_file, decode_chunk, make_plan, regular_hadamard, unswizzle_scales


def descriptor(**config):
    return torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / "original.safetensors"
        self.dest = Path(self.temp.name) / "converted.safetensors"

    def tearDown(self):
        self.temp.cleanup()

    def convert(self, tensors, dtype="fp16"):
        save_file(tensors, str(self.source))
        before = hashlib.sha256(self.source.read_bytes()).digest()
        convert_file(self.source, self.dest, dtype, chunk_mb=1, progress=lambda _: None)
        self.assertEqual(before, hashlib.sha256(self.source.read_bytes()).digest())
        with safe_open(str(self.dest), framework="pt", device="cpu") as reader:
            return {key: reader.get_tensor(key) for key in reader.keys()}

    def test_scaled_fp8_and_precision_preservation(self):
        w = torch.tensor([[1., -2.], [3., 4.]]).to(torch.float8_e4m3fn)
        result = self.convert({"layer.weight": w, "layer.weight_scale": torch.tensor(.25),
                               "layer.input_scale": torch.tensor(2.),
                               "layer.comfy_quant": descriptor(format="float8_e4m3fn"),
                               "sensitive_buffer": torch.tensor([1.234567], dtype=torch.float32)})
        torch.testing.assert_close(result["layer.weight"], w.float().half() * .25)
        self.assertEqual(result["sensitive_buffer"].dtype, torch.float32)
        self.assertEqual(set(result), {"layer.weight", "sensitive_buffer"})

    def test_int8_row_scale(self):
        q = torch.tensor([[1, -127, 3], [127, 0, -4]], dtype=torch.int8)
        scale = torch.tensor([[.5], [.125]])
        result = self.convert({"x.weight": q, "x.weight_scale": scale,
                               "x.comfy_quant": descriptor(format="int8_tensorwise")})
        torch.testing.assert_close(result["x.weight"], (q.float() * scale).half())

    def test_convrot_preserves_linear_output(self):
        torch.manual_seed(42)
        q = torch.randint(-127, 128, (7, 256), dtype=torch.int8)
        scale = torch.rand(7, 1) / 127
        result = self.convert({"x.weight": q, "x.weight_scale": scale,
                               "x.comfy_quant": descriptor(format="int8_tensorwise", convrot=True, convrot_groupsize=256)})
        x = torch.randn(5, 256)
        reference = (x @ regular_hadamard(256)) @ (q.float() * scale).T
        torch.testing.assert_close(x @ result["x.weight"].float().T, reference, rtol=.01, atol=.003)

    def test_nvfp4_layout_and_awq_input_scale(self):
        # Distinct logical scales exercise every tiled row/group position.
        rows, groups = 128, 8
        logical = (torch.arange(rows * groups).reshape(rows, groups) % 4 + 1).float()
        blocked = torch.empty_like(logical)
        flat = blocked.reshape(-1)
        for r in range(rows):
            for g in range(groups):
                position = (((((r // 128) * (groups // 4) + g // 4) * 32 + r % 32) * 4 + (r % 128) // 32) * 4 + g % 4)
                flat[position] = logical[r, g]
        packed = torch.full((rows, groups * 8), 0x2D, dtype=torch.uint8) # [1.0, -3.0]
        pre = torch.arange(groups * 16).float() / 128 + .5
        result = self.convert({"x.weight": packed, "x.weight_scale": blocked.to(torch.float8_e4m3fn),
                               "x.weight_scale_2": torch.tensor(.125), "x.pre_quant_scale": pre,
                               "x.comfy_quant": descriptor(format="nvfp4", full_precision_matrix_mult=True)})
        expected = torch.tensor([1., -3.]).repeat(rows, groups * 8)
        expected *= logical.repeat_interleave(16, dim=1) * .125 * pre
        torch.testing.assert_close(result["x.weight"], expected.half())
        torch.testing.assert_close(unswizzle_scales(blocked, rows, groups * 16), logical)

    def test_portable_int8_and_embedding(self):
        torch.manual_seed(3)
        w = torch.randn(100, 64).to(torch.float8_e4m3fn)
        result = self.convert({"x.weight": w, "x.weight_scale": torch.tensor(.75),
                               "x.comfy_quant": descriptor(format="float8_e4m3fn"),
                               "model.embed_tokens.weight": torch.ones(5, 4, dtype=torch.int8),
                               "model.embed_tokens.weight_scale": torch.ones(5, 1) * .25,
                               "model.embed_tokens.comfy_quant": descriptor(format="int8_tensorwise")}, "int8")
        decoded = result["x.weight"].float() * result["x.weight_scale"]
        error = (decoded - w.float() * .75).abs()
        self.assertLess(error.max().item(), result["x.weight_scale"].max().item() * .501)
        self.assertEqual(result["model.embed_tokens.weight"].dtype, torch.bfloat16)
        self.assertNotIn("model.embed_tokens.comfy_quant", result)

    def test_cache_and_stale_output(self):
        self.convert({"x.weight": torch.ones(2, 2)})
        before = self.dest.stat().st_mtime_ns
        convert_file(self.source, self.dest, "fp16", progress=lambda _: None)
        self.assertEqual(before, self.dest.stat().st_mtime_ns)
        with self.dest.open("r+b") as stream:
            stream.truncate(self.dest.stat().st_size - 1)
        with self.assertRaises(FileExistsError):
            convert_file(self.source, self.dest, "fp16")

    def test_unknown_packed_weight_rejected(self):
        save_file({"x.weight": torch.ones(2, 2, dtype=torch.uint8)}, str(self.source))
        with self.assertRaises(ValueError):
            make_plan(self.source)
        self.assertFalse(self.dest.exists())

    def test_fp16_overflow_does_not_publish(self):
        save_file({"x.weight": torch.ones(2, 2).to(torch.float8_e4m3fn),
                   "x.weight_scale": torch.tensor(1e8),
                   "x.comfy_quant": descriptor(format="float8_e4m3fn")}, str(self.source))
        with self.assertRaises(ValueError):
            convert_file(self.source, self.dest, "fp16", progress=lambda _: None)
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(Path(self.temp.name).glob("*.partial")), [])
        self.assertEqual(list(Path(self.temp.name).glob("*.lock")), [])


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()

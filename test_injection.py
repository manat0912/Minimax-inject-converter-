"""Check the process-local injection and its MiniMax-only memory scope."""
from types import SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import maestro
import install_injection


class InjectionTests(unittest.TestCase):
    def module(self):
        self.seen = []
        def profile(pipe, *args, **kwargs):
            self.seen.append(kwargs)
            return kwargs
        return SimpleNamespace(
            _load_transformer=lambda *a, **k: None,
            _load_conditioner=lambda *a, **k: (a, k),
            _load_video_vae=lambda *a, **k: None,
            _load_audio_vae=lambda *a, **k: None,
            offload=SimpleNamespace(profile=profile),
        )

    def test_other_models_keep_their_memory_policy(self):
        module = self.module()
        maestro.patch_module(module)
        result = module.offload.profile({"transformer": object()}, pinnedMemory=True, budgets={"*": 1234})
        self.assertEqual(result, {"pinnedMemory": True, "budgets": {"*": 1234}})

    def test_minimax_budgets_and_pinning(self):
        module = self.module()
        transformer = type("MiniMaxH3Transformer", (), {})()
        maestro.patch_module(module)
        with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=32 * 1024**3)):
            result = module.offload.profile({"transformer": transformer}, pinnedMemory=True,
                                            budgets={"*": 1000}, vram_safety_coefficient=.56)
        self.assertFalse(result["pinnedMemory"])
        self.assertEqual(result["budgets"]["transformer"], 24576)
        self.assertEqual(result["budgets"]["text_encoder"], 27648)
        self.assertEqual(result["vram_safety_coefficient"], .85)

    def test_gguf_keeps_its_original_loader(self):
        module = self.module()
        maestro.patch_module(module)
        args, kwargs = module._load_conditioner("model.gguf", "assets", variant="gguf_q4_k_m")
        self.assertEqual(args, ("model.gguf", "assets"))
        self.assertEqual(kwargs, {"variant": "gguf_q4_k_m"})

    def test_install_does_not_patch_twice(self):
        module = self.module()
        maestro.patch_module(module)
        previous = module._load_transformer
        maestro.patch_module(module)
        self.assertIs(previous, module._load_transformer)

    def test_repeated_install_keeps_one_import_hook(self):
        with patch.object(sys, "meta_path", []), patch.dict(sys.modules):
            sys.modules.pop(maestro.TARGET_MODULE, None)
            maestro.install()
            maestro.install()
            self.assertEqual(len(sys.meta_path), 1)

    def test_owned_startup_hook_install_and_remove(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "prefix", "test-venv"), \
                 patch("install_injection.sysconfig.get_paths", return_value={"purelib": directory}), \
                 patch.object(sys, "argv", ["install_injection.py", "--dtype", "int8"]):
                install_injection.main()
                hook = Path(directory) / install_injection.NAME
                self.assertTrue(hook.read_text().startswith(install_injection.MARKER))
                # Check that the generated import is valid Python and resolves
                # the installed package, including Windows path quoting.
                compile(hook.read_text().splitlines()[1], str(hook), "exec")
                install_injection.main()
                sys.argv = ["install_injection.py", "--uninstall"]
                install_injection.main()
                self.assertFalse(hook.exists())

    def test_startup_hook_does_not_replace_unowned_file(self):
        with tempfile.TemporaryDirectory() as directory:
            hook = Path(directory) / install_injection.NAME
            hook.write_text("import existing_user_hook\n")
            with patch.object(sys, "prefix", "test-venv"), \
                 patch("install_injection.sysconfig.get_paths", return_value={"purelib": directory}), \
                 patch.object(sys, "argv", ["install_injection.py"]):
                with self.assertRaises(FileExistsError):
                    install_injection.main()
            self.assertEqual(hook.read_text(), "import existing_user_hook\n")


if __name__ == "__main__":
    unittest.main()

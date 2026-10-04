"""Install/remove an owned, venv-local MiniMax startup hook without editing Maestro."""
import argparse
from pathlib import Path
import sys
import sysconfig

MARKER = "# minimax-amd-converter owned startup hook v1\n"
NAME = "minimax_amd_converter.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("int8", "fp16", "bf16"), default="int8")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Run this with the target application's virtual-environment Python")
    package = Path(__file__).resolve().parent
    if package.name != "minimax_amd" or not (package / "__init__.py").is_file():
        parser.error("The converter package folder must be named minimax_amd")
    hook = Path(sysconfig.get_paths()["purelib"]) / NAME
    if hook.exists() and not hook.read_text(encoding="utf-8").startswith(MARKER):
        raise FileExistsError(f"Refusing to replace an unrecognized hook: {hook}")
    if args.uninstall:
        hook.unlink(missing_ok=True)
        print(f"Removed {hook}")
        return
    parent = str(package.parent)
    # A .pth executes only single lines beginning with import. Use a literal
    # path generated at install time; no user path is baked into the source.
    content = (MARKER + "import sys; sys.path.insert(0, " + repr(parent) + "); "
               "from minimax_amd.maestro import install; install(" + repr(args.dtype) + ")\n")
    hook.write_text(content, encoding="utf-8")
    print(f"Installed {hook}; restart Maestro to activate {args.dtype} file conversion")


if __name__ == "__main__":
    main()

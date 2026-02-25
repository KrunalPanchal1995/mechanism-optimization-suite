import os
import re
import glob
import shutil
import sys


def generate_module_name(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', filename)


def strip_python_abi_suffix(name: str) -> str:
    name = re.sub(r'\.cpython-\d+[a-zA-Z0-9_-]*$', '', name)
    name = re.sub(r'\.cp\d+[a-zA-Z0-9_-]*$', '', name)
    return name


def rename_extensions_to_so(build_lib: str, final_dir: str) -> None:
    build_lib = os.path.abspath(build_lib)
    final_dir = os.path.abspath(final_dir)
    os.makedirs(final_dir, exist_ok=True)

    # RECURSIVE search for extension modules
    patterns = [
        os.path.join(build_lib, "**", "*.pyd"),  # Windows
        os.path.join(build_lib, "**", "*.so"),   # Linux/macOS
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    if not files:
        print(f"No extension modules found in {build_lib} (searched recursively).")
        return

    for src in files:
        base = os.path.basename(src)
        stem, _ = os.path.splitext(base)

        stem = strip_python_abi_suffix(stem)
        module_name = generate_module_name(stem)

        dst = os.path.join(final_dir, module_name + ".so")

        if os.path.exists(dst):
            os.remove(dst)

        shutil.move(src, dst)
        print(f"Renamed {src} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_so.py <build_lib> <output_dir>")
        sys.exit(1)

    rename_extensions_to_so(sys.argv[1], sys.argv[2])
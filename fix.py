#!/usr/bin/env python3
"""
Automatic fix for _vvtk_core ABI mismatch errors.

Run this script whenever you see an ImportError like:

    ImportError: .../_vvtk_core*.so: undefined symbol: _ZN...

This typically happens after upgrading or reinstalling PyTorch.
The script removes stale .so files and rebuilds the C++ extension
against the currently installed PyTorch.

Usage:
    python fix.py
"""

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1. Remove stale .so files
    patterns = [
        os.path.join(ROOT, "_vvtk_core*.so"),
        os.path.join(ROOT, "build", "**", "_vvtk_core*.so"),
    ]
    removed = []
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            os.remove(path)
            removed.append(path)

    if removed:
        print(f"Removed {len(removed)} stale .so file(s):")
        for p in removed:
            print(f"  {p}")
    else:
        print("No stale .so files found.")

    # 2. Rebuild
    print("\nRebuilding _vvtk_core extension...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("\nBuild failed. Check the output above for errors.")
        sys.exit(1)

    # 3. Verify
    print("\nVerifying import...")
    verify = subprocess.run(
        [sys.executable, "-c", "import _vvtk_core; print('OK — _vvtk_core loaded successfully')"],
        cwd=ROOT,
    )
    if verify.returncode != 0:
        print("Import verification failed. You may need to reinstall:")
        print(f"  pip install -e {ROOT}")
        sys.exit(1)

    print("\nDone! The extension is rebuilt and working.")


if __name__ == "__main__":
    main()

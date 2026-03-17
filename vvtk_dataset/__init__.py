from .datasets import VVTKDataset

try:
    from .loader import VVTKDataLoader
except ImportError as _exc:
    import sys as _sys

    _msg = (
        "\n"
        "Failed to import the _vvtk_core C++ extension.\n"
        "\n"
        "Fix — rebuild from the repo root:\n"
        "\n"
        "    python setup.py build_ext --inplace\n"
        "\n"
        "If you upgraded PyTorch, run:  python fix.py\n"
        "\n"
        f"Original error: {_exc}\n"
    )
    print(_msg, file=_sys.stderr)
    raise

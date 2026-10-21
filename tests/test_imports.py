from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import attention

ROOT = Path(__file__).resolve().parents[1]


def test_version_and_kernels():
    assert attention.__version__
    assert set(attention.list_kernels()) == {
        "sdpa",
        "mqa",
        "gqa",
        "fta",
        "psa",
        "moba",
        "qsa",
    }


def test_registry_docs_exist():
    for spec in attention.KERNELS.values():
        assert (ROOT / spec.docs).is_file(), spec.docs


def test_import_does_not_load_flash_attn():
    code = (
        "import sys, attention; "
        "assert 'flash_attn' not in sys.modules, 'flash_attn was imported eagerly'"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


def test_available_backends():
    assert attention.available_backends("sdpa") == ("reference", "torch")
    assert "reference" in attention.available_backends("moba")


def test_compatibility_aliases():
    assert callable(attention.scaled_dot_product_attention)
    assert attention.QSA is attention.QwenSparseAttention

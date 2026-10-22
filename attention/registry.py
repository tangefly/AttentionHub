"""Static metadata for every attention variant in the hub.

The registry is intentionally dependency-free so that it can be imported without
pulling in torch or any optional kernel backend.  It powers the README table and
can be used by tooling to discover kernels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    name: str
    full_name: str
    year: int
    category: str
    paper: str | None
    backends: tuple[str, ...]
    docs: str


KERNELS: dict[str, KernelSpec] = {
    "sdpa": KernelSpec(
        name="sdpa",
        full_name="Scaled Dot-Product Attention",
        year=2017,
        category="dense",
        paper="https://arxiv.org/abs/1706.03762",
        backends=("reference", "torch"),
        docs="docs/sdpa.md",
    ),
    "mqa": KernelSpec(
        name="mqa",
        full_name="Multi-Query Attention",
        year=2019,
        category="grouped",
        paper="https://arxiv.org/abs/1911.02150",
        backends=("reference",),
        docs="docs/mqa.md",
    ),
    "fta": KernelSpec(
        name="fta",
        full_name="Factorized Attention",
        year=2019,
        category="factorized",
        paper="https://arxiv.org/abs/1904.10509",
        backends=("reference",),
        docs="docs/fta.md",
    ),
    "psa": KernelSpec(
        name="psa",
        full_name="Predictive Sparse Attention",
        year=2022,
        category="sparse",
        paper="https://arxiv.org/abs/2110.11299",
        backends=("reference",),
        docs="docs/psa.md",
    ),
    "gqa": KernelSpec(
        name="gqa",
        full_name="Grouped-Query Attention",
        year=2023,
        category="grouped",
        paper="https://arxiv.org/abs/2305.13245",
        backends=("reference",),
        docs="docs/gqa.md",
    ),
    "moba": KernelSpec(
        name="moba",
        full_name="Mixture of Block Attention",
        year=2025,
        category="sparse",
        paper="https://arxiv.org/abs/2502.13189",
        backends=("reference", "flash"),
        docs="docs/moba.md",
    ),
    "qsa": KernelSpec(
        name="qsa",
        full_name="Qwen Sparse Attention",
        year=2026,
        category="sparse",
        paper=None,
        backends=("reference",),
        docs="docs/qsa.md",
    ),
    "dsa": KernelSpec(
        name="dsa",
        full_name="DeepSeek Sparse Attention",
        year=2025,
        category="sparse",
        paper="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp",
        backends=("reference",),
        docs="docs/dsa.md",
    ),
    "mla": KernelSpec(
        name="mla",
        full_name="Multi-Head Latent Attention",
        year=2024,
        category="compressed",
        paper="https://arxiv.org/abs/2405.04434",
        backends=("reference",),
        docs="docs/mla.md",
    ),
}


def get_spec(name: str) -> KernelSpec:
    try:
        return KERNELS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown kernel {name!r}; available: {', '.join(sorted(KERNELS))}"
        ) from exc


def markdown_table() -> str:
    """Render the support matrix as a Markdown table."""
    headers = [
        "Year",
        "Kernel",
        "Docs",
        "Paper",
        "Reference",
        "Torch",
        "Flash",
        "Triton",
        "CUDA",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":---:" for _ in headers) + " |",
    ]
    backend_columns = ("reference", "torch", "flash", "triton", "cuda")
    for spec in sorted(KERNELS.values(), key=lambda item: (item.year, item.name)):
        paper = f"[Link]({spec.paper})" if spec.paper else "-"
        row = [
            str(spec.year),
            f"{spec.full_name} ({spec.name.upper()})",
            f"[{spec.name.upper()}]({spec.docs})",
            paper,
        ]
        row.extend("✅" if backend in spec.backends else "❌" for backend in backend_columns)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    print(markdown_table())


if __name__ == "__main__":
    main()

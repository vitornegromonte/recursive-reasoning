"""Shared plotting utilities for MI experiments.

Provides consistent styling, color palette, and save helpers across
all experiment scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


# Color palette
COLORS = {
    # Model colors - colorblind-safe blue/orange pair
    "trm": "#0072B2",           # CB-safe blue (Wong palette)
    "transformer": "#E69F00",   # CB-safe orange (Wong palette)
    "trm_light": "#56B4E9",     # CB-safe sky blue
    "transformer_light": "#F0E442", # CB-safe yellow (use sparingly)
    
    # Semantic colors
    "correct": "#009E73",       # CB-safe green (Wong palette)
    "incorrect": "#D55E00",     # CB-safe vermillion, distinct from orange
    "neutral": "#999999",       # Grey unchanged
    
    # Accent for thresholds/markers across all figures
    "critical": "#CC79A7",      # CB-safe pink - for ρ=1 lines, step-5 markers
}

LABELS = {
    "trm": "TRM",
    "transformer": "Transformer",
}


def set_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, name: str, output_dir: str | Path) -> Path:
    """Save a figure as PNG.

    Args:
        fig: Matplotlib figure.
        name: Filename stem (without extension).
        output_dir: Output directory.

    Returns:
        Path to the saved PNG file.
    """
    out_dir = Path(output_dir)
    
    # Auto-label plots based on the output directory (e.g., n1k, n5k, n10k, random)
    model_label = out_dir.name
    if model_label in ["n1k", "n5k", "n10k", "random", "global"]:
        # Append label to filename
        name = f"{name}_{model_label}"
        
        # Append label to figure title
        label_str = f" [{model_label.upper()}]"
        assigned = False
        if getattr(fig, "_suptitle", None) and fig._suptitle.get_text():
            t = fig._suptitle.get_text()
            if label_str not in t:
                fig.suptitle(f"{t}{label_str}")
            assigned = True
        else:
            for ax in fig.axes:
                t = ax.get_title()
                if t and label_str not in t:
                    ax.set_title(f"{t}{label_str}")
                    assigned = True
                    
        # Fallback if no titles existed
        if not assigned:
            fig.text(0.5, 0.98, label_str.strip(), ha='center', va='top', fontsize=12, fontweight='bold')

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_json(data: Any, name: str, output_dir: str | Path) -> Path:
    """Save data as JSON.

    Args:
        data: JSON-serializable data.
        name: Filename stem (without extension).
        output_dir: Output directory.

    Returns:
        Path to the saved JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"

    # Convert numpy types to Python types
    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(data), f, indent=2)
    return path

"""Utilities: logging setup, progress display, and checkpoint I/O."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger.  verbose=True shows DEBUG messages."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(output_path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON checkpoint to disk.

    Writes to a temp file first, then renames — avoids corrupting the output
    if the process is killed mid-write.
    """
    path = Path(output_path)
    tmp_path = path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(path)


def load_checkpoint(output_path: str | Path) -> dict[str, Any] | None:
    """Load an existing checkpoint file, or return None if it doesn't exist."""
    path = Path(output_path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_completed_configs(checkpoint: dict[str, Any]) -> set[tuple[int, int]]:
    """Return the set of (i,j) configs already present in a checkpoint."""
    completed: set[tuple[int, int]] = set()
    for r in checkpoint.get("results", []):
        cfg = r.get("config")
        if cfg and len(cfg) == 2:
            completed.add((cfg[0], cfg[1]))
    return completed


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def make_progress_bar(total: int, desc: str = "Scanning") -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        unit="cfg",
        dynamic_ncols=True,
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Model parameter count
# ---------------------------------------------------------------------------

def count_params(model: Any) -> str:
    """Return a human-readable parameter count string like '3.09B'."""
    total = sum(p.numel() for p in model.parameters())
    if total >= 1e9:
        return f"{total / 1e9:.2f}B"
    if total >= 1e6:
        return f"{total / 1e6:.1f}M"
    return str(total)


# ---------------------------------------------------------------------------
# Post-processing: rankings and heatmap matrices
# ---------------------------------------------------------------------------

def compute_rankings(results: list[dict[str, Any]], top_n: int = 10) -> dict[str, Any]:
    """Return top-N configs by combined delta and per-probe delta (v1.1 format)."""
    non_baseline = [r for r in results if r["config"] != [0, 0]]

    def top_by(key_fn) -> list[list[int]]:
        sorted_r = sorted(non_baseline, key=key_fn, reverse=True)
        return [r["config"] for r in sorted_r[:top_n]]

    top_combined = top_by(lambda r: r.get("combined_delta", 0.0))

    # Collect all probe names present across results
    all_probe_names: set[str] = set()
    for r in non_baseline:
        all_probe_names.update(r.get("probe_deltas", {}).keys())

    probe_top = {
        name: top_by(lambda r, n=name: r.get("probe_deltas", {}).get(n, 0.0))
        for name in sorted(all_probe_names)
    }

    return {
        "top_combined": top_combined,
        "probe_top": probe_top,
    }


def build_heatmap_matrices(
    results: list[dict[str, Any]],
    num_layers: int,
) -> dict[str, Any]:
    """Build the 2D heatmap matrices for combined delta and per-probe deltas (v1.1 format).

    matrix[i][j] = delta for config (i,j), or None if not measured.
    The matrix is (num_layers+1) x (num_layers+1) to accommodate j up to N.
    """
    size = num_layers + 1

    # combined_delta matrix
    combined = [[None] * size for _ in range(size)]

    # Collect all probe names
    all_probe_names: set[str] = set()
    for r in results:
        all_probe_names.update(r.get("probe_deltas", {}).keys())

    probe_matrices: dict[str, list] = {
        name: [[None] * size for _ in range(size)]
        for name in all_probe_names
    }

    for r in results:
        ci, cj = r["config"]
        if 0 <= ci < size and 0 <= cj < size:
            combined[ci][cj] = r.get("combined_delta")
            for name, mat in probe_matrices.items():
                mat[ci][cj] = r.get("probe_deltas", {}).get(name)

    return {
        "combined_delta": {
            "description": (
                "2D array where matrix[i][j] = combined_delta for config (i,j). "
                "null if config not measured."
            ),
            "data": combined,
        },
        "probe_deltas": {
            name: {
                "description": (
                    f"2D array where matrix[i][j] = {name}_delta for config (i,j). "
                    "null if not measured."
                ),
                "data": mat,
            }
            for name, mat in probe_matrices.items()
        },
    }


# ---------------------------------------------------------------------------
# Upgrade helpers
# ---------------------------------------------------------------------------

def detect_missing_probes(
    checkpoint: dict[str, Any],
    active_probes: set[str],
) -> set[str]:
    """Return probe names in active_probes that are not present in the checkpoint results.

    Handles both v1.0 files (pubmedqa_score/eq_score fields) and v1.1 files
    (probe_scores dict).
    """
    results = checkpoint.get("results", [])
    if not results:
        return active_probes

    # Sample from non-baseline results to detect present probes
    sample = [r for r in results[:5] if r.get("config") != [0, 0]]
    if not sample:
        sample = results[:5]

    present: set[str] = set()
    for r in sample:
        probe_scores = dict(r.get("probe_scores", {}))
        # Also check v1.0 legacy fields
        if "pubmedqa_score" in r:
            probe_scores["pubmedqa"] = r["pubmedqa_score"]
        if "eq_score" in r:
            probe_scores["eq"] = r["eq_score"]
        present.update(probe_scores.keys())

    return active_probes - present


def migrate_v1_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1.0 result dict to v1.1 format in-place.

    Moves pubmedqa_score/eq_score into probe_scores,
    pubmedqa_delta/eq_delta into probe_deltas,
    renames combined_score/combined_delta to combined_score_v1/combined_delta_v1.
    """
    probe_scores: dict[str, float] = {}
    probe_deltas: dict[str, float] = {}

    if "pubmedqa_score" in result:
        probe_scores["pubmedqa"] = result.pop("pubmedqa_score")
    if "eq_score" in result:
        probe_scores["eq"] = result.pop("eq_score")
    if "pubmedqa_delta" in result:
        probe_deltas["pubmedqa"] = result.pop("pubmedqa_delta")
    if "eq_delta" in result:
        probe_deltas["eq"] = result.pop("eq_delta")

    old_combined = result.pop("combined_score", None)
    old_combined_delta = result.pop("combined_delta", None)

    result["probe_scores"] = probe_scores
    result["probe_deltas"] = probe_deltas
    result["combined_score_v1"] = old_combined
    result["combined_delta_v1"] = old_combined_delta
    # combined_score and combined_delta will be recomputed by the scanner

    return result

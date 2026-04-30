"""Main sweep orchestrator.

Coordinates model loading, config generation, per-config evaluation,
incremental checkpointing, and final output assembly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llmri.relayer import build_layer_path, get_duplicated_layers, generate_all_configs, get_num_layers
from llmri.utils import (
    utc_now_iso,
    save_checkpoint,
    load_checkpoint,
    get_completed_configs,
    make_progress_bar,
    count_params,
    compute_rankings,
    build_heatmap_matrices,
    detect_missing_probes,
    migrate_v1_result,
)

logger = logging.getLogger(__name__)

_DATASETS_DIR = Path(__file__).parent.parent / "datasets"


# ---------------------------------------------------------------------------
# Probe registry
# ---------------------------------------------------------------------------

@dataclass
class ProbeSpec:
    name: str
    default_dataset: Path
    load_fn: Callable[[Path], list[dict]]
    score_fn: Callable[[list[str], list[dict]], float]


def _build_registry() -> dict[str, ProbeSpec]:
    from llmri.scoring.pubmedqa_scorer import load_pubmedqa_dataset, score_pubmedqa_batch
    from llmri.scoring.eq_scorer import load_eq_dataset, score_eq_batch
    from llmri.scoring.boolq_scorer import load_boolq_dataset, score_boolq_batch
    from llmri.scoring.arc_scorer import load_arc_dataset, score_arc_batch
    from llmri.scoring.winogrande_scorer import load_winogrande_dataset, score_winogrande_batch
    from llmri.scoring.truthfulqa_scorer import load_truthfulqa_dataset, score_truthfulqa_batch

    return {
        "pubmedqa": ProbeSpec(
            name="pubmedqa",
            default_dataset=_DATASETS_DIR / "pubmedqa_16.json",
            load_fn=load_pubmedqa_dataset,
            score_fn=score_pubmedqa_batch,
        ),
        "eq": ProbeSpec(
            name="eq",
            default_dataset=_DATASETS_DIR / "eq_16.json",
            load_fn=load_eq_dataset,
            score_fn=score_eq_batch,
        ),
        "boolq": ProbeSpec(
            name="boolq",
            default_dataset=_DATASETS_DIR / "boolq_16.json",
            load_fn=load_boolq_dataset,
            score_fn=score_boolq_batch,
        ),
        "arc": ProbeSpec(
            name="arc",
            default_dataset=_DATASETS_DIR / "arc_16.json",
            load_fn=load_arc_dataset,
            score_fn=score_arc_batch,
        ),
        "winogrande": ProbeSpec(
            name="winogrande",
            default_dataset=_DATASETS_DIR / "winogrande_16.json",
            load_fn=load_winogrande_dataset,
            score_fn=score_winogrande_batch,
        ),
        "truthfulqa": ProbeSpec(
            name="truthfulqa",
            default_dataset=_DATASETS_DIR / "truthfulqa_16.json",
            load_fn=load_truthfulqa_dataset,
            score_fn=score_truthfulqa_batch,
        ),
    }


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def compute_combined(probe_scores: dict[str, float]) -> float:
    return sum(probe_scores.values()) / len(probe_scores) if probe_scores else 0.0


def compute_combined_v1(probe_scores: dict[str, float]) -> float | None:
    v1_scores = [probe_scores[k] for k in ("pubmedqa", "eq") if k in probe_scores]
    return sum(v1_scores) / len(v1_scores) if v1_scores else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_scan(
    model_path: str,
    output_path: str,
    backend: str,
    device: str,
    probes: set[str],
    probe_dataset_paths: dict[str, str],
    resume: bool,
    checkpoint_every: int,
    max_new_tokens: int,
    cache_dir: str | None = None,
    offline: bool = False,
    verbose: bool = False,
) -> None:
    """Run the full (i,j) sweep and write results to output_path."""
    PROBE_REGISTRY = _build_registry()

    # -----------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------
    probe_data: dict[str, list[dict]] = {}
    probe_scorers: dict[str, Callable] = {}

    for probe_name in probes:
        spec = PROBE_REGISTRY[probe_name]
        path = probe_dataset_paths.get(probe_name, str(spec.default_dataset))
        probe_data[probe_name] = spec.load_fn(path)
        probe_scorers[probe_name] = spec.score_fn
        logger.info(f"Loaded {len(probe_data[probe_name])} {probe_name} probes from {path}")

    # -----------------------------------------------------------------
    # 2. Load model config (weights not needed yet for architecture info)
    # -----------------------------------------------------------------
    if backend == "hf":
        from llmri.backends.hf_backend import load_model_config, load_model, detect_device
        if device == "auto":
            device = detect_device()
        model_cfg = load_model_config(
            model_path, cache_dir=cache_dir, local_files_only=offline
        )
    elif backend == "exllama":
        raise SystemExit("ExLlama backend not yet implemented. Use --backend hf.")
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    num_layers: int = get_num_layers(model_cfg)
    logger.info(f"Model has {num_layers} layers. Generating configs ...")

    # -----------------------------------------------------------------
    # 3. Generate sweep queue
    # -----------------------------------------------------------------
    all_configs = generate_all_configs(num_layers)
    total_configs = len(all_configs)
    logger.info(f"Total configs to sweep: {total_configs}")

    # -----------------------------------------------------------------
    # 4. Resume: load existing results and determine what to do
    # -----------------------------------------------------------------
    existing_results: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()
    upgrade_mode = False
    probes_to_run: set[str] = probes

    if resume:
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            existing_results = checkpoint.get("results", [])
            completed = get_completed_configs(checkpoint)
            missing_probes = detect_missing_probes(checkpoint, probes)

            if missing_probes:
                logger.info(
                    f"Upgrade mode: missing probes {missing_probes} detected. "
                    f"Will re-run all {total_configs} configs for missing probes only."
                )
                # Migrate all existing results to v1.1 format
                existing_results = [migrate_v1_result(r) for r in existing_results]
                probes_to_run = missing_probes
                upgrade_mode = True
            else:
                probes_to_run = probes
                logger.info(
                    f"Resuming scan: {len(completed)} configs already done, "
                    f"{total_configs - len(completed)} remaining."
                )
        else:
            logger.info("No existing checkpoint found; starting fresh.")

    # In upgrade mode we re-evaluate all configs (for missing probes); otherwise skip completed
    if upgrade_mode:
        pending_configs = list(all_configs)
    else:
        pending_configs = [c for c in all_configs if c not in completed]

    if not upgrade_mode and not pending_configs:
        logger.info("All configs already completed. Nothing to do.")
        return

    # -----------------------------------------------------------------
    # 5. Load model weights
    # -----------------------------------------------------------------
    model, tokenizer = load_model(
        model_path, device, cache_dir=cache_dir, local_files_only=offline
    )
    total_params_str = count_params(model)
    logger.info(f"Model loaded. Parameters: {total_params_str}")

    # -----------------------------------------------------------------
    # 6. Run baseline (0,0) — needed to compute deltas
    # -----------------------------------------------------------------
    from llmri.backends.hf_backend import evaluate_config

    baseline_probe_scores: dict[str, float] | None = None

    # Check if baseline already has all required probes in existing results
    for r in existing_results:
        if r["config"] == [0, 0]:
            existing_ps = r.get("probe_scores", {})
            if all(p in existing_ps for p in probes_to_run):
                baseline_probe_scores = dict(existing_ps)
                logger.info(f"Baseline loaded from checkpoint: {baseline_probe_scores}")
            break

    if baseline_probe_scores is None:
        logger.info("Evaluating baseline (0, 0) ...")
        new_baseline_scores = evaluate_config(
            model, tokenizer,
            i=0, j=0,
            num_layers=num_layers,
            probe_data={k: v for k, v in probe_data.items() if k in probes_to_run},
            probe_scorers=probe_scorers,
            max_new_tokens=max_new_tokens,
            device=device,
            active_probes=probes_to_run,
        )
        # Merge with any previously existing baseline scores
        existing_baseline: dict[str, float] = {}
        for r in existing_results:
            if r["config"] == [0, 0]:
                existing_baseline = dict(r.get("probe_scores", {}))
                break
        baseline_probe_scores = {**existing_baseline, **new_baseline_scores}

    baseline_combined = compute_combined(baseline_probe_scores)
    baseline_combined_v1 = compute_combined_v1(baseline_probe_scores)

    if verbose:
        logger.info(f"Baseline: {baseline_probe_scores} combined={baseline_combined:.4f}")

    # Build baseline entry
    baseline_entry: dict[str, Any] = {
        "config": [0, 0],
        "probe_scores": baseline_probe_scores,
        "probe_deltas": {k: 0.0 for k in baseline_probe_scores},
        "combined_score": round(baseline_combined, 6),
        "combined_score_v1": round(baseline_combined_v1, 6) if baseline_combined_v1 is not None else None,
        "combined_delta": 0.0,
        "combined_delta_v1": 0.0 if baseline_combined_v1 is not None else None,
    }

    # -----------------------------------------------------------------
    # 7. Prepare metadata skeleton (written to every checkpoint)
    # -----------------------------------------------------------------
    scan_start = utc_now_iso()

    probe_datasets_meta = {
        name: {
            "path": probe_dataset_paths.get(name, str(PROBE_REGISTRY[name].default_dataset)),
            "size": len(probe_data.get(name, [])),
        }
        for name in probes
    }

    metadata_base: dict[str, Any] = {
        "model_name": model_path,
        "model_type": model_cfg.model_type,
        "num_layers": num_layers,
        "hidden_size": getattr(model_cfg, "hidden_size", None),
        "num_attention_heads": getattr(model_cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(model_cfg, "num_key_value_heads", None),
        "total_params_base": total_params_str,
        "backend": backend,
        "device": device,
        "scan_start_utc": scan_start,
        "scan_end_utc": None,
        "scan_duration_seconds": None,
        "total_configs": total_configs,
        "completed_configs": len(completed),
        "probe_datasets": probe_datasets_meta,
        "max_new_tokens": max_new_tokens,
    }

    # -----------------------------------------------------------------
    # 8. Build a lookup for existing results (upgrade mode)
    # -----------------------------------------------------------------
    # In upgrade mode, index existing results by (i,j) so we can merge new scores
    existing_by_config: dict[tuple[int, int], dict[str, Any]] = {}
    if upgrade_mode:
        for r in existing_results:
            cfg = tuple(r["config"])
            existing_by_config[cfg] = r

    # -----------------------------------------------------------------
    # 9. Sweep loop
    # -----------------------------------------------------------------
    import time
    sweep_start_time = time.monotonic()

    # all_results holds the final merged list; start from existing (upgrade) or empty (fresh/resume)
    if upgrade_mode:
        all_results: list[dict[str, Any]] = []  # will be rebuilt from existing_by_config
    else:
        all_results = list(existing_results)

    done_count = 0 if upgrade_mode else len(completed)

    pbar = make_progress_bar(total=total_configs, desc="LLMRI")
    if not upgrade_mode:
        pbar.update(done_count)

    for cfg_idx, (i, j) in enumerate(pending_configs):
        if (i, j) == (0, 0):
            # Baseline already evaluated; in upgrade mode just update existing entry
            if upgrade_mode:
                all_results.append(baseline_entry)
            pbar.update(1)
            done_count += 1
            continue

        new_scores = evaluate_config(
            model, tokenizer,
            i=i, j=j,
            num_layers=num_layers,
            probe_data={k: v for k, v in probe_data.items() if k in probes_to_run},
            probe_scorers=probe_scorers,
            max_new_tokens=max_new_tokens,
            device=device,
            active_probes=probes_to_run,
        )

        # Merge with existing probe scores (upgrade mode)
        if upgrade_mode:
            existing_r = existing_by_config.get((i, j), {})
            merged_scores = {**existing_r.get("probe_scores", {}), **new_scores}
            combined_v1_existing = existing_r.get("combined_score_v1")
        else:
            merged_scores = new_scores
            combined_v1_existing = None

        combined = compute_combined(merged_scores)
        combined_v1 = compute_combined_v1(merged_scores)

        probe_deltas = {
            name: round(score - baseline_probe_scores.get(name, 0.0), 6)
            for name, score in merged_scores.items()
        }
        combined_delta = round(combined - baseline_combined, 6)
        combined_delta_v1 = (
            round(combined_v1 - baseline_combined_v1, 6)
            if combined_v1 is not None and baseline_combined_v1 is not None
            else None
        )

        layer_path = build_layer_path(i, j, num_layers)
        dup_layers = get_duplicated_layers(i, j)
        num_dup = len(dup_layers)
        param_increase_pct = round(num_dup / num_layers * 100, 2)

        result: dict[str, Any] = {
            "config": [i, j],
            "probe_scores": merged_scores,
            "probe_deltas": probe_deltas,
            "combined_score": round(combined, 6),
            "combined_score_v1": round(combined_v1, 6) if combined_v1 is not None else None,
            "combined_delta": combined_delta,
            "combined_delta_v1": combined_delta_v1,
            "duplicated_layers": dup_layers,
            "num_duplicated": num_dup,
            "layer_path": layer_path,
            "total_layers_in_path": len(layer_path),
            "param_increase_pct": param_increase_pct,
        }

        if upgrade_mode:
            # Preserve old combined_delta_v1 if we didn't recompute v1 probes
            if combined_delta_v1 is None and "combined_delta_v1" in existing_by_config.get((i, j), {}):
                result["combined_delta_v1"] = existing_by_config[(i, j)]["combined_delta_v1"]
            if combined_v1 is None and combined_v1_existing is not None:
                result["combined_score_v1"] = combined_v1_existing
            all_results.append(result)
        else:
            all_results.append(result)

        done_count += 1
        pbar.update(1)

        if verbose:
            logger.debug(
                f"({i:3d},{j:3d}) combined={combined:.4f} Δcombined={combined_delta:+.4f} "
                + " ".join(f"{k}={v:.4f}" for k, v in merged_scores.items())
            )

        # Checkpoint
        if done_count % checkpoint_every == 0:
            _write_output(
                output_path=output_path,
                metadata_base=metadata_base,
                baseline_entry=baseline_entry,
                all_results=all_results,
                num_layers=num_layers,
                done_count=done_count,
                total_configs=total_configs,
                sweep_start_time=sweep_start_time,
                final=False,
            )
            logger.debug(f"Checkpoint saved at {done_count} configs.")

    pbar.close()

    # -----------------------------------------------------------------
    # 10. Final output
    # -----------------------------------------------------------------
    _write_output(
        output_path=output_path,
        metadata_base=metadata_base,
        baseline_entry=baseline_entry,
        all_results=all_results,
        num_layers=num_layers,
        done_count=done_count,
        total_configs=total_configs,
        sweep_start_time=sweep_start_time,
        final=True,
    )
    logger.info(f"Scan complete. Results written to {output_path}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_output(
    output_path: str,
    metadata_base: dict[str, Any],
    baseline_entry: dict[str, Any],
    all_results: list[dict[str, Any]],
    num_layers: int,
    done_count: int,
    total_configs: int,
    sweep_start_time: float,
    final: bool,
) -> None:
    import time

    elapsed = time.monotonic() - sweep_start_time
    end_utc = utc_now_iso() if final else None

    metadata = {
        **metadata_base,
        "completed_configs": done_count,
        "scan_end_utc": end_utc,
        "scan_duration_seconds": round(elapsed, 1) if final else None,
    }

    # Include non-baseline results for rankings/heatmaps
    non_baseline = [r for r in all_results if r["config"] != [0, 0]]
    rankings = compute_rankings(non_baseline) if final else None
    heatmaps = build_heatmap_matrices(non_baseline, num_layers) if final else None

    output: dict[str, Any] = {
        "llmri_version": "1.1.0",
        "scan_metadata": metadata,
        "baseline": baseline_entry,
        "results": all_results,
    }
    if rankings:
        output["rankings"] = rankings
    if heatmaps:
        output["heatmap_matrices"] = heatmaps

    save_checkpoint(output_path, output)

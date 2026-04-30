"""LLMRI CLI entry point.

Commands:
  llmri scan           — run the (i,j) sweep on a model
  llmri convert        — convert RYS pickle files to LLMRI JSON
  llmri create-dataset — download and build the bundled probe datasets
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# Default dataset paths (bundled with the package)
_PKG_ROOT = Path(__file__).parent
_DATASETS_DIR = _PKG_ROOT.parent / "datasets"
_DEFAULT_PUBMEDQA = _DATASETS_DIR / "pubmedqa_16.json"
_DEFAULT_EQ = _DATASETS_DIR / "eq_16.json"
_DEFAULT_BOOLQ = _DATASETS_DIR / "boolq_16.json"
_DEFAULT_ARC = _DATASETS_DIR / "arc_16.json"
_DEFAULT_WINOGRANDE = _DATASETS_DIR / "winogrande_16.json"
_DEFAULT_TRUTHFULQA = _DATASETS_DIR / "truthfulqa_16.json"

ALL_PROBE_NAMES = ("pubmedqa", "eq", "boolq", "arc", "winogrande", "truthfulqa")


@click.group()
def cli() -> None:
    """LLMRI — RYS layer-duplication sweep tool."""


# ---------------------------------------------------------------------------
# llmri scan
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    help="HuggingFace model ID or path to a local checkpoint.",
)
@click.option(
    "--output", "-o",
    default="scan_results.json",
    show_default=True,
    help="Path to write the scan results JSON.",
)
@click.option(
    "--backend",
    type=click.Choice(["hf", "exllama"], case_sensitive=False),
    default="hf",
    show_default=True,
    help=(
        '"hf" uses HuggingFace transformers (works on Mac/MPS, CPU, CUDA). '
        '"exllama" requires an NVIDIA GPU and ExLlamaV2 installed manually.'
    ),
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help='Compute device: "cuda", "mps", "cpu", or "auto" (auto-detect).',
)
@click.option(
    "--probes",
    default="pubmedqa,eq",
    show_default=True,
    help=(
        'Comma-separated probe sets to run. Valid values: '
        'pubmedqa, eq, boolq, arc, winogrande, truthfulqa, all. '
        '"all" expands to all six probes.'
    ),
)
@click.option(
    "--pubmedqa-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom PubMedQA probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_PUBMEDQA.name}."
    ),
)
@click.option(
    "--eq-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom EQ-Bench probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_EQ.name}."
    ),
)
@click.option(
    "--boolq-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom BoolQ probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_BOOLQ.name}."
    ),
)
@click.option(
    "--arc-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom ARC-Challenge probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_ARC.name}."
    ),
)
@click.option(
    "--winogrande-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom WinoGrande probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_WINOGRANDE.name}."
    ),
)
@click.option(
    "--truthfulqa-dataset",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom TruthfulQA probe JSON file. "
        f"Defaults to the bundled {_DEFAULT_TRUTHFULQA.name}."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "Resume an interrupted scan. Reads --output and skips configs already "
        "recorded there (matched by (i,j) pair). When used with --probes all on "
        "a v1.0 file, detects missing probes and runs only those."
    ),
)
@click.option(
    "--max-new-tokens",
    default=16,
    show_default=True,
    help=(
        "Maximum tokens to generate per probe. PubMedQA needs 1-3; "
        "EQ-Bench needs more — the backend automatically uses max(this, 128) "
        "for EQ probes."
    ),
)
@click.option(
    "--checkpoint-every",
    default=20,
    show_default=True,
    help="Save a checkpoint to --output every N completed configs.",
)
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(file_okay=False),
    envvar="LLMRI_CACHE_DIR",
    help=(
        "Directory to cache downloaded model files. "
        "Defaults to the standard HuggingFace hub cache (~/.cache/huggingface/hub/). "
        "Can also be set via the LLMRI_CACHE_DIR environment variable."
    ),
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    envvar="LLMRI_OFFLINE",
    help=(
        "Run in offline mode: never hit the network, use only locally cached model files. "
        "Raises an error if the model has not been downloaded yet. "
        "Useful for repeated sweeps after the first download."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-config scores as they complete.",
)
def scan(
    model: str,
    output: str,
    backend: str,
    device: str,
    probes: str,
    pubmedqa_dataset: str | None,
    eq_dataset: str | None,
    boolq_dataset: str | None,
    arc_dataset: str | None,
    winogrande_dataset: str | None,
    truthfulqa_dataset: str | None,
    resume: bool,
    max_new_tokens: int,
    checkpoint_every: int,
    cache_dir: str | None,
    offline: bool,
    verbose: bool,
) -> None:
    """Run the RYS (i,j) layer-duplication sweep on MODEL.

    Produces a single JSON file at --output that the LLMRI Viewer can
    consume to render interactive heatmaps.

    Examples:

    \b
    # Minimal — scan a HuggingFace model (downloads to HF cache on first run)
    llmri scan --model Qwen/Qwen2.5-3B-Instruct

    \b
    # Run all six probes
    llmri scan --model Qwen/Qwen2.5-3B-Instruct --probes all

    \b
    # Resume an interrupted scan without re-downloading
    llmri scan --model Qwen/Qwen2.5-3B-Instruct --output scan.json --resume --offline

    \b
    # Upgrade a v1.0 scan file to v1.1 by adding the four new probes
    llmri scan --model Qwen/Qwen2.5-3B-Instruct --output scan.json --resume --probes all

    \b
    # PubMedQA probes only, force MPS device
    llmri scan --model /path/to/model --probes pubmedqa --device mps
    """
    from llmri.utils import setup_logging
    setup_logging(verbose=verbose)

    # Parse probe set — "all" expands to every registered probe
    raw_probes = [p.strip().lower() for p in probes.split(",")]
    if "all" in raw_probes:
        active_probes: set[str] = set(ALL_PROBE_NAMES)
    else:
        active_probes = set()
        for p in raw_probes:
            if p not in ALL_PROBE_NAMES:
                raise click.BadParameter(
                    f"Unknown probe set {p!r}. "
                    f"Choose from: {', '.join(ALL_PROBE_NAMES)}, all",
                    param_hint="--probes",
                )
            active_probes.add(p)

    # Resolve dataset paths (per-probe flags override defaults)
    _default_paths: dict[str, Path] = {
        "pubmedqa": _DEFAULT_PUBMEDQA,
        "eq": _DEFAULT_EQ,
        "boolq": _DEFAULT_BOOLQ,
        "arc": _DEFAULT_ARC,
        "winogrande": _DEFAULT_WINOGRANDE,
        "truthfulqa": _DEFAULT_TRUTHFULQA,
    }
    _flag_overrides: dict[str, str | None] = {
        "pubmedqa": pubmedqa_dataset,
        "eq": eq_dataset,
        "boolq": boolq_dataset,
        "arc": arc_dataset,
        "winogrande": winogrande_dataset,
        "truthfulqa": truthfulqa_dataset,
    }

    probe_dataset_paths: dict[str, str] = {}
    for probe_name in active_probes:
        override = _flag_overrides.get(probe_name)
        path = override if override else str(_default_paths[probe_name])
        probe_dataset_paths[probe_name] = path

    # Validate that dataset files exist for each active probe
    for probe_name, path in probe_dataset_paths.items():
        if not Path(path).exists():
            raise click.UsageError(
                f"{probe_name} dataset not found at {path!r}.\n"
                f"Run: llmri create-dataset --{probe_name}\n"
                f"Or supply a custom path with --{probe_name}-dataset."
            )

    from llmri.scanner import run_scan

    try:
        run_scan(
            model_path=model,
            output_path=output,
            backend=backend.lower(),
            device=device.lower(),
            probes=active_probes,
            probe_dataset_paths=probe_dataset_paths,
            resume=resume,
            checkpoint_every=checkpoint_every,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            offline=offline,
            verbose=verbose,
        )
    except KeyboardInterrupt:
        click.echo("\nScan interrupted. Results up to this point are saved.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# llmri convert
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--pkl-pubmedqa",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a RYS pubmedqa .pkl results file (or a math .pkl — see note).",
)
@click.option(
    "--pkl-eq",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a RYS eq .pkl results file.",
)
@click.option(
    "--model-name",
    required=True,
    help='Model name string to embed in the output, e.g. "Qwen/Qwen2.5-3B-Instruct".',
)
@click.option(
    "--num-layers",
    required=True,
    type=int,
    help="Number of transformer layers in the model.",
)
@click.option(
    "--output", "-o",
    default="scan_results.json",
    show_default=True,
    help="Path to write the converted LLMRI JSON.",
)
def convert(
    pkl_pubmedqa: str | None,
    pkl_eq: str | None,
    model_name: str,
    num_layers: int,
    output: str,
) -> None:
    """Convert RYS repo pickle files to LLMRI JSON format.

    The RYS repo outputs separate .pkl files for each probe type.  Use this
    command to combine them into a single scan_results.json for the viewer.

    Note: RYS math .pkl files store "math_score" rather than "pubmedqa_score".
    When converting, the math score is mapped to the pubmedqa_score field and
    a "legacy_math_score" flag is added to scan_metadata so the viewer knows
    the axis label should say "Math" rather than "PubMedQA".
    """
    import pickle
    import json
    from llmri.relayer import build_layer_path, get_duplicated_layers
    from llmri.utils import build_heatmap_matrices, compute_rankings, save_checkpoint, utc_now_iso

    if not pkl_pubmedqa and not pkl_eq:
        raise click.UsageError("Provide at least one of --pkl-pubmedqa or --pkl-eq.")

    # Load pickle files
    pubmedqa_data: dict = {}
    eq_data: dict = {}
    is_legacy_math = False

    if pkl_pubmedqa:
        with open(pkl_pubmedqa, "rb") as f:
            raw = pickle.load(f)
        # RYS pickles are dicts keyed by (i,j) tuple
        # Values may be floats or dicts — handle both
        for key, val in raw.items():
            score = val if isinstance(val, float) else val.get("score", val.get("math_score", 0.0))
            pubmedqa_data[key] = score
        # Detect if this is a math pkl (heuristic: check keys in raw)
        if any(
            "math" in str(k).lower() or "math_score" in (str(val) if not isinstance(val, float) else "")
            for k, val in raw.items()
        ):
            is_legacy_math = True

    if pkl_eq:
        with open(pkl_eq, "rb") as f:
            raw = pickle.load(f)
        for key, val in raw.items():
            score = val if isinstance(val, float) else val.get("score", val.get("eq_score", 0.0))
            eq_data[key] = score

    # Merge into results list
    all_keys: set = set(pubmedqa_data.keys()) | set(eq_data.keys())
    baseline_pubmedqa = pubmedqa_data.get((0, 0), 0.0)
    baseline_eq = eq_data.get((0, 0), 0.0)
    baseline_combined = (baseline_pubmedqa + baseline_eq) / 2.0

    results = []
    for key in sorted(all_keys, key=lambda k: (k[0], k[1])):
        i, j = key
        pmqa = pubmedqa_data.get(key, 0.0)
        eq = eq_data.get(key, 0.0)
        combined = (pmqa + eq) / 2.0
        layer_path = build_layer_path(i, j, num_layers)
        dup = get_duplicated_layers(i, j)
        results.append({
            "config": [i, j],
            "pubmedqa_score": pmqa,
            "eq_score": eq,
            "combined_score": combined,
            "pubmedqa_delta": round(pmqa - baseline_pubmedqa, 6),
            "eq_delta": round(eq - baseline_eq, 6),
            "combined_delta": round(combined - baseline_combined, 6),
            "duplicated_layers": dup,
            "num_duplicated": len(dup),
            "layer_path": layer_path,
            "total_layers_in_path": len(layer_path),
            "param_increase_pct": round(len(dup) / num_layers * 100, 2),
        })

    baseline_entry = {
        "config": [0, 0],
        "pubmedqa_score": baseline_pubmedqa,
        "eq_score": baseline_eq,
        "combined_score": baseline_combined,
    }

    metadata = {
        "model_name": model_name,
        "model_type": "unknown",
        "num_layers": num_layers,
        "hidden_size": 0,
        "num_attention_heads": 0,
        "num_key_value_heads": None,
        "total_params_base": "unknown",
        "backend": "rys-pkl-import",
        "device": "unknown",
        "scan_start_utc": utc_now_iso(),
        "scan_end_utc": utc_now_iso(),
        "scan_duration_seconds": None,
        "total_configs": len(results),
        "completed_configs": len(results),
        "pubmedqa_dataset": str(pkl_pubmedqa or ""),
        "pubmedqa_dataset_size": 0,
        "eq_dataset": str(pkl_eq or ""),
        "eq_dataset_size": 0,
        "max_new_tokens": 0,
        "legacy_math_score": is_legacy_math,
    }

    out = {
        "llmri_version": "1.0.0",
        "scan_metadata": metadata,
        "baseline": baseline_entry,
        "results": results,
        "rankings": compute_rankings(results),
        "heatmap_matrices": build_heatmap_matrices(results, num_layers),
    }

    save_checkpoint(output, out)
    click.echo(f"Converted {len(results)} configs → {output}")


# ---------------------------------------------------------------------------
# llmri create-dataset
# ---------------------------------------------------------------------------

@cli.command("create-dataset")
@click.option(
    "--pubmedqa",
    "create_pubmedqa",
    is_flag=True,
    default=False,
    help="Download and create datasets/pubmedqa_16.json and pubmedqa_100.json.",
)
@click.option(
    "--boolq",
    "create_boolq",
    is_flag=True,
    default=False,
    help="Download google/boolq and create datasets/boolq_16.json.",
)
@click.option(
    "--arc",
    "create_arc",
    is_flag=True,
    default=False,
    help="Download allenai/ai2_arc (ARC-Challenge) and create datasets/arc_16.json.",
)
@click.option(
    "--winogrande",
    "create_winogrande",
    is_flag=True,
    default=False,
    help="Download allenai/winogrande and create datasets/winogrande_16.json.",
)
@click.option(
    "--truthfulqa",
    "create_truthfulqa",
    is_flag=True,
    default=False,
    help="Download truthfulqa/truthful_qa and create datasets/truthfulqa_16.json.",
)
@click.option(
    "--all",
    "create_all",
    is_flag=True,
    default=False,
    help="Create all datasets (equivalent to --pubmedqa --boolq --arc --winogrande --truthfulqa).",
)
@click.option(
    "--output-dir",
    default=str(_DATASETS_DIR),
    show_default=True,
    help="Directory to write the dataset files.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for question selection.",
)
def create_dataset(
    create_pubmedqa: bool,
    create_boolq: bool,
    create_arc: bool,
    create_winogrande: bool,
    create_truthfulqa: bool,
    create_all: bool,
    output_dir: str,
    seed: int,
) -> None:
    """Download and create the bundled probe dataset files.

    Requires the 'datasets' extra: pip install llmri[datasets]

    This only needs to be run once after installation.  The eq_16.json file
    is already bundled; only pubmedqa_16.json needs to be downloaded.
    """
    if create_all:
        create_pubmedqa = True
        create_boolq = True
        create_arc = True
        create_winogrande = True
        create_truthfulqa = True

    if not any([create_pubmedqa, create_boolq, create_arc, create_winogrande, create_truthfulqa]):
        click.echo(
            "Nothing to do. Use one or more of: "
            "--pubmedqa, --boolq, --arc, --winogrande, --truthfulqa, --all"
        )
        return

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise click.UsageError(
            "The 'datasets' package is required. Install it with:\n"
            "  pip install llmri[datasets]\n"
            "  or: pip install datasets"
        )

    import random
    import json

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PubMedQA
    # ------------------------------------------------------------------
    if create_pubmedqa:
        click.echo("Downloading qiaojin/PubMedQA (pqa_labeled split) ...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

        yes_qs = [q for q in ds if q["final_decision"] == "yes"]
        no_qs = [q for q in ds if q["final_decision"] == "no"]
        maybe_qs = [q for q in ds if q["final_decision"] == "maybe"]

        random.seed(seed)

        probes_16 = (
            random.sample(yes_qs, 7)
            + random.sample(no_qs, 5)
            + random.sample(maybe_qs, 4)
        )
        random.shuffle(probes_16)

        probes_100 = (
            random.sample(yes_qs, 45)
            + random.sample(no_qs, 30)
            + random.sample(maybe_qs, 25)
        )
        random.shuffle(probes_100)

        def fmt_pubmedqa(probes: list) -> list[dict]:
            out = []
            for q in probes:
                context = " ".join(q["context"]["contexts"])
                out.append({
                    "id": str(q["pubid"]),
                    "prompt": (
                        f"Context: {context}\n\n"
                        f"Question: {q['question']}\n\n"
                        f"Answer with just yes, no, or maybe:"
                    ),
                    "answer": q["final_decision"],
                    "type": "pubmedqa",
                })
            return out

        p16 = out_dir / "pubmedqa_16.json"
        p100 = out_dir / "pubmedqa_100.json"

        json.dump(fmt_pubmedqa(probes_16), open(p16, "w"), indent=2)
        click.echo(f"Created {p16} ({len(probes_16)} questions)")

        json.dump(fmt_pubmedqa(probes_100), open(p100, "w"), indent=2)
        click.echo(f"Created {p100} ({len(probes_100)} questions)")

    # ------------------------------------------------------------------
    # BoolQ
    # ------------------------------------------------------------------
    if create_boolq:
        click.echo("Downloading google/boolq (validation split) ...")
        ds = load_dataset("google/boolq", split="validation")

        random.seed(seed)

        true_qs = [q for q in ds if q["answer"] is True]
        false_qs = [q for q in ds if q["answer"] is False]

        selected = random.sample(true_qs, 8) + random.sample(false_qs, 8)
        random.shuffle(selected)

        probes = []
        for idx, q in enumerate(selected):
            probes.append({
                "id": str(idx),
                "prompt": (
                    f"Passage: {q['passage']}\n\n"
                    f"Question: {q['question']}\n\n"
                    f"Answer with just yes or no:"
                ),
                "answer": "yes" if q["answer"] else "no",
                "type": "boolq",
            })

        p = out_dir / "boolq_16.json"
        json.dump(probes, open(p, "w"), indent=2)
        click.echo(f"Created {p} ({len(probes)} questions)")

    # ------------------------------------------------------------------
    # ARC-Challenge
    # ------------------------------------------------------------------
    if create_arc:
        click.echo("Downloading allenai/ai2_arc ARC-Challenge (test split) ...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

        random.seed(seed)

        LETTER_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}

        def normalize_arc(item: dict) -> dict | None:
            """Normalize choice labels to A/B/C/D, return None if not 4 choices."""
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            if len(labels) != 4:
                return None
            norm_labels = [LETTER_MAP.get(l, l) for l in labels]
            if set(norm_labels) != {"A", "B", "C", "D"}:
                return None
            choice_map = dict(zip(norm_labels, texts))
            answer = LETTER_MAP.get(item["answerKey"], item["answerKey"])
            if answer not in ("A", "B", "C", "D"):
                return None
            return {
                "id": item["id"],
                "question": item["question"],
                "choices": choice_map,
                "answer": answer,
            }

        normalized = [n for item in ds if (n := normalize_arc(item)) is not None]

        by_answer: dict[str, list] = {"A": [], "B": [], "C": [], "D": []}
        for item in normalized:
            by_answer[item["answer"]].append(item)

        selected = []
        for letter in ("A", "B", "C", "D"):
            selected.extend(random.sample(by_answer[letter], min(4, len(by_answer[letter]))))
        random.shuffle(selected)

        probes = []
        for item in selected:
            c = item["choices"]
            probes.append({
                "id": item["id"],
                "prompt": (
                    f"Question: {item['question']}\n\n"
                    f"A) {c['A']}\n"
                    f"B) {c['B']}\n"
                    f"C) {c['C']}\n"
                    f"D) {c['D']}\n\n"
                    f"Answer with just the letter A, B, C, or D:"
                ),
                "answer": item["answer"],
                "type": "arc",
            })

        p = out_dir / "arc_16.json"
        json.dump(probes, open(p, "w"), indent=2)
        click.echo(f"Created {p} ({len(probes)} questions)")

    # ------------------------------------------------------------------
    # WinoGrande
    # ------------------------------------------------------------------
    if create_winogrande:
        click.echo("Downloading allenai/winogrande winogrande_debiased (validation split) ...")
        ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")

        random.seed(seed)

        opt1_qs = [q for q in ds if q["answer"] == "1"]
        opt2_qs = [q for q in ds if q["answer"] == "2"]

        selected = random.sample(opt1_qs, 8) + random.sample(opt2_qs, 8)
        random.shuffle(selected)

        probes = []
        for idx, q in enumerate(selected):
            probes.append({
                "id": str(idx),
                "prompt": (
                    f"{q['sentence']}\n\n"
                    f"1) {q['option1']}\n"
                    f"2) {q['option2']}\n\n"
                    f"Answer with just 1 or 2:"
                ),
                "answer": q["answer"],
                "type": "winogrande",
            })

        p = out_dir / "winogrande_16.json"
        json.dump(probes, open(p, "w"), indent=2)
        click.echo(f"Created {p} ({len(probes)} questions)")

    # ------------------------------------------------------------------
    # TruthfulQA
    # ------------------------------------------------------------------
    if create_truthfulqa:
        click.echo("Downloading truthfulqa/truthful_qa multiple_choice (validation split) ...")
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

        random.seed(seed)

        # Try to stratify by category; fall back to random sample if no category field
        by_category: dict[str, list] = {}
        for item in ds:
            cat = item.get("category") or "Unknown"
            by_category.setdefault(cat, []).append(item)

        if len(by_category) > 1:
            # Stratified: sort categories by name for reproducibility, take top 16 by size
            sorted_cats = sorted(by_category.keys(), key=lambda c: (-len(by_category[c]), c))
            top_cats = sorted_cats[:16]
            selected = []
            for cat in top_cats:
                selected.append(random.choice(by_category[cat]))
            random.shuffle(selected)
        else:
            # No category field — plain random sample
            all_items = list(ds)
            selected = random.sample(all_items, 16)

        probes = []
        for idx, item in enumerate(selected):
            mc1 = item["mc1_targets"]
            choices = mc1["choices"]
            labels = mc1["labels"]

            # First entry with label=1 is correct; first with label=0 is best incorrect
            correct_idx = next((i for i, l in enumerate(labels) if l == 1), 0)
            incorrect_idx = next((i for i, l in enumerate(labels) if l == 0), 1)

            correct = choices[correct_idx]
            incorrect = choices[incorrect_idx]

            probes.append({
                "id": str(idx),
                "prompt": (
                    f"Question: {item['question']}\n\n"
                    f"A) {correct}\n"
                    f"B) {incorrect}\n\n"
                    f"Answer with just A or B:"
                ),
                "answer": "A",
                "type": "truthfulqa",
                "category": item.get("category") or "Unknown",
            })

        p = out_dir / "truthfulqa_16.json"
        json.dump(probes, open(p, "w"), indent=2)
        click.echo(f"Created {p} ({len(probes)} questions)")


if __name__ == "__main__":
    cli()

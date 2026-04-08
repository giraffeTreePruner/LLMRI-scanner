# LLMRI

LLMRI (Language Model — Model Relayering Imaging) runs exhaustive layer-duplication sweeps on transformer models to measure how duplicating different layer ranges affects model performance. For every valid `(i, j)` pair in an N-layer model, it constructs a temporary layer path where layers `i` through `j−1` execute twice, scores the resulting model on two probe sets, and records the results in a structured JSON file.

---

## Background

### RYS — Repeat Yourself Smarter

LLMRI is built on top of [RYS](https://github.com/dnhkng/RYS), a technique discovered by David Noel Ng (dnhkng) and described in two articles: [LLM Neuroanatomy](https://dnhkng.github.io/posts/rys/) and [LLM Neuroanatomy II](https://dnhkng.github.io/posts/rys-ii/).

The core observation is that you can improve a transformer's performance by duplicating a contiguous block of its middle layers and running inference through that block twice — no retraining, no weight copies, no merging. The repeated layers execute via pointer, so the VRAM cost is zero. Compute overhead is proportional to how many layers you add (e.g. repeating 7 layers in an 80-layer model costs ~9% more FLOPs per forward pass).

A configuration is expressed as `(i, j)`: layers `0` through `j−1` execute first, then layers `i` through `N−1` execute, so layers `i` through `j−1` run twice. The baseline `(0, 0)` is the unmodified model.

Applied to Qwen2-72B with configuration `(45, 52)`, RYS reached #1 on the HuggingFace Open LLM Leaderboard with a +2.61% average gain — including +8.16% on MATH Level 5 and +17.72% on MuSR. The model acquired no new knowledge and used no additional weights.

### Why it works: transformer neuroanatomy

Plotting the performance delta of every `(i, j)` pair as a 2D heatmap reveals a consistent three-phase anatomy across model families:

- **Encoding (early layers):** Rapidly normalize diverse surface forms — different languages, tokenizations, encodings — into a shared abstract representation. Disrupting these layers hurts performance.
- **Reasoning (middle layers):** Operate in a format-agnostic conceptual space. These are the layers that benefit from duplication because they function as complete *circuits* — multi-step reasoning pipelines that must execute in full to produce a coherent intermediate state.
- **Decoding (late layers):** Collapse the abstract representation back into output tokens. Duplication here provides little benefit.

A key mechanistic finding: duplicating a *single* layer produces little or no gain. The middle layers don't perform independent iterative refinement — they work as circuits. Repeating the full block gives the model a second complete reasoning pass on its own output.

This three-phase structure is confirmed cross-lingually: middle-layer representations are more similar for same-topic content in different languages than for different-topic content in the same language — confirming the reasoning phase is format-agnostic.

### RYS II: sweeping 2 million candidates

The original RYS paper found its configuration empirically. RYS II asked: what's the *optimal* configuration, and how does the search scale? Using Qwen3.5-27B as the target, it ran a multi-stage search:

1. Full `(i, j)` grid scan
2. Per-layer repeat-count sweep (2× through 8× for individual layers)
3. Beam search over multi-block compositions
4. An XGBoost surrogate model trained on 4,643 measured configurations, used to rank ~2 million candidates (Spearman ρ = 0.933 on held-out configs)
5. Final validation of the top candidates on expanded 120-question math and 139-scenario EQ benchmarks

The exhaustive search confirmed what the first article suggested: **simple contiguous mid-stack blocks dominate the efficiency frontier**. After evaluating 2 million candidates, the Pareto-optimal configurations were all straightforward `(i, j)` pairs clustered around layers 26–34. The single-layer-pair `(33, 34)` — duplicating just one layer — captured most of the EQ benefit (+0.0945) at only +1.56% overhead. The author called it "a free lunch, or at least a very cheap snack."

### Why LLMRI?

The heatmap output that both RYS articles center their analysis on is, literally, an image of the model's internal layer structure — a scan that reveals the three-phase anatomy the technique depends on. The articles frame this throughout in neuroanatomy language: circuits, phases, encoding and decoding regions, disruption deficits.

**LLMRI** (Large Language — Model Relayering Imaging) names both halves of what this tool does:

- **Relayering** is the exact technical operation: constructing a modified layer execution path without touching weights
- **Imaging** captures both the heatmap visualization output and the MRI metaphor — this tool produces scans of a model's internal structure the same way an MRI produces scans of biological tissue

The name deliberately echoes medical imaging because the methodology is the same: apply a probe, measure response, build a 2D map, interpret the anatomy.

---

## What it does

For a model with N layers, LLMRI evaluates N×(N+1)/2 + 1 configurations (including a no-duplication baseline). Each configuration is scored on:

- **PubMedQA** — biomedical yes/no/maybe question answering (accuracy)
- **EQ-Bench** — emotional intelligence dialogues (MAE-based score with confidence weighting)

Outputs include per-configuration scores, deltas from baseline, top-10 rankings by each metric, and 2D heatmap matrices ready for visualization.

---

## Features

- Full `(i, j)` sweep with no weight copies — uses shallow layer references to avoid memory duplication
- Resume interrupted scans with `--resume`; atomic checkpointing every N configs
- Pluggable inference backends (HuggingFace Transformers primary; ExLlama stub for future CUDA use)
- Auto-detects layer paths for Llama, Qwen, Mistral, GPT-2, Falcon, MPT, and variants
- Bundled 16-question probe sets for fast sweeps; 100-question PubMedQA set for post-sweep validation
- Legacy RYS pickle import via `llmri convert`
- Output includes rankings and heatmap matrices for immediate analysis

---

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/giraffeTreePruner/LLMRI-scanner.git
cd LLMRI-scanner
uv sync
```

For GPU inference, ensure a CUDA-enabled PyTorch build is installed before running `uv sync`.

To generate the bundled PubMedQA dataset from the HuggingFace hub (optional — already included):

```bash
uv run llmri create-dataset --pubmedqa
```

---

## Usage

### Run a sweep

```bash
uv run llmri scan \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output model_scans/my-scan.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | _(required)_ | HuggingFace model ID or local path |
| `--output` | `scan_results.json` | Output JSON file |
| `--backend` | `hf` | Inference backend (`hf`) |
| `--device` | auto-detected | `cuda`, `mps`, or `cpu` |
| `--probes` | `pubmedqa,eq` | Comma-separated probe sets to run |
| `--pubmedqa-dataset` | bundled | Path to PubMedQA dataset JSON |
| `--eq-dataset` | bundled | Path to EQ-Bench dataset JSON |
| `--resume` | off | Resume from an existing output file |
| `--checkpoint-every` | `20` | Save progress every N configs |

### Resume an interrupted scan

```bash
uv run llmri scan --model Qwen/Qwen2.5-3B-Instruct --output scan.json --resume
```

### Convert a legacy RYS pickle file

```bash
uv run llmri convert --pkl-pubmedqa rys_pubmedqa.pkl --model-name Qwen/Qwen2.5-3B-Instruct --num-layers 28
```

---

## Directory structure

```
LLMRI-scanner/
├── llmri/                      # Main package
│   ├── cli.py                  # Click CLI entry points
│   ├── scanner.py              # Sweep orchestrator
│   ├── relayer.py              # Layer path construction and model patching
│   ├── schema.py               # Pydantic output schema
│   ├── utils.py                # Logging, checkpointing, ranking, heatmap utilities
│   ├── backends/
│   │   ├── hf_backend.py       # HuggingFace Transformers backend
│   │   └── exllama_backend.py  # ExLlama backend (stub)
│   └── scoring/
│       ├── pubmedqa_scorer.py  # Accuracy scorer for PubMedQA
│       └── eq_scorer.py        # MAE-based scorer for EQ-Bench
├── datasets/
│   ├── manifest.json           # Dataset metadata
│   ├── eq_16.json              # 16 EQ-Bench scenarios (bundled)
│   ├── pubmedqa_16.json        # 16 PubMedQA questions (sweep set)
│   └── pubmedqa_100.json       # 100 PubMedQA questions (validation)
├── model_scans/                # Pre-computed scan outputs
│   ├── Qwen2-5-Instruct-3B.json
│   ├── llama3-2-Instruct-3B.json
│   └── llama3-1_8B_instruct.json
└── pyproject.toml
```

---

## Output format

Each scan produces a single JSON file conforming to the following structure:

```jsonc
{
  "llmri_version": "1.0.0",
  "scan_metadata": {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "num_layers": 36,
    "total_configs": 667,
    "completed_configs": 667,
    "scan_duration_seconds": 5280.0,
    // ...
  },
  "baseline": {
    "config": [0, 0],
    "pubmedqa_score": 0.25,
    "eq_score": 0.524,
    "combined_score": 0.387
  },
  "results": [
    {
      "config": [0, 1],          // (i, j): duplicate layers i..j-1
      "pubmedqa_score": 0.5,
      "eq_score": 0.524,
      "combined_score": 0.512,
      "pubmedqa_delta": 0.25,    // score - baseline
      "eq_delta": 0.0,
      "combined_delta": 0.125,
      "duplicated_layers": [0],
      "num_duplicated": 1,
      "layer_path": [0, 0, 1, 2, "..."],
      "total_layers_in_path": 37,
      "param_increase_pct": 2.78
    }
    // N*(N+1)/2 more entries
  ],
  "rankings": {
    "top_combined": [[3, 7], [2, 6], "..."],  // top-10 configs
    "top_pubmedqa": ["..."],
    "top_eq": ["..."]
  },
  "heatmap_matrices": {
    "combined_delta": {
      "description": "matrix[i][j] = combined_delta for config (i,j)",
      "data": [[null, 0.125, "..."], "..."]   // null where j <= i
    },
    "pubmedqa_delta": { "..." },
    "eq_delta": { "..." }
  }
}
```

**Score definitions:**

| Field | Range | Description |
|-------|-------|-------------|
| `pubmedqa_score` | 0–1 | Fraction of correct yes/no/maybe answers |
| `eq_score` | 0–1 | `1 - (MAE / 40)`, confidence-weighted toward 0.5 on parse failure |
| `combined_score` | 0–1 | Simple average of the two scores |
| `*_delta` | −1 to 1 | Score minus baseline |

**Layer path semantics:**

Config `(i, j)` produces path `[0 .. j-1] + [i .. N-1]`, so layers `i` through `j-1` execute twice. Config `(0, 0)` is the unmodified baseline.

---

## Tech stack

| Component | Library |
|-----------|---------|
| CLI | [Click](https://click.palletsprojects.com/) |
| Model inference | [Transformers](https://github.com/huggingface/transformers) |
| Deep learning | [PyTorch](https://pytorch.org/) |
| Schema validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Numerical ops | [NumPy](https://numpy.org/) |
| Progress display | [tqdm](https://github.com/tqdm/tqdm) |
| Build system | [Hatchling](https://hatch.pypa.io/) / [uv](https://github.com/astral-sh/uv) |

---

## Pre-computed scans

Three reference scans are included in `model_scans/`:

| File | Model | Configs |
|------|-------|---------|
| `Qwen2-5-Instruct-3B.json` | Qwen/Qwen2.5-3B-Instruct | 667 |
| `llama3-2-Instruct-3B.json` | meta-llama/Llama-3.2-3B-Instruct | 407 |
| `llama3-1_8B_instruct.json` | meta-llama/Llama-3.1-8B-Instruct | 529 |

---

## Based on

[dnhkng/RYS](https://github.com/dnhkng/RYS) — the original Repeat Yourself Smarter technique and sweep methodology.

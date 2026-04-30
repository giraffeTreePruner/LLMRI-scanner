"""Pydantic models for the LLMRI output JSON schema."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ScanMetadata(BaseModel):
    model_name: str
    model_type: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    total_params_base: str
    backend: str
    device: str
    scan_start_utc: str
    scan_end_utc: Optional[str] = None
    scan_duration_seconds: Optional[float] = None
    total_configs: int
    completed_configs: int
    max_new_tokens: int
    # v1.1 field: probe_datasets replaces the per-probe fields below
    probe_datasets: Optional[dict[str, dict]] = None
    # Legacy Optional fields for reading v1.0 files (not written in v1.1 output):
    pubmedqa_dataset: Optional[str] = None
    pubmedqa_dataset_size: Optional[int] = None
    eq_dataset: Optional[str] = None
    eq_dataset_size: Optional[int] = None


class BaselineResult(BaseModel):
    config: list[int]
    combined_score: float
    # v1.1 fields:
    probe_scores: Optional[dict[str, float]] = None
    combined_score_v1: Optional[float] = None
    # Legacy Optional fields for reading v1.0 files (not written in v1.1 output):
    pubmedqa_score: Optional[float] = None
    eq_score: Optional[float] = None


class ConfigResult(BaseModel):
    config: list[int]
    combined_score: float
    combined_delta: float
    duplicated_layers: list[int]
    num_duplicated: int
    layer_path: list[int]
    total_layers_in_path: int
    param_increase_pct: float
    # v1.1 fields:
    probe_scores: Optional[dict[str, float]] = None
    probe_deltas: Optional[dict[str, float]] = None
    combined_score_v1: Optional[float] = None
    combined_delta_v1: Optional[float] = None
    # Legacy Optional fields for reading v1.0 files (not written in v1.1 output):
    pubmedqa_score: Optional[float] = None
    eq_score: Optional[float] = None
    pubmedqa_delta: Optional[float] = None
    eq_delta: Optional[float] = None


class Rankings(BaseModel):
    top_combined: list[list[int]]
    # v1.1 field:
    probe_top: Optional[dict[str, list[list[int]]]] = None
    # Legacy Optional fields for reading v1.0 files:
    top_pubmedqa: Optional[list[list[int]]] = None
    top_eq: Optional[list[list[int]]] = None


class HeatmapMatrix(BaseModel):
    description: str
    data: list[list[Optional[float]]]


class HeatmapMatrices(BaseModel):
    combined_delta: HeatmapMatrix
    # v1.1 field:
    probe_deltas: Optional[dict[str, HeatmapMatrix]] = None
    # Legacy Optional fields for reading v1.0 files:
    pubmedqa_delta: Optional[HeatmapMatrix] = None
    eq_delta: Optional[HeatmapMatrix] = None


class ScanResults(BaseModel):
    llmri_version: str = "1.1.0"
    scan_metadata: ScanMetadata
    baseline: Optional[BaselineResult] = None
    results: list[ConfigResult] = Field(default_factory=list)
    rankings: Optional[Rankings] = None
    heatmap_matrices: Optional[HeatmapMatrices] = None

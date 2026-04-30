"""ARC-Challenge multiple-choice scorer.

Each probe presents a science question with four choices (A/B/C/D) and asks
the model to answer with a single letter.  Scoring is exact-match: 1.0 if the
first letter in the response matches the correct answer, 0.0 otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_ANSWERS = ("a", "b", "c", "d")


def score_arc(response: str, correct_answer: str) -> float:
    """Return 1.0 if the model's response matches the correct answer, else 0.0.

    Checks the first 5 characters of the stripped response for A, B, C, or D
    (case-insensitive, ignoring punctuation).
    """
    snippet = response.strip().lower()[:5]
    # Strip punctuation to catch "A.", "A)", etc.
    cleaned = "".join(c for c in snippet if c.isalpha())
    if cleaned and cleaned[0] in VALID_ANSWERS:
        return 1.0 if cleaned[0] == correct_answer.lower() else 0.0
    return 0.0


def score_arc_batch(
    responses: list[str],
    probes: list[dict[str, Any]],
) -> float:
    """Return the mean accuracy across a batch of ARC responses.

    Args:
        responses: Raw text outputs from the model, one per probe.
        probes: List of probe dicts with an "answer" key (the ground truth).

    Returns:
        Mean score in [0.0, 1.0].
    """
    if not probes:
        return 0.0
    scores = [
        score_arc(resp, probe["answer"])
        for resp, probe in zip(responses, probes)
    ]
    return sum(scores) / len(scores)


def load_arc_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load an ARC-Challenge probe file and return the list of probe dicts.

    Expected format per entry:
        {
            "id": "Mercury_SC_415702",
            "prompt": "Question: ...\n\nA) ...\nB) ...\nC) ...\nD) ...\n\nAnswer with just the letter A, B, C, or D:",
            "answer": "C",
            "type": "arc"
        }
    """
    with open(path) as f:
        probes = json.load(f)

    for p in probes:
        if "prompt" not in p or "answer" not in p:
            raise ValueError(
                f"Probe {p.get('id', '?')} is missing 'prompt' or 'answer' keys."
            )
        if p["answer"].upper() not in ("A", "B", "C", "D"):
            raise ValueError(
                f"Probe {p.get('id', '?')} has invalid answer: {p['answer']!r}. "
                f"Expected one of A, B, C, D."
            )

    return probes

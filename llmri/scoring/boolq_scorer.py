"""BoolQ yes/no scorer.

Each probe presents a passage and asks the model to answer yes or no.
Scoring is exact-match: 1.0 if the first valid keyword in the response
matches the ground-truth label, 0.0 otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_ANSWERS = ("yes", "no")


def score_boolq(response: str, correct_answer: str) -> float:
    """Return 1.0 if the model's response matches the correct answer, else 0.0.

    Checks the first 10 characters of the stripped response for "yes" or "no",
    case-insensitively.
    """
    snippet = response.strip().lower()[:10]
    for answer in VALID_ANSWERS:
        if answer in snippet:
            return 1.0 if answer == correct_answer.lower() else 0.0
    return 0.0


def score_boolq_batch(
    responses: list[str],
    probes: list[dict[str, Any]],
) -> float:
    """Return the mean accuracy across a batch of BoolQ responses.

    Args:
        responses: Raw text outputs from the model, one per probe.
        probes: List of probe dicts with an "answer" key (the ground truth).

    Returns:
        Mean score in [0.0, 1.0].
    """
    if not probes:
        return 0.0
    scores = [
        score_boolq(resp, probe["answer"])
        for resp, probe in zip(responses, probes)
    ]
    return sum(scores) / len(scores)


def load_boolq_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a BoolQ probe file and return the list of probe dicts.

    Expected format per entry:
        {
            "id": "some-id",
            "prompt": "Passage: ...\n\nQuestion: ...\n\nAnswer with just yes or no:",
            "answer": "yes",
            "type": "boolq"
        }
    """
    with open(path) as f:
        probes = json.load(f)

    for p in probes:
        if "prompt" not in p or "answer" not in p:
            raise ValueError(
                f"Probe {p.get('id', '?')} is missing 'prompt' or 'answer' keys."
            )
        if p["answer"].lower() not in VALID_ANSWERS:
            raise ValueError(
                f"Probe {p.get('id', '?')} has invalid answer: {p['answer']!r}. "
                f"Expected one of {VALID_ANSWERS}."
            )

    return probes

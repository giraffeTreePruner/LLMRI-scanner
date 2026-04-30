"""WinoGrande fill-in-the-blank scorer.

Each probe presents a sentence with a blank and two candidate options (1 or 2).
Scoring is exact-match: 1.0 if the first digit in the response matches the
correct option, 0.0 otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_ANSWERS = ("1", "2")


def score_winogrande(response: str, correct_answer: str) -> float:
    """Return 1.0 if the model's response matches the correct answer, else 0.0.

    Checks the first 5 characters of the stripped response for "1" or "2".
    """
    snippet = response.strip()[:5]
    for answer in VALID_ANSWERS:
        if answer in snippet:
            return 1.0 if answer == correct_answer else 0.0
    return 0.0


def score_winogrande_batch(
    responses: list[str],
    probes: list[dict[str, Any]],
) -> float:
    """Return the mean accuracy across a batch of WinoGrande responses.

    Args:
        responses: Raw text outputs from the model, one per probe.
        probes: List of probe dicts with an "answer" key (the ground truth).

    Returns:
        Mean score in [0.0, 1.0].
    """
    if not probes:
        return 0.0
    scores = [
        score_winogrande(resp, probe["answer"])
        for resp, probe in zip(responses, probes)
    ]
    return sum(scores) / len(scores)


def load_winogrande_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a WinoGrande probe file and return the list of probe dicts.

    Expected format per entry:
        {
            "id": "some-id",
            "prompt": "{sentence}\n\n1) {option1}\n2) {option2}\n\nAnswer with just 1 or 2:",
            "answer": "1",
            "type": "winogrande"
        }
    """
    with open(path) as f:
        probes = json.load(f)

    for p in probes:
        if "prompt" not in p or "answer" not in p:
            raise ValueError(
                f"Probe {p.get('id', '?')} is missing 'prompt' or 'answer' keys."
            )
        if p["answer"] not in VALID_ANSWERS:
            raise ValueError(
                f"Probe {p.get('id', '?')} has invalid answer: {p['answer']!r}. "
                f"Expected one of {VALID_ANSWERS}."
            )

    return probes

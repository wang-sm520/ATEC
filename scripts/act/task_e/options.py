"""Pure option helpers for Task E ACT data collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FULL_ORDER_OBJECTS = [1, 2, 3]
VALID_OBJECT_IDS = set(FULL_ORDER_OBJECTS)


def _validate_object_id(object_id: int) -> int:
    if object_id not in VALID_OBJECT_IDS:
        raise ValueError("object id must be one of [1, 2, 3]")
    return object_id


def resolve_pick_objects(pick_objects: Sequence[int], full_order_123: bool) -> list[int]:
    """Resolve object IDs to collect for Task E."""
    if full_order_123:
        return list(FULL_ORDER_OBJECTS)

    resolved = sorted({_validate_object_id(int(object_id)) for object_id in pick_objects})
    if not resolved:
        raise ValueError("pick_objects must not be empty")
    return resolved


def parse_grasp_z_offsets(items: Iterable[str]) -> dict[int, float]:
    """Parse OBJECT_ID:OFFSET pairs into an object-id keyed offset map."""
    offsets: dict[int, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError("expected OBJECT_ID:OFFSET")
        object_text, offset_text = item.split(":", 1)
        try:
            object_id = int(object_text)
            offset = float(offset_text)
        except ValueError as exc:
            raise ValueError("expected OBJECT_ID:OFFSET") from exc
        offsets[_validate_object_id(object_id)] = offset
    return offsets


def load_grasp_offset_json(path: str) -> dict[int, float]:
    """Load best grasp-z offsets from a sweep JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    best_offsets = data.get("best_offsets", {})
    return {_validate_object_id(int(object_id)): float(offset) for object_id, offset in best_offsets.items()}


def resolve_grasp_z_offsets(
    cli_offsets: Iterable[str] | None,
    json_path: str | None,
    defaults: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """Resolve grasp-z offsets with defaults, JSON, then CLI override precedence."""
    resolved: dict[int, float] = {}
    if defaults:
        resolved.update({_validate_object_id(int(object_id)): float(offset) for object_id, offset in defaults.items()})
    if json_path:
        resolved.update(load_grasp_offset_json(json_path))
    if cli_offsets:
        resolved.update(parse_grasp_z_offsets(cli_offsets))
    return resolved


def choose_best_offset(rows: Sequence[Mapping[str, float]]) -> float:
    """Choose the best offset by success rate, early terminations, then middle tie."""
    if not rows:
        raise ValueError("rows must not be empty")

    best_success_rate = max(float(row.get("success_rate", 0.0)) for row in rows)
    candidates = [row for row in rows if float(row.get("success_rate", 0.0)) == best_success_rate]

    fewest_early_terminations = min(int(row.get("early_terminations", 0)) for row in candidates)
    candidates = [
        row for row in candidates if int(row.get("early_terminations", 0)) == fewest_early_terminations
    ]

    offsets = sorted(float(row["offset"]) for row in candidates)
    return offsets[len(offsets) // 2]

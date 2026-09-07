"""Turning Autoware object classifications into class ids.

``ObjectClassification.label`` is a ``uint8`` enum, so the mapping runs in two visible
stages: enum value -> canonical name, here, and then name -> registry id through the
shared :func:`~t4perceval.importer._labels.encode_class_ids`. Baking enum-to-id directly
would tie the recording to Autoware's numbering and make two registries built from
different sources silently incompatible.

An object carries a *list* of classifications, each with a probability; the one with the
highest probability names the object. An empty list is ``"unknown"``, which is a real
class (id 0 when the registry has it), distinct from
:data:`~t4perceval.label.UNKNOWN_CLASS_ID` -- that sentinel only appears through
``unknown_labels="unknown"`` when the registry does not know a name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from t4perceval.importer._labels import UnknownLabels, encode_class_ids
from t4perceval.label import LabelRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = (
    "AUTOWARE_CLASS_NAMES",
    "UnknownLabels",
    "class_name_of",
    "classification_name",
    "encode_class_ids",
    "label_registry_from_autoware",
    "top_classification",
)

#: Canonical names indexed by ``ObjectClassification`` enum value. The legacy
#: ``autoware_auto_perception_msgs`` enum is the prefix ending at ``PEDESTRIAN``.
AUTOWARE_CLASS_NAMES: Final[tuple[str, ...]] = (
    "unknown",
    "car",
    "truck",
    "bus",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "animal",
    "hazard",
    "over_drivable",
    "under_drivable",
)


def class_name_of(label: int) -> str:
    """Return the canonical name of one ``ObjectClassification.label`` value."""
    if not 0 <= label < len(AUTOWARE_CLASS_NAMES):
        raise ValueError(
            f"ObjectClassification label {label} is outside the known enum "
            f"(0..{len(AUTOWARE_CLASS_NAMES) - 1})",
        )
    return AUTOWARE_CLASS_NAMES[label]


def top_classification(classification: Sequence[Any]) -> Any | None:
    """Return the classification with the highest probability, or ``None`` if empty."""
    if len(classification) == 0:
        return None
    return max(classification, key=lambda entry: float(entry.probability))


def classification_name(classification: Sequence[Any]) -> str:
    """Return the canonical name an object's classification list gives it."""
    best = top_classification(classification)
    return "unknown" if best is None else class_name_of(int(best.label))


def label_registry_from_autoware(
    *,
    prefix: str = "autoware",
    colors: Mapping[str, tuple[int, int, int]] | None = None,
) -> LabelRegistry:
    """Build a registry over the Autoware enum, ids in enum order.

    Discovery, kept separate from importing exactly as for T4: passing the result to the
    importer is the caller's explicit step, and a ground-truth source evaluated against
    this bag should be handed the same registry.
    """
    return LabelRegistry.from_names(AUTOWARE_CLASS_NAMES, prefix=prefix, colors=colors)

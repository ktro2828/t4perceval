"""Filters on what an object is: its class, its instance."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.descriptors import CLASS_ID, INSTANCE_ID
from t4perceval.system.base import SystemContext
from t4perceval.system.filter.base import MaskSystem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayBool


def _as_class_ids(values: Sequence[str | int] | None) -> tuple[str | int, ...] | None:
    return None if values is None else tuple(values)


def resolve_class_ids(
    values: tuple[str | int, ...],
    ctx: SystemContext,
    *,
    field_name: str,
) -> set[int]:
    """Turn a mix of class names and class ids into a set of ids."""
    names = [value for value in values if isinstance(value, str)]
    if names and ctx.labels is None:
        raise ValueError(
            f"{field_name} names {names} require a LabelRegistry; "
            "pass one as SystemContext(labels=...)",
        )
    # An unknown name raises rather than matching nothing: a typo must not silently
    # turn a filter into a no-op that skews every downstream metric.
    return {
        value if isinstance(value, int) else ctx.labels.class_id(value)  # type: ignore[union-attr]
        for value in values
    }


@define(slots=True)
class FilterByLabelSystem(MaskSystem):
    """Keep objects whose class is in ``labels`` and not in ``exclude``.

    Both accept class names, resolved through :attr:`SystemContext.labels`, or raw class
    ids. ``labels=None`` admits every class, so the two together express the original
    ``target_labels`` allowlist and ``ignore_attributes`` denylist. An unknown class name
    raises.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    FILTER_NAME: ClassVar[str] = "label"

    labels: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )
    exclude: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.labels is not None and len(self.labels) == 0:
            raise ValueError("labels must not be empty; pass None to admit every class")

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        class_ids = view.component(CLASS_ID).values
        keep = np.ones(len(class_ids), dtype=np.bool_)

        if self.labels is not None:
            wanted = resolve_class_ids(self.labels, ctx, field_name="labels")
            keep &= np.isin(class_ids, sorted(wanted))

        if self.exclude:
            unwanted = resolve_class_ids(self.exclude, ctx, field_name="exclude")
            keep &= ~np.isin(class_ids, sorted(unwanted))

        return keep


@define(slots=True)
class FilterByInstanceSystem(MaskSystem):
    """Keep objects whose instance is in ``instances`` and not in ``exclude``.

    Both accept dataset UUIDs, resolved through :attr:`SystemContext.instances`, or raw
    instance ids. This is the original ``target_uuids``. An unknown UUID raises.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (INSTANCE_ID,)
    FILTER_NAME: ClassVar[str] = "instance"

    instances: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )
    exclude: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.instances is not None and len(self.instances) == 0:
            raise ValueError("instances must not be empty; pass None to admit every instance")

    @staticmethod
    def _resolve(values: tuple[str | int, ...], ctx: SystemContext, *, field_name: str) -> set[int]:
        uuids = [value for value in values if isinstance(value, str)]
        if uuids and ctx.instances is None:
            raise ValueError(
                f"{field_name} uuids {uuids} require an InstanceRegistry; "
                "pass one as SystemContext(instances=...)",
            )
        return {
            value if isinstance(value, int) else ctx.instances.instance_id(value)  # type: ignore[union-attr]
            for value in values
        }

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        instance_ids = view.component(INSTANCE_ID).values
        keep = np.ones(len(instance_ids), dtype=np.bool_)

        if self.instances is not None:
            wanted = self._resolve(self.instances, ctx, field_name="instances")
            keep &= np.isin(instance_ids, sorted(wanted))

        if self.exclude:
            unwanted = self._resolve(self.exclude, ctx, field_name="exclude")
            keep &= ~np.isin(instance_ids, sorted(unwanted))

        return keep

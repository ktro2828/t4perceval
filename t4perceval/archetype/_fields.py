from __future__ import annotations

from typing import TYPE_CHECKING, Any

from attrs import Converter, field

from t4perceval.core.archetype import COMPONENT_METADATA_KEY, as_component

if TYPE_CHECKING:
    from t4perceval.core.component import Component
    from t4perceval.core.descriptor import ComponentDescriptor

__all__ = ("component_field",)


def _make_converter(component_type: type[Component], *, optional: bool) -> Converter:
    if optional:

        def convert_optional(value: Any) -> Component | None:
            return None if value is None else as_component(value, component_type)

        return Converter(convert_optional)

    def convert(value: Any) -> Component:
        return as_component(value, component_type)

    return Converter(convert)


def component_field(
    descriptor: ComponentDescriptor,
    component_type: type[Component],
    *,
    optional: bool = False,
    kw_only: bool = False,
) -> Any:
    """Declare one component column of an archetype.

    The descriptor is stored in the field metadata, which is how
    :class:`~t4perceval.core.archetype.Archetype` implements ``select()``,
    ``as_components()`` and the chunk round-trip generically.

    Args:
        descriptor: Canonical descriptor from :mod:`t4perceval.descriptors`.
        component_type: Component class the value is coerced to.
        optional: When set, the field defaults to ``None`` and may be absent.
        kw_only: Force the field to be keyword-only.
    """
    metadata = {COMPONENT_METADATA_KEY: (descriptor, component_type)}
    converter = _make_converter(component_type, optional=optional)

    if optional:
        return field(default=None, converter=converter, metadata=metadata, kw_only=kw_only)
    return field(converter=converter, metadata=metadata, kw_only=kw_only)

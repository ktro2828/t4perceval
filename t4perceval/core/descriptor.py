from __future__ import annotations

from attrs import define, field

__all__ = ("ComponentDescriptor",)

_SEPARATOR = ":"


@define(frozen=True, slots=True)
class ComponentDescriptor:
    """Fully describes the semantics of one column of data.

    Identity is carried by :attr:`component` alone: every component at a given entity
    path is uniquely identified by it. :attr:`archetype` and :attr:`component_type` are
    semantic hints only and are excluded from equality and hashing, so that the store
    keys stay stable even when the archetype layer is reorganized.

    Examples:
        >>> a = ComponentDescriptor("BatchDetection3D:position", archetype="BatchDetection3D")
        >>> b = ComponentDescriptor("BatchDetection3D:position")
        >>> a == b and hash(a) == hash(b)
        True
    """

    component: str = field(converter=str)
    archetype: str | None = field(default=None, eq=False, kw_only=True)
    component_type: str | None = field(default=None, eq=False, kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.component == "":
            raise ValueError("ComponentDescriptor.component must not be empty")

    @classmethod
    def of(
        cls,
        archetype: str,
        field_name: str,
        *,
        component_type: str | None = None,
    ) -> ComponentDescriptor:
        """Build a descriptor named ``"<archetype>:<field_name>"``.

        Args:
            archetype: Name of the owning archetype, e.g. ``"BatchDetection3D"``.
            field_name: Name of the archetype field, e.g. ``"position"``.
            component_type: Name of the component class, e.g. ``"BatchPosition3D"``.
        """
        return cls(
            f"{archetype}{_SEPARATOR}{field_name}",
            archetype=archetype,
            component_type=component_type,
        )

    def __str__(self) -> str:
        return self.component

    @property
    def field_name(self) -> str:
        """Return the part after ``:``, or the whole name when there is no ``:``."""
        _, _, tail = self.component.rpartition(_SEPARATOR)
        return tail or self.component

    def tagged(self, archetype: str) -> ComponentDescriptor:
        """Return an equal descriptor carrying ``archetype`` as its hint.

        :attr:`component` is untouched, so the result compares and hashes equal to
        ``self``. This only annotates provenance for debugging and Arrow metadata.
        """
        if self.archetype == archetype:
            return self
        return ComponentDescriptor(
            self.component,
            archetype=archetype,
            component_type=self.component_type,
        )

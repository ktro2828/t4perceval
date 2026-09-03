"""Coordinate transforms as recorded data.

A transform is not hidden state owned by a service: it is a row in the store like any
other observation. One row is one edge of the frame graph, split the way ROS splits a
``TransformStamped``::

    Chunk.frame_id            the parent frame
    Transform3D.child_frame_id  the child frame
    Transform3D.translation / .rotation

Whether an edge is static (a sensor calibration) or temporal (an ego pose) is a statement
about *time*, not about the kind of data, so both use ``Transform3D`` and differ only in
being logged with ``log_static`` or ``log``.

This package reads those rows back: :func:`~t4perceval.transform.graph.transform_edges`
and :class:`~t4perceval.transform.graph.FrameGraph` recover the graph from the data alone,
and :class:`~t4perceval.transform.lookup.TransformResolver` walks it -- inverting and
composing edges -- to answer one frame's pose in another. Nothing here owns a transform;
everything is rebuilt from what a recording holds, so a saved recording still knows its
frame tree.
"""

from __future__ import annotations

from t4perceval.transform.compose import chain, compose, identity, interpolate, invert
from t4perceval.transform.graph import (
    DEFAULT_ROOT,
    FrameGraph,
    TransformEdge,
    transform_edges,
)
from t4perceval.transform.lookup import LookupPolicy, TransformResolver

__all__ = (
    "DEFAULT_ROOT",
    "FrameGraph",
    "LookupPolicy",
    "TransformEdge",
    "TransformResolver",
    "chain",
    "compose",
    "identity",
    "interpolate",
    "invert",
    "transform_edges",
)

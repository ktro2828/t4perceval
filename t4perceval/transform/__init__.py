"""Coordinate transforms as recorded data.

A transform is not hidden state owned by a service: it is a row in the store like any
other observation, addressed by the pair of frames it relates.

    /transforms/<parent>/<child>        Transform3D(translation, rotation)

This package holds the addressing rules. Resolving a chain of edges into a single
relationship, and applying one to a chunk, come later and build on these.
"""

from __future__ import annotations

from t4perceval.transform.paths import DEFAULT_ROOT, edges, frames_of, transform_path

__all__ = ("DEFAULT_ROOT", "edges", "frames_of", "transform_path")

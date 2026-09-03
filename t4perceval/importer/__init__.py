"""Conversion from external formats into the t4perceval recording model.

The boundary this package draws:

* ``t4perceval.importer`` turns an **external representation** -- a T4 dataset, an MCAP
  ROS bag -- into components, archetypes, chunks and entity paths.
* ``t4perceval.io`` moves an **already-native** recording to and from persistent storage.

Reading a saved recording is therefore ``io``; reading a dataset is ``importer``.

Nothing is re-exported here. Each source pulls in its own optional dependency, so
importing this package must not pull in any of them -- ``import t4perceval`` stays free
of ``t4_devkit`` and of the MCAP libraries. Import the source you want directly::

    from t4perceval.importer.t4 import T4Importer
"""

from __future__ import annotations

__all__ = ()

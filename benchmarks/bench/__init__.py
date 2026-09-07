"""Benchmark support code: scene generation, the two library adapters, and the report.

Nothing here imports either library at module scope. The two adapters (`side_t4perceval`,
`side_perception_eval`) import theirs inside functions, which is what lets one script run
under two incompatible NumPy majors.
"""

from __future__ import annotations

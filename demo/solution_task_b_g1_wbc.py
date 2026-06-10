"""Dev alias for the Task B WBC solution.

The canonical, submittable implementation lives in ``demo/solution.py``. This module
re-exports it so existing local scripts that import ``demo.solution_task_b_g1_wbc``
keep working.
"""

from __future__ import annotations

from demo.solution import AlgSolution  # noqa: F401

__all__ = ["AlgSolution"]

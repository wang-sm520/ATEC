"""Guard against demo/task_b/ drifting from the demo/ submission sources.

The submission is uploaded from demo/task_b/. Every file there must be a
byte-identical copy of its demo/ source; otherwise an edit to a demo/ module
silently ships a stale copy. This makes that a red test, not a submission-day
surprise. Paths are resolved from this file's location (no cwd assumption).
"""

import filecmp
import os
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEMO = os.path.join(_REPO_ROOT, "demo")
_STAGED = os.path.join(_DEMO, "task_b")

SUBMISSION_FILES = [
    "solution.py",
    "mini_wbc.py",
    "task_b_nav.py",
    "task_b_perception.py",
    "task_b_planner.py",
    "policy18.onnx",
    "requirements.txt",
]


class SubmissionStagingTest(unittest.TestCase):
    def test_staged_copies_match_sources(self):
        for name in SUBMISSION_FILES:
            src = os.path.join(_DEMO, name)
            staged = os.path.join(_STAGED, name)
            with self.subTest(file=name):
                self.assertTrue(os.path.exists(src), f"missing source demo/{name}")
                self.assertTrue(os.path.exists(staged), f"missing staged demo/task_b/{name}")
                self.assertTrue(
                    filecmp.cmp(src, staged, shallow=False),
                    f"demo/task_b/{name} has drifted from demo/{name} (re-sync the copy)",
                )


if __name__ == "__main__":
    unittest.main()

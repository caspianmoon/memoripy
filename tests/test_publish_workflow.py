from __future__ import annotations

import re
import unittest
from pathlib import Path


class PublishWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-pypi.yml"
        cls.workflow = cls.workflow_path.read_text(encoding="utf-8")

    def test_requires_manual_confirmation_from_master(self) -> None:
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn("CONFIRM_PUBLISH", self.workflow)
        self.assertIn('refs/heads/master', self.workflow)
        self.assertNotIn("types: [published]", self.workflow)

    def test_publishes_exact_github_release_artifacts(self) -> None:
        self.assertIn("gh release download", self.workflow)
        self.assertIn("memoripy-*.whl", self.workflow)
        self.assertIn("memoripy-*.tar.gz", self.workflow)
        self.assertNotIn("python -m build", self.workflow)
        self.assertIn("Tag {expected_version} does not match wheel version {version}", self.workflow)
        self.assertIn("python -m twine check dist/*", self.workflow)

    def test_uses_least_privilege_oidc_without_static_tokens(self) -> None:
        self.assertEqual(self.workflow.count("id-token: write"), 1)
        self.assertIn("name: pypi", self.workflow)
        self.assertNotIn("PYPI_TOKEN", self.workflow)
        self.assertNotIn("password:", self.workflow)
        self.assertRegex(
            self.workflow,
            re.compile(r"pypa/gh-action-pypi-publish@[0-9a-f]{40}"),
        )


if __name__ == "__main__":
    unittest.main()

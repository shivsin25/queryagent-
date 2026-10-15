"""The repository holds everything the app needs to run elsewhere."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_every_source_file_is_committed():
    """A .gitignore pattern once matched src/query_tool/storage/ (meant for the local data
    folder), so the deployed app failed with ModuleNotFoundError. Catch that here."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    ignored = subprocess.run(
        [git, "ls-files", "--others", "--ignored", "--exclude-standard", "--",
         "src", "frontend", "tests", "data", "scripts"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout.splitlines()
    assert [path for path in ignored if "__pycache__" not in path] == []

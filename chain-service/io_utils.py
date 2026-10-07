"""
io_utils.py — atomic JSON writes shared by run_screen.py and
rank_shortlist.py.

A checkpoint that dies mid-write leaves a truncated, unparseable JSON file
that poisons the next read of it. `json.dump` straight to the destination
path is not atomic; write-to-a-temp-file-then-rename is, on POSIX
(`os.replace` within the same filesystem is atomic there), so every writer
in this pipeline goes through `atomic_write_json` rather than calling
`open()` + `json.dump()` directly.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Union


def atomic_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp_path, path)


def git_commit_short() -> Optional[str]:
    """Current HEAD short hash, or None if it can't be determined.

    Never raises: a manifest missing this field is fine, a run crashing
    because of it is not. In particular, the container image built for the
    sidecar (python:3.11-slim) has no git installed by default, so this
    returning None there is an expected, not exceptional, case.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

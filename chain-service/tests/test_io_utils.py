import json
import os
from unittest.mock import patch

import pytest

from io_utils import atomic_write_json, git_commit_short


class TestAtomicWriteJson:
    def test_writes_readable_json(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2, 3]}

    def test_no_tmp_file_left_behind_on_success(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"ok": True})
        assert not (tmp_path / "out.json.tmp").exists()
        assert set(os.listdir(tmp_path)) == {"out.json"}

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"version": 1})
        atomic_write_json(path, {"version": 2})
        assert json.loads(path.read_text()) == {"version": 2}

    def test_interrupted_write_leaves_real_file_untouched(self, tmp_path):
        """The whole point of write-tmp-then-rename: if the process dies
        before os.replace runs, the destination still holds the last good
        write, not a truncated one. Simulated by making os.replace raise
        after the tmp file has already been written."""
        path = tmp_path / "out.json"
        atomic_write_json(path, {"version": 1})  # establish a "last good" file

        with patch("io_utils.os.replace", side_effect=OSError("simulated kill mid-checkpoint")):
            with pytest.raises(OSError):
                atomic_write_json(path, {"version": 2, "big": "x" * 10_000})

        # The temp file exists, holding the write that never got promoted...
        tmp_file = tmp_path / "out.json.tmp"
        assert tmp_file.exists()
        assert json.loads(tmp_file.read_text())["version"] == 2

        # ...and the real path is untouched: still version 1, still valid JSON.
        assert json.loads(path.read_text()) == {"version": 1}

    def test_first_ever_write_survives_the_same_interrupt(self, tmp_path):
        """Same guarantee when there is no prior file: an interrupted first
        write must not leave a partial file at the destination path."""
        path = tmp_path / "out.json"

        with patch("io_utils.os.replace", side_effect=OSError("simulated kill")):
            with pytest.raises(OSError):
                atomic_write_json(path, {"version": 1})

        assert not path.exists()
        assert (tmp_path / "out.json.tmp").exists()


class TestGitCommitShort:
    def test_returns_a_short_hash_in_this_repo(self):
        commit = git_commit_short()
        assert commit is not None
        assert 7 <= len(commit) <= 12

    def test_returns_none_rather_than_raising_when_git_is_unavailable(self):
        """The sidecar's container image (python:3.11-slim) has no git
        installed by default -- this must degrade, not crash the run."""
        with patch("io_utils.subprocess.run", side_effect=FileNotFoundError("no git")):
            assert git_commit_short() is None

    def test_returns_none_on_nonzero_exit(self):
        class FakeResult:
            returncode = 128
            stdout = ""

        with patch("io_utils.subprocess.run", return_value=FakeResult()):
            assert git_commit_short() is None

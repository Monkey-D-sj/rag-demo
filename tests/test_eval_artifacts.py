from pathlib import Path

import pytest

from rag.eval.artifacts import (
    assert_resume_compatible,
    atomic_write_json,
    create_run,
    load_manifest,
    make_run_id,
    read_item,
    update_manifest,
    write_item,
)


def test_artifacts_are_atomic_and_state_transitions_are_checked(tmp_path: Path):
    run_id = make_run_id(tmp_path)
    create_run(tmp_path, run_id, {"sample_fingerprint": "x"})
    write_item(tmp_path, run_id, "q/1", {"id": "q/1", "answer": "a"})
    assert read_item(tmp_path, run_id, "q/1")["answer"] == "a"
    update_manifest(tmp_path, run_id, status="collected")
    update_manifest(tmp_path, run_id, status="scoring")
    with pytest.raises(ValueError):
        update_manifest(tmp_path, run_id, status="collecting")


def test_completed_run_cannot_resume(tmp_path: Path):
    run_id = make_run_id(tmp_path)
    create_run(tmp_path, run_id, {"sample_fingerprint": "x"})
    update_manifest(tmp_path, run_id, status="collected")
    update_manifest(tmp_path, run_id, status="scoring")
    update_manifest(tmp_path, run_id, status="completed")
    with pytest.raises(ValueError):
        assert_resume_compatible(load_manifest(tmp_path, run_id), {"sample_fingerprint": "x"})


import os
from unittest.mock import MagicMock

import orchestrator as orch


def _check(**kwargs):
    orch.run_single_check(
        kaggle_username=kwargs.get("kaggle_username", "kaggle_user"),
        lightning_teamspace=kwargs.get("lightning_teamspace", "team/space"),
        hf_repo=kwargs.get("hf_repo", "hf/repo"),
        hf_token=kwargs.get("hf_token", "tok"),
        total_iters_target=kwargs.get("total_iters_target", 6000),
    )
    return orch.load_state()


def test_state_machine_rotates_kaggle_lightning_kaggle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orch, "training_complete", lambda *a, **k: (False, 0))
    monkeypatch.setattr(orch, "launch_kaggle", lambda username, hf_token, hf_repo: "fake/kernel")
    monkeypatch.setattr(orch, "launch_lightning_job", lambda teamspace: "fake-lightning-job")

    kaggle_status = {"value": "running"}
    monkeypatch.setattr(orch, "check_kaggle_status", lambda username: kaggle_status["value"])
    lightning_status = {"value": "running"}
    monkeypatch.setattr(orch, "check_lightning_status",
                         lambda job_id, teamspace: lightning_status["value"])

    # 1. no state file -> launch kaggle
    assert not os.path.exists(orch.STATE_PATH)
    state = _check()
    assert state.active_platform == "kaggle"
    assert state.active_job_id == "fake/kernel"

    # 2. kaggle running -> no transition
    state = _check()
    assert state.active_platform == "kaggle"

    # 3. kaggle complete -> hours tallied, goes idle
    kaggle_status["value"] = "complete"
    state = _check()
    assert state.active_platform is None
    assert state.kaggle_hours_used_this_week > 0

    # A fast unit test can't accumulate 30 real hours, so seed the quota
    # directly to exercise the "kaggle at budget -> fall through to
    # lightning" branch. week_key/month_key are already correct from the
    # calls above, so this won't be clobbered by the period-reset check.
    state.kaggle_hours_used_this_week = orch.KAGGLE_HOURS_PER_WEEK_BUDGET
    orch.save_state(state)

    # 4. idle, kaggle at quota -> launch lightning
    state = _check()
    assert state.active_platform == "lightning"
    assert state.active_job_id == "fake-lightning-job"

    # 5. lightning running -> no transition
    state = _check()
    assert state.active_platform == "lightning"

    # 6. lightning complete -> hours tallied, goes idle
    lightning_status["value"] = "complete"
    state = _check()
    assert state.active_platform is None
    assert state.lightning_hours_used_this_month > 0

    # simulate the weekly reset (a new ISO week) freeing up kaggle again
    state.kaggle_hours_used_this_week = 0.0
    orch.save_state(state)

    # 7. idle, kaggle back under budget -> back to kaggle
    state = _check()
    assert state.active_platform == "kaggle"


def test_both_platforms_at_quota_exits_cleanly_without_launching(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orch, "training_complete", lambda *a, **k: (False, 0))
    launched = []
    monkeypatch.setattr(orch, "launch_kaggle", lambda username, hf_token, hf_repo: launched.append("kaggle"))
    monkeypatch.setattr(orch, "launch_lightning_job", lambda teamspace: launched.append("lightning"))

    # prime week_key/month_key via a real call, then seed both quotas as exhausted
    state = _check()
    state.active_platform = None
    state.active_job_id = None
    state.kaggle_hours_used_this_week = orch.KAGGLE_HOURS_PER_WEEK_BUDGET
    state.lightning_hours_used_this_month = orch.LIGHTNING_HOURS_PER_MONTH_BUDGET
    orch.save_state(state)
    launched.clear()  # discard the priming call's launch

    capsys.readouterr()
    state = _check()
    out = capsys.readouterr().out

    assert launched == []
    assert state.active_platform is None
    assert "waiting for reset" in out


def test_training_complete_stops_launching_new_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orch, "training_complete", lambda *a, **k: (True, 5999))
    launched = []
    monkeypatch.setattr(orch, "launch_kaggle", lambda username, hf_token, hf_repo: launched.append("kaggle"))

    state = _check(total_iters_target=6000)

    assert launched == []
    assert state.last_checked_iter == 5999


def test_state_round_trips_atomically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = orch.OrchestratorState(active_platform="kaggle", active_job_id="x/y",
                                    kaggle_hours_used_this_week=3.5)
    orch.save_state(state)

    assert not os.path.exists(orch.STATE_PATH + ".tmp")
    loaded = orch.load_state()
    assert loaded == state


def test_check_kaggle_status_parses_enum_qualified_value(monkeypatch):
    # real kaggle-cli output observed in production: the status is the
    # enum-qualified "KernelWorkerStatus.ERROR", not a bare "error"
    fake_result = MagicMock(
        stdout='abdullahrasheed4500/mini-gpt-ddp-orchestrator has status "KernelWorkerStatus.ERROR"\n',
        stderr="",
    )
    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: fake_result)

    assert orch.check_kaggle_status("abdullahrasheed4500") == "error"


def test_check_kaggle_status_parses_bare_value(monkeypatch):
    fake_result = MagicMock(
        stdout='someuser/mini-gpt-ddp-orchestrator has status "running"\n',
        stderr="",
    )
    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: fake_result)

    assert orch.check_kaggle_status("someuser") == "running"


def test_check_kaggle_status_parses_cancel_acknowledged(monkeypatch):
    # real kaggle-cli output observed in production after a ~8hr training
    # session was cancelled: an underscored multi-word enum value that an
    # exact-match allowlist (a prior fix) failed to recognize, leaving the
    # orchestrator stuck thinking a dead kernel was still active
    fake_result = MagicMock(
        stdout='someuser/mini-gpt-ddp-orchestrator has status '
               '"KernelWorkerStatus.CANCEL_ACKNOWLEDGED"\n',
        stderr="",
    )
    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: fake_result)

    assert orch.check_kaggle_status("someuser") == "error"


def test_launch_kaggle_injects_credentials_into_pushed_source_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / orch.KAGGLE_PROJECT_DIR
    project_dir.mkdir()
    (project_dir / "kernel-metadata.json").write_text("{}")
    original_entry_src = '"""entry docstring"""\nimport os\nprint("hi")\n'
    (project_dir / "kaggle_entry.py").write_text(original_entry_src)

    captured = {}

    def fake_run(cmd, **kwargs):
        push_dir = cmd[cmd.index("-p") + 1]
        with open(os.path.join(push_dir, "kaggle_entry.py")) as f:
            captured["pushed_src"] = f.read()
        captured["push_dir"] = push_dir
        return MagicMock(returncode=0, stdout="Kernel version 1 successfully pushed.\n", stderr="")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)

    kernel = orch.launch_kaggle("someuser", "tok-value", "some/repo")

    assert kernel == f"someuser/{orch.KAGGLE_KERNEL_SLUG}"
    pushed_src = captured["pushed_src"]
    assert "tok-value" in pushed_src
    assert "some/repo" in pushed_src
    # the injected credentials come before the rest of the script runs
    assert pushed_src.index("tok-value") < pushed_src.index("print(")
    # the pushed copy is a temp dir, not the tracked kaggle_project/ directory
    assert captured["push_dir"] != str(project_dir)
    # the real, git-tracked file is never modified
    assert (project_dir / "kaggle_entry.py").read_text() == original_entry_src
    compile(pushed_src, "kaggle_entry.py", "exec")  # injected prefix must be valid syntax


def test_launch_kaggle_detects_error_message_despite_exit_zero(tmp_path, monkeypatch):
    # real kaggle-cli behavior observed in production: it printed
    # "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached."
    # and still exited 0, so check=True alone never caught this failure
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / orch.KAGGLE_PROJECT_DIR
    project_dir.mkdir()
    (project_dir / "kernel-metadata.json").write_text("{}")
    (project_dir / "kaggle_entry.py").write_text('"""doc"""\nimport os\n')

    fake_result = MagicMock(
        returncode=0,
        stdout="Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.\n",
        stderr="",
    )
    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: fake_result)

    kernel = orch.launch_kaggle("someuser", "tok-value", "some/repo")

    assert kernel is None


def test_launch_lightning_job_splits_teamspace_into_user_and_name(monkeypatch):
    # real lightning-cli behavior observed in production: `job run` rejected
    # the combined "owner/name" form accepted by `job list`/`job inspect`
    # with "Neither user or org are specified, but one of them has to be
    # the owner of the Teamspace" -- it wants --user and a bare --teamspace
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr(orch.subprocess, "run", fake_run)

    job_name = orch.launch_lightning_job("abdullahrasheed45/default-project")

    assert job_name is not None
    cmd = captured["cmd"]
    assert cmd[cmd.index("--user") + 1] == "abdullahrasheed45"
    assert cmd[cmd.index("--teamspace") + 1] == "default-project"


def test_kaggle_push_failure_falls_through_to_lightning_same_check(tmp_path, monkeypatch):
    # simulates Kaggle's real server-side quota running out mid-week, which
    # our wall-clock hour tracking can't see coming until a push fails
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orch, "training_complete", lambda *a, **k: (False, 0))
    monkeypatch.setattr(orch, "launch_kaggle", lambda username, hf_token, hf_repo: None)
    monkeypatch.setattr(orch, "launch_lightning_job", lambda teamspace: "fake-lightning-job")

    state = _check()

    assert state.kaggle_hours_used_this_week == orch.KAGGLE_HOURS_PER_WEEK_BUDGET
    # fell through to lightning within the same check, not a second cycle
    assert state.active_platform == "lightning"
    assert state.active_job_id == "fake-lightning-job"


def test_lightning_launch_failure_treated_as_exhausted_without_crashing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orch, "training_complete", lambda *a, **k: (False, 0))
    monkeypatch.setattr(orch, "launch_kaggle", lambda username, hf_token, hf_repo: "fake/kernel")
    monkeypatch.setattr(orch, "launch_lightning_job", lambda teamspace: None)

    # prime week_key/month_key, then put kaggle at quota so lightning is tried
    state = _check()
    state.active_platform = None
    state.active_job_id = None
    state.kaggle_hours_used_this_week = orch.KAGGLE_HOURS_PER_WEEK_BUDGET
    orch.save_state(state)

    capsys.readouterr()
    state = _check()
    out = capsys.readouterr().out

    assert state.lightning_hours_used_this_month == orch.LIGHTNING_HOURS_PER_MONTH_BUDGET
    assert state.active_platform is None
    assert "waiting for reset" in out

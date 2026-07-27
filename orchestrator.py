"""Single-check-and-act GPU training orchestrator.

Rotates training between Kaggle (30 free GPU-hrs/week, 2x T4 via DDP) and
Lightning AI (free monthly GPU credits, 1x T4), handing off through a
checkpoint stored on the Hugging Face Hub. Meant to be invoked repeatedly
(e.g. every 15 min by a GitHub Actions cron job) via `--single-check` --
each invocation makes at most one state transition, then exits. It is not a
long-running loop.

Colab is intentionally excluded from this automation; see README.md.
"""

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from checkpoint_sync import pull_checkpoint_meta

STATE_PATH = "orchestrator_state.json"
CKPT_PATH = os.path.join("checkpoints", "ddp_2gpu.pt")

KAGGLE_HOURS_PER_WEEK_BUDGET = 30.0
LIGHTNING_HOURS_PER_MONTH_BUDGET = 75.0  # ~5hr buffer under the ~80hr free tier
ASSUMED_SESSION_HOURS = 12.0  # fallback if a launch timestamp is somehow missing

KAGGLE_KERNEL_SLUG = "mini-gpt-ddp-orchestrator"
KAGGLE_PROJECT_DIR = "kaggle_project"
LIGHTNING_IMAGE = "pytorch/pytorch:latest"
REPO_URL = "https://github.com/AbdullahRasheed45/mini-gpt-ddp.git"


@dataclass
class OrchestratorState:
    active_platform: str | None = None
    active_job_id: str | None = None
    active_started_at: float | None = None
    kaggle_hours_used_this_week: float = 0.0
    kaggle_week_key: str | None = None
    lightning_hours_used_this_month: float = 0.0
    lightning_month_key: str | None = None
    total_iters_target: int = 6000
    last_checked_iter: int = 0


# ----------------------------------------------------------------------------
# state I/O
# ----------------------------------------------------------------------------

def load_state() -> OrchestratorState:
    if not os.path.exists(STATE_PATH):
        return OrchestratorState()
    with open(STATE_PATH) as f:
        data = json.load(f)
    return OrchestratorState(**data)


def save_state(state: OrchestratorState) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, STATE_PATH)


def _iso_week_key(now: datetime) -> str:
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def _month_key(now: datetime) -> str:
    return f"{now.year}-{now.month:02d}"


def apply_period_resets(state: OrchestratorState, now: datetime) -> None:
    week_key = _iso_week_key(now)
    if state.kaggle_week_key != week_key:
        state.kaggle_hours_used_this_week = 0.0
        state.kaggle_week_key = week_key

    month_key = _month_key(now)
    if state.lightning_month_key != month_key:
        state.lightning_hours_used_this_month = 0.0
        state.lightning_month_key = month_key


def _elapsed_hours(started_at: float | None, now: datetime, default: float) -> float:
    if started_at is None:
        return default
    elapsed = now.timestamp() - started_at
    if elapsed <= 0:
        return default
    return elapsed / 3600.0


# ----------------------------------------------------------------------------
# Kaggle
# ----------------------------------------------------------------------------

def check_kaggle_status(kaggle_username: str) -> str:
    """Returns one of: 'running', 'queued', 'complete', 'error', 'unknown'."""
    kernel = f"{kaggle_username}/{KAGGLE_KERNEL_SLUG}"
    result = subprocess.run(["kaggle", "kernels", "status", kernel],
                             capture_output=True, text=True)
    match = re.search(r'has status "(\w+)"', result.stdout)
    if not match:
        print(f"[orchestrator] could not parse kaggle status: "
              f"stdout={result.stdout!r} stderr={result.stderr!r}")
        return "unknown"

    status = match.group(1).lower()
    if status == "complete":
        return "complete"
    if status in ("error", "cancelled", "cancelacknowledged"):
        return "error"
    if status == "running":
        return "running"
    if status == "queued":
        return "queued"
    return "unknown"


def launch_kaggle(kaggle_username: str) -> str:
    subprocess.run(["kaggle", "kernels", "push", "-p", KAGGLE_PROJECT_DIR], check=True)
    kernel = f"{kaggle_username}/{KAGGLE_KERNEL_SLUG}"
    print(f"[orchestrator] launched kaggle kernel {kernel}")
    return kernel


# ----------------------------------------------------------------------------
# Lightning AI
# ----------------------------------------------------------------------------

def check_lightning_status(job_id: str, teamspace: str) -> str:
    """Returns one of: 'running', 'queued', 'complete', 'error', 'unknown'."""
    result = subprocess.run(
        ["lightning", "job", "inspect", job_id, "--teamspace", teamspace],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[orchestrator] lightning job inspect failed: {result.stderr}")
        return "unknown"

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"[orchestrator] could not parse lightning job JSON: {result.stdout!r}")
        return "unknown"

    status = str(info.get("status") or info.get("state") or "").lower()
    if status in ("completed", "complete", "succeeded", "success"):
        return "complete"
    if status in ("failed", "error", "stopped", "cancelled"):
        return "error"
    if status in ("running", "active"):
        return "running"
    if status in ("pending", "queued", "provisioning", "pending_execution"):
        return "queued"
    return "unknown"


def launch_lightning_job(teamspace: str) -> str:
    job_name = f"minigpt-{int(time.time())}"
    command = (
        f"git clone --depth 1 {REPO_URL} repo && cd repo && "
        "pip install -q -r requirements.txt && python lightning_train.py"
    )
    cmd = [
        "lightning", "job", "run",
        "--name", job_name,
        "--teamspace", teamspace,
        "--machine", "T4",
        "--image", LIGHTNING_IMAGE,
        "--command", command,
        "--env", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "--env", f"HF_REPO_ID={os.environ.get('HF_REPO_ID', '')}",
    ]
    subprocess.run(cmd, check=True)
    print(f"[orchestrator] launched lightning job {job_name} on {teamspace}")
    return job_name


# ----------------------------------------------------------------------------
# completion check
# ----------------------------------------------------------------------------

def training_complete(hf_repo: str, hf_token: str | None, total_iters_target: int) -> tuple[bool, int]:
    meta = pull_checkpoint_meta(hf_repo, CKPT_PATH, hf_token)
    if meta is None:
        return False, 0
    last_iter = int(meta.get("iter", 0))
    return last_iter >= total_iters_target - 1, last_iter


# ----------------------------------------------------------------------------
# main state machine (one transition per call)
# ----------------------------------------------------------------------------

def run_single_check(kaggle_username: str, lightning_teamspace: str,
                      hf_repo: str, hf_token: str | None, total_iters_target: int) -> None:
    state = load_state()
    now = datetime.now(timezone.utc)
    apply_period_resets(state, now)
    state.total_iters_target = total_iters_target

    complete, last_iter = training_complete(hf_repo, hf_token, total_iters_target)
    state.last_checked_iter = last_iter
    if complete:
        print(f"[orchestrator] training complete at iter {last_iter} "
              f"(target {total_iters_target}); nothing to launch")
        save_state(state)
        return

    if state.active_platform is None and state.kaggle_hours_used_this_week < KAGGLE_HOURS_PER_WEEK_BUDGET:
        state.active_job_id = launch_kaggle(kaggle_username)
        state.active_platform = "kaggle"
        state.active_started_at = now.timestamp()
        save_state(state)
        return

    if state.active_platform == "kaggle":
        status = check_kaggle_status(kaggle_username)
        if status in ("complete", "error"):
            elapsed = _elapsed_hours(state.active_started_at, now, ASSUMED_SESSION_HOURS)
            state.kaggle_hours_used_this_week += elapsed
            print(f"[orchestrator] kaggle run {status}; +{elapsed:.2f}h "
                  f"(week total {state.kaggle_hours_used_this_week:.2f}h)")
            state.active_platform = None
            state.active_job_id = None
            state.active_started_at = None
        else:
            print(f"[orchestrator] kaggle run still {status}; nothing to do this cycle")
        save_state(state)
        return

    if (state.active_platform is None
            and state.lightning_hours_used_this_month < LIGHTNING_HOURS_PER_MONTH_BUDGET):
        state.active_job_id = launch_lightning_job(lightning_teamspace)
        state.active_platform = "lightning"
        state.active_started_at = now.timestamp()
        save_state(state)
        return

    if state.active_platform == "lightning":
        status = check_lightning_status(state.active_job_id, lightning_teamspace)
        if status in ("complete", "error"):
            elapsed = _elapsed_hours(state.active_started_at, now, ASSUMED_SESSION_HOURS)
            state.lightning_hours_used_this_month += elapsed
            print(f"[orchestrator] lightning run {status}; +{elapsed:.2f}h "
                  f"(month total {state.lightning_hours_used_this_month:.2f}h)")
            state.active_platform = None
            state.active_job_id = None
            state.active_started_at = None
        else:
            print(f"[orchestrator] lightning run still {status}; nothing to do this cycle")
        save_state(state)
        return

    print("[orchestrator] both platforms at quota for this period; waiting for reset")
    save_state(state)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"[orchestrator] missing required environment variable: {name}")
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-check", action="store_true", required=True,
                     help="the only supported entry mode; orchestrator.py is not a long-running loop")
    ap.add_argument("--total-iters-target", type=int,
                     default=int(os.environ.get("TOTAL_ITERS_TARGET", 6000)))
    args = ap.parse_args()

    kaggle_username = _require_env("KAGGLE_USERNAME")
    lightning_teamspace = _require_env("LIGHTNING_TEAMSPACE")
    hf_repo = _require_env("HF_REPO_ID")
    hf_token = os.environ.get("HF_TOKEN")

    run_single_check(kaggle_username, lightning_teamspace, hf_repo, hf_token, args.total_iters_target)


if __name__ == "__main__":
    main()

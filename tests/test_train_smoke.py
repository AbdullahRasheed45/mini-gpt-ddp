import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_train_smoke_test_runs_end_to_end(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--smoke_test",
            "--data_dir",
            str(tmp_path / "data"),
            "--out_dir",
            str(tmp_path / "checkpoints"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "training complete" in result.stdout
    assert (tmp_path / "checkpoints" / "ddp_2gpu.pt").exists()

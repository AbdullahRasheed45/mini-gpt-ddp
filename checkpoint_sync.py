"""Shared Hugging Face Hub checkpoint pull/push for train.py and lightning_train.py.

The checkpoint's path is used unchanged as both the local file path and the
path inside the HF model repo (e.g. "checkpoints/ddp_2gpu.pt" on both sides),
so a checkpoint written by a Kaggle run can be resumed by a Lightning AI run
and vice versa.
"""

import json
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download


def _meta_path(ckpt_path: str) -> str:
    return ckpt_path.removesuffix(".pt") + ".meta.json"


def pull_checkpoint(repo_id: str | None, path: str, token: str | None) -> bool:
    """Download `path` from `repo_id` into the local `path`. Returns True on success."""
    if not repo_id:
        print("[checkpoint_sync] no HF_REPO_ID set, skipping checkpoint pull")
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        downloaded = hf_hub_download(repo_id=repo_id, filename=path, token=token)
        shutil.copy(downloaded, path)
        print(f"[checkpoint_sync] pulled checkpoint hf://{repo_id}/{path} -> {path}")
        return True
    except Exception as e:
        print(f"[checkpoint_sync] no remote checkpoint at hf://{repo_id}/{path} "
              f"({type(e).__name__}: {e}); starting fresh")
        return False


def push_checkpoint(repo_id: str | None, path: str, token: str | None) -> None:
    """Upload local `path` to the same path in `repo_id`."""
    if not repo_id:
        print("[checkpoint_sync] no HF_REPO_ID set, skipping checkpoint push")
        return
    try:
        HfApi(token=token).upload_file(
            path_or_fileobj=path,
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[checkpoint_sync] pushed checkpoint {path} -> hf://{repo_id}/{path}")
    except Exception as e:
        print(f"[checkpoint_sync] checkpoint push failed ({type(e).__name__}: {e})")


def push_checkpoint_meta(repo_id: str | None, ckpt_path: str, iter_value: int,
                          total_iters_target: int, token: str | None) -> None:
    """Push a small sidecar JSON with the checkpoint's iter count.

    Lets the orchestrator check training progress without downloading the
    full (potentially multi-hundred-MB) checkpoint file on every poll.
    """
    if not repo_id:
        return
    meta_path = _meta_path(ckpt_path)
    try:
        with open(meta_path, "w") as f:
            json.dump({"iter": iter_value, "total_iters_target": total_iters_target}, f)
        HfApi(token=token).upload_file(
            path_or_fileobj=meta_path,
            path_in_repo=meta_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[checkpoint_sync] pushed checkpoint metadata {meta_path} -> hf://{repo_id}/{meta_path}")
    except Exception as e:
        print(f"[checkpoint_sync] checkpoint metadata push failed ({type(e).__name__}: {e})")


def pull_checkpoint_meta(repo_id: str | None, ckpt_path: str, token: str | None) -> dict | None:
    """Download and parse the sidecar JSON pushed by push_checkpoint_meta."""
    if not repo_id:
        return None
    meta_path = _meta_path(ckpt_path)
    try:
        local = hf_hub_download(repo_id=repo_id, filename=meta_path, token=token)
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        print(f"[checkpoint_sync] no checkpoint metadata at hf://{repo_id}/{meta_path} "
              f"({type(e).__name__}: {e})")
        return None

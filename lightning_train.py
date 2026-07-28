"""Single-GPU training entry point for Lightning AI (free tier: 1x T4).

Thin wrapper around train.py: pulls the latest checkpoint from the HF Hub
(so a Lightning run can pick up where a Kaggle 2-GPU run left off, or vice
versa) and then shells out to train.py with nproc_per_node=1, since there's
only one GPU here. All HF Hub logic lives in checkpoint_sync.py and is
reused, not duplicated, from train.py.
"""

import argparse
import os
import subprocess
import sys

from checkpoint_sync import pull_checkpoint


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="ddp_2gpu")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--hf_repo", default=os.environ.get("HF_REPO_ID"))
    return ap.parse_known_args()


def main():
    args, passthrough = get_args()
    hf_token = os.environ.get("HF_TOKEN")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.run_name}.pt")
    pull_checkpoint(args.hf_repo, ckpt_path, hf_token)

    # a fresh Lightning job runs in a stateless container, unlike a
    # persistent Studio -- tokenize TinyStories here too (same as
    # kaggle_entry.py) rather than assuming data/train.bin already exists
    if not os.path.exists(os.path.join(args.data_dir, "train.bin")):
        print("[lightning_train] no tokenized data found, running data.py")
        subprocess.run(["python", "data.py", "--out_dir", args.data_dir, "--num_proc", "4"], check=True)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", "train.py",
        "--run_name", args.run_name, "--out_dir", args.out_dir,
        "--data_dir", args.data_dir,
    ]
    if args.hf_repo:
        cmd += ["--hf_repo", args.hf_repo]
    cmd += passthrough

    print(f"[lightning_train] launching: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

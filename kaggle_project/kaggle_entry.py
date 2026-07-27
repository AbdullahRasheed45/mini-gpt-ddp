"""Script-kernel entry point for the automated training orchestrator.

Pushed by orchestrator.py via `kaggle kernels push`. Clones the public
mini-gpt-ddp repo fresh on every run (so it always trains with the latest
code, without a separately-maintained Kaggle Dataset to keep in sync),
installs dependencies, tokenizes TinyStories if needed, and launches the
2-GPU DDP training run.

Credentials: Kaggle Secrets (kaggle_secrets.UserSecretsClient) are only
reachable by kernel runs triggered through Kaggle's own UI, not by runs
triggered via `kaggle kernels push` -- confirmed empirically, since the
same secrets that fail here work fine when the kernel is run manually via
"Save Version". So orchestrator.py instead writes a secrets.json file into
this directory right before pushing (never committed to git, sourced from
GitHub Actions' own secrets), which takes priority; Kaggle Secrets remain a
fallback for manual, UI-triggered runs of this same kernel.
"""

import json
import os
import subprocess

REPO_URL = "https://github.com/AbdullahRasheed45/mini-gpt-ddp.git"
WORKDIR = "/kaggle/working/mini-gpt-ddp"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
    if os.path.exists(secrets_path):
        with open(secrets_path) as f:
            creds = json.load(f)
        os.environ["HF_TOKEN"] = creds["HF_TOKEN"]
        os.environ["HF_REPO_ID"] = creds["HF_REPO_ID"]
    else:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
        os.environ["HF_REPO_ID"] = secrets.get_secret("HF_REPO_ID")

    if not os.path.isdir(WORKDIR):
        run(["git", "clone", "--depth", "1", REPO_URL, WORKDIR])
    os.chdir(WORKDIR)

    run(["pip", "install", "-q", "-r", "requirements.txt"])

    if not os.path.exists("data/train.bin"):
        run(["python", "data.py", "--num_proc", "4"])

    run([
        "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
        "--run_name", "ddp_2gpu",
    ])


if __name__ == "__main__":
    main()

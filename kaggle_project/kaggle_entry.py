"""Script-kernel entry point for the automated training orchestrator.

Pushed by orchestrator.py via `kaggle kernels push`. Clones the public
mini-gpt-ddp repo fresh on every run (so it always trains with the latest
code, without a separately-maintained Kaggle Dataset to keep in sync),
installs dependencies, tokenizes TinyStories if needed, and launches the
2-GPU DDP training run. HF_TOKEN and HF_REPO_ID come from Kaggle Secrets
(Add-ons > Secrets) so the checkpoint can hand off to/from Lightning AI.
"""

import os
import subprocess

REPO_URL = "https://github.com/AbdullahRasheed45/mini-gpt-ddp.git"
WORKDIR = "/kaggle/working/mini-gpt-ddp"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
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

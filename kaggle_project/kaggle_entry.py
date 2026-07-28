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
"Save Version". `kaggle kernels push` also has no mechanism to upload any
file other than this single code_file (its content becomes the entire
request body -- there is no sibling-file bundling), so orchestrator.py
instead prepends two `os.environ[...] = ...` lines directly to this
script's source before pushing the rendered copy (sourced from GitHub
Actions' own secrets; never committed to git, never written to disk here).
If HF_TOKEN is already set when main() runs, that injected value is used;
otherwise this falls back to Kaggle Secrets, for manual UI-triggered runs
of this same kernel.

Dependencies: Kaggle has assigned Tesla P100 GPUs (compute capability
sm_60) on every observed run so far, not the requested T4s -- unclear
whether machine_shape in kernel-metadata.json just isn't honored for
API-pushed kernels, or T4s simply weren't available in the free-tier pool.

`pip install torch --index-url .../cu118` (no version pin, no
--force-reinstall) turned out to be a no-op: pip treats an unversioned
requirement as already satisfied by whatever's installed and never
touches the index at all. The diagnostic print below confirmed it left
Kaggle's base-image torch (2.10.0+cu128) in place both times, which is
why "skip reinstalling" and "reinstall via cu118" produced byte-identical
failures -- they were never actually different. torch 2.6.0's real
cu118 build script (checked directly against PyTorch's GitHub tag) still
includes 6.0 in TORCH_CUDA_ARCH_LIST, and it's confirmed available for
Kaggle's cp312 -- so this pins that version explicitly with
--force-reinstall to guarantee the swap actually happens this time.
"""

import os
import subprocess

REPO_URL = "https://github.com/AbdullahRasheed45/mini-gpt-ddp.git"
WORKDIR = "/kaggle/working/mini-gpt-ddp"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    if "HF_TOKEN" not in os.environ:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
        os.environ["HF_REPO_ID"] = secrets.get_secret("HF_REPO_ID")

    if not os.path.isdir(WORKDIR):
        run(["git", "clone", "--depth", "1", REPO_URL, WORKDIR])
    os.chdir(WORKDIR)

    run(["pip", "install", "-q", "--force-reinstall", "torch==2.6.0",
         "--index-url", "https://download.pytorch.org/whl/cu118"])
    run(["pip", "install", "-q", "huggingface_hub", "tiktoken", "datasets"])
    run(["python", "-c",
         "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, "
         "'archs', torch.cuda.get_arch_list())"])

    if not os.path.exists("data/train.bin"):
        run(["python", "data.py", "--num_proc", "4"])

    run([
        "torchrun", "--standalone", "--nproc_per_node=2", "train.py",
        "--run_name", "ddp_2gpu",
    ])


if __name__ == "__main__":
    main()

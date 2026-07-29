# miniGPT-DDP

[![CI](https://github.com/AbdullahRasheed45/mini-gpt-ddp/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdullahRasheed45/mini-gpt-ddp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GPT pretraining from scratch with DistributedDataParallel on 2x Kaggle T4s.
Project A of a three-project ladder targeting ML/research engineer roles at
frontier labs: (A) distributed pretraining, (B) inference engine, (C) GRPO
post-training with a vLLM rollout server.

## What this demonstrates

- Transformer implemented from first principles (SDPA attention, weight
  tying, GPT-2 init scheme) -- no HuggingFace model code
- Real multi-GPU data-parallel training: NCCL, gradient sync scheduling
  (`no_sync` during accumulation), per-rank data sharding
- fp16 mixed precision done correctly on pre-Ampere hardware (GradScaler,
  loss-scale monitoring) -- T4s have no bf16, which forces you to actually
  understand loss scaling
- Production habits: atomic checkpointing, bit-exact resume (optimizer +
  scaler + RNG state) across Kaggle's 12h session limit, throughput
  instrumentation
- Automated multi-cloud training rotation (Kaggle <-> Lightning AI) via a
  GitHub Actions state machine, handing off through checkpoints on the
  Hugging Face Hub -- run to actual completion end-to-end, unattended (see
  [Results](#results))

## Layout

| path | purpose |
|---|---|
| `model.py` | the GPT (config: 8L/8H/512d, 50.9M params measured -- see [Results](#results)) |
| `data.py` | TinyStories -> tokenized uint16 memmap binaries |
| `train.py` | DDP training loop; also runs single-GPU and `--smoke_test` on CPU |
| `benchmark.py` | tokens/sec measurement for scaling-efficiency numbers |
| `checkpoint_sync.py` | Hugging Face Hub checkpoint pull/push, shared by `train.py`, `lightning_train.py`, and `orchestrator.py` |
| `lightning_train.py` | single-GPU entry point for a Lightning AI job |
| `orchestrator.py` | the Kaggle <-> Lightning AI rotation state machine (see [Automated Training Orchestration](#automated-training-orchestration)) |
| `kaggle_project/` | script-kernel pushed to Kaggle by `orchestrator.py` (`kaggle_entry.py` + `kernel-metadata.json`) |
| `notebooks/kaggle_launcher.ipynb` | exact cells to run manually on Kaggle |
| `tests/` | model + orchestrator unit tests and an end-to-end CPU training smoke test, run in CI |
| `.github/workflows/ci.yml` | lint (ruff) + test (pytest) on every push/PR |
| `.github/workflows/orchestrator.yml` | the cron that runs `orchestrator.py` every 15 min |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,dev]"   # core + data-prep (datasets, tiktoken) + dev (pytest, ruff)
```

Data prep pulls TinyStories from the Hugging Face Hub. Unauthenticated
requests work fine but are rate-limited; to raise the limit, log in once with
`hf auth login` (or set `HF_TOKEN` in your shell).

## Quickstart

```bash
python train.py --smoke_test                      # CPU sanity check, <1 min
python data.py                                    # tokenize (once)
torchrun --standalone --nproc_per_node=2 train.py # the real thing
```

## Development

```bash
pytest      # unit tests + CPU training smoke test
ruff check . # lint
```

## Automated Training Orchestration

A GitHub Actions cron job (`.github/workflows/orchestrator.yml`, every 15
min) runs `orchestrator.py --single-check`, a state machine that rotates
training between **Kaggle** (30 free GPU-hrs/week, 2x T4 via DDP) and
**Lightning AI** (free monthly GPU credits, 1x T4), so training keeps
progressing without anyone babysitting sessions. The two platforms hand off
through a checkpoint on the **Hugging Face Hub**: whichever platform is
active pulls the latest checkpoint before resuming and pushes after every
save (`checkpoint_sync.py`), so a run can pick up on the other platform
exactly where the last one left off.

Colab is deliberately **not** part of this automation -- Google's Colab
free-tier terms prohibit automated/remote-triggered execution. It remains a
manual, human-driven fallback only.

### Required GitHub repo secrets

Set these under **Settings > Secrets and variables > Actions**, or via
`gh secret set NAME --repo <owner>/<repo>` (prompts for hidden input, so the
value never has to be pasted anywhere visible):

| secret | where it comes from |
|---|---|
| `KAGGLE_USERNAME` | your Kaggle username -- the `username` field in `~/.kaggle/kaggle.json` |
| `KAGGLE_KEY` | your Kaggle API key -- the `key` field in `~/.kaggle/kaggle.json` (Kaggle Account settings > Create New API Token) |
| `HF_TOKEN` | a **write**-access token from https://huggingface.co/settings/tokens |
| `HF_REPO_ID` | the private HF model repo used as the checkpoint store, e.g. `your-username/mini-gpt-ddp` |
| `LIGHTNING_API_KEY` | from https://lightning.ai/settings |
| `LIGHTNING_USER_ID` | also from https://lightning.ai/settings (shown next to the API key) -- both are needed together for non-interactive `lightning` CLI auth, without a browser |
| `LIGHTNING_TEAMSPACE` | your Lightning AI teamspace, formatted `{owner}/{teamspace-name}`, shown in the dashboard URL |

### Checking status

- **GitHub Actions tab** -> "GPU Training Orchestrator" -> pick a run -> the
  `Run orchestrator single check` step logs every decision
  (`[orchestrator] ...` lines: launched, still running, completed, quota
  reached, etc).
- **Training progress**: the `iter` field in
  `checkpoints/ddp_2gpu.meta.json` on the HF repo (a tiny sidecar file, so
  checking progress doesn't require downloading the full checkpoint).
- **Orchestrator state** (`active_platform`, hours used this week/month,
  last checked iter): persisted as `orchestrator_state.json` in the
  `orchestrator-state-*` GitHub Actions cache -- visible under **Settings >
  Actions > Caches**, or read directly from any workflow run's logs.

### Stopping the rotation

Go to the **Actions** tab -> "GPU Training Orchestrator" -> **...** ->
**Disable workflow**. No code changes needed; re-enable the same way.

## Results

A full run completed end-to-end through the automated rotation above, with
no manual intervention once launched: 6,000 iterations, checkpointing every
250 iterations to the Hugging Face Hub, spanning a Kaggle session (until it
hit the 12h session cap, then the 30h/week quota) and a Lightning AI job
that picked up the exact checkpoint Kaggle left off at and carried it to
completion.

**Loss**: 1.84 at iter 700, down to 1.25-1.30 by iter ~4900, still trending
down at completion.

**Sample generation** (greedy top-k sampling, `temperature=0.8, top_k=50`,
at completion):

> Once upon a time there was a little girl named Lily. She loved to play in
> the park with her friends. One day, Lily saw a big, black dog with an
> unusual face. She had never seen anything like it before. Lily's friends
> were scared of the dog, but Lily wanted to pet it. She walked up to the
> dog and said, "Hi, Mr. Dog! You are very big and furry." The dog didn't
> say anything back, but Lily thought he was friendly...

**Cross-GPU throughput** (single GPU each, fp16 + GradScaler):

| platform | GPU | architecture | tensor cores | tok/s |
|---|---|---|---|---|
| Kaggle | Tesla P100 | Pascal | none | ~19,700 |
| Lightning AI | T4 | Turing | yes | ~41,000 |

~2.1x on the T4, entirely from tensor-core fp16 acceleration -- Pascal has
no dedicated matmul hardware for fp16, so it runs at roughly fp32 speed.

**What actually shipped vs. what was planned, honestly**: `train.py`'s DDP
path is real and correct (NCCL, `no_sync` gradient-sync scheduling,
per-rank sharding, exercised by the CPU smoke test), but every real run
captured above landed on a single GPU per session, not the 2x-T4 DDP setup
the project was originally scoped around. Kaggle assigned Tesla P100s
instead of the requested T4x2 on every observed API-triggered run
(`kernel-metadata.json`'s `machine_shape: GPU_T4X2` was not honored), and
Lightning AI's free tier is 1x T4 by design. The scaling-efficiency and
sync-cost experiments below (numbers 1-3) need an actual 2-GPU session to
run as originally designed; the cross-GPU table above is the real
substitute result that came out of what the free tiers actually granted.
Getting there also surfaced 16 real platform-specific bugs -- credential
injection paths, GPU-compatibility pins, inconsistent CLI output formats
across both platforms -- each fixed and regression-tested; see the git
history for the full debugging trail.

## Experiments to run (these become the writeup)

1. **Scaling efficiency**: benchmark.py at world_size 1 vs 2. Report
   efficiency = tps_2gpu / (2 x tps_1gpu).
2. **Sync-cost curve**: 2-GPU efficiency at grad_accum ∈ {1, 4, 8, 32}.
   Accumulation amortizes the all-reduce; the curve makes the
   communication/computation tradeoff visible.
3. **no_sync ablation**: comment out the `no_sync` context and re-measure.
   Quantifies what naive DDP costs.
4. **fp16 vs fp32**: throughput and peak memory both ways. On T4 expect
   ~2x throughput from tensor cores -- confirmed above, cross-architecture,
   as a side effect of free-tier GPU allocation.
5. **Loss-scale trace**: log `scaler.get_scale()` over training. If it ever
   crashes downward, you've caught fp16 instability in the wild -- document it.
   (In the completed run, scale rose from 65536 to 262144 over the course of
   training with no downward corrections -- stable throughout.)
6. **Batch-size / micro-batch sweep**: find the VRAM ceiling, note where
   throughput saturates.

Writeup framing: "What 90 hours/week of free T4s teaches you about
distributed training" -- lead with the efficiency curve, be honest about
where the T4s bottleneck, show loss curves and generated samples.

## Roadmap: Projects B and C

**B -- inference lab**: KV cache for this model (measure the naive
`generate()` above vs cached), batched inference, then speculative decoding
with the draft model on GPU 0 and the target on GPU 1.

**C -- GRPO on dual T4**: Qwen2.5-0.5B-base + LoRA + GSM8K. Architecture:
GPU 0 runs vLLM as a rollout server, GPU 1 trains, LoRA adapters sync
between them each round -- a miniature of how verl/OpenRLHF structure
generation/training separation at scale. Phase 0 is a pass@1 vs pass@16
baseline eval of the base model on GSM8K (an afternoon, and it's figure 1
of that project's writeup).

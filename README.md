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

## Layout

| path | purpose |
|---|---|
| `model.py` | the GPT (config: 8L/8H/512d ≈ 38M params) |
| `data.py` | TinyStories -> tokenized uint16 memmap binaries |
| `train.py` | DDP training loop; also runs single-GPU and `--smoke_test` on CPU |
| `benchmark.py` | tokens/sec measurement for scaling-efficiency numbers |
| `notebooks/kaggle_launcher.ipynb` | exact cells to run on Kaggle |
| `tests/` | model unit tests + an end-to-end CPU smoke test, run in CI |
| `.github/workflows/ci.yml` | lint (ruff) + test (pytest) on every push/PR |

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

## Experiments to run (these become the writeup)

1. **Scaling efficiency**: benchmark.py at world_size 1 vs 2. Report
   efficiency = tps_2gpu / (2 x tps_1gpu).
2. **Sync-cost curve**: 2-GPU efficiency at grad_accum ∈ {1, 4, 8, 32}.
   Accumulation amortizes the all-reduce; the curve makes the
   communication/computation tradeoff visible.
3. **no_sync ablation**: comment out the `no_sync` context and re-measure.
   Quantifies what naive DDP costs.
4. **fp16 vs fp32**: throughput and peak memory both ways. On T4 expect
   ~2x throughput from tensor cores.
5. **Loss-scale trace**: log `scaler.get_scale()` over training. If it ever
   crashes downward, you've caught fp16 instability in the wild -- document it.
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

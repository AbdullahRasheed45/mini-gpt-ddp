"""
Measure training throughput to quantify DDP scaling efficiency.

Run twice on Kaggle and compare:
    torchrun --standalone --nproc_per_node=1 benchmark.py
    torchrun --standalone --nproc_per_node=2 benchmark.py

Reports tokens/sec after warmup. Scaling efficiency = (2-GPU tps) / (2 * 1-GPU tps).
Expect roughly 85-95% on T4s: the gap is NCCL all-reduce cost, and this number
is a headline result for the writeup. Bonus experiment: vary --grad_accum_steps
(1, 4, 16, 32) and watch efficiency rise as accumulation amortizes the sync --
that curve IS the "communication vs computation" story told in data.
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro_batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--measure_steps", type=int, default=20)
    args = ap.parse_args()

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0

    cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    use_amp = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=use_amp)

    # synthetic batches: benchmark isolates compute+comm, not data loading
    def batch():
        x = torch.randint(0, cfg.vocab_size, (args.micro_batch_size, args.block_size), device=device)
        return x, x.clone()

    tokens_per_step = args.micro_batch_size * args.block_size * args.grad_accum_steps * world_size

    def step():
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum_steps):
            x, y = batch()
            sync = nullcontext() if (not is_ddp or micro == args.grad_accum_steps - 1) else model.no_sync()
            with sync, amp_ctx:
                _, loss = model(x, y)
            scaler.scale(loss / args.grad_accum_steps).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup_steps):
        step()
    if use_amp:
        torch.cuda.synchronize()
    if is_ddp:
        dist.barrier()

    t0 = time.time()
    for _ in range(args.measure_steps):
        step()
    if use_amp:
        torch.cuda.synchronize()
    if is_ddp:
        dist.barrier()
    dt = time.time() - t0

    if master:
        tps = tokens_per_step * args.measure_steps / dt
        print(f"world_size={world_size} grad_accum={args.grad_accum_steps} "
              f"micro_bs={args.micro_batch_size}")
        print(f"throughput: {tps:,.0f} tokens/sec "
              f"({dt / args.measure_steps:.3f} s/step)")
        if use_amp:
            print(f"peak memory: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB / GPU")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

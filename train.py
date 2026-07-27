"""
DDP training for miniGPT on 2x Kaggle T4s.

Launch (both GPUs):
    torchrun --standalone --nproc_per_node=2 train.py

Single GPU (for the scaling comparison -- run this too, it's a key result):
    torchrun --standalone --nproc_per_node=1 train.py --run_name single_gpu

CPU smoke test (tiny model, verifies the loop end-to-end):
    python train.py --smoke_test

T4-specific decisions, worth understanding rather than copying blindly:
- fp16 + GradScaler, NOT bf16. T4 (Turing, sm75) has no bf16 support.
  fp16 has a narrow exponent range, so gradients can underflow to zero;
  GradScaler multiplies the loss by a large factor before backward and
  un-scales before the optimizer step, keeping small gradients alive.
  Watch scaler.get_scale() in the logs: repeated halving means instability.
- Gradient accumulation gives a "large batch" (~0.5M tokens) despite 16GB
  VRAM. DDP detail: we skip gradient sync (model.no_sync()) on all but the
  final micro-step, otherwise we'd pay NCCL all-reduce cost per micro-step
  for no benefit.
- Checkpoint/resume is load-bearing on Kaggle: sessions die at 12h. We save
  model, optimizer, scaler, and RNG state so resume is bit-exact.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint_sync import pull_checkpoint, push_checkpoint, push_checkpoint_meta
from model import GPT, GPTConfig

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

def get_args():
    ap = argparse.ArgumentParser()
    # run
    ap.add_argument("--run_name", default="ddp_2gpu")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--smoke_test", action="store_true")
    # model
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--block_size", type=int, default=512)
    # optimization
    ap.add_argument("--micro_batch_size", type=int, default=16)   # per GPU, per micro-step
    ap.add_argument("--grad_accum_steps", type=int, default=32)   # per GPU; batch = micro*accum*world
    ap.add_argument("--max_iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # eval / logging / checkpointing
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=100)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--ckpt_interval", type=int, default=250)
    ap.add_argument("--wandb", action="store_true")
    # checkpoint sync (Hugging Face Hub, for rotating between Kaggle/Lightning)
    ap.add_argument("--hf_repo", default=os.environ.get("HF_REPO_ID"))
    return ap.parse_args()


# ----------------------------------------------------------------------------
# data loading: random crops from a memmap'd token stream
# ----------------------------------------------------------------------------

class BinDataLoader:
    """Random-offset batch sampler over a flat token file.

    Each rank seeds its RNG differently, so ranks see different data without
    any coordination -- with random cropping over ~500M tokens, overlap
    between ranks is negligible and this is simpler and faster than a
    DistributedSampler over a Dataset.
    """

    def __init__(self, path: str, block_size: int, batch_size: int,
                 device: torch.device, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)

    def get_batch(self):
        ix = self.rng.integers(0, len(self.data) - self.block_size - 1, size=self.batch_size)
        bs = self.block_size
        x = torch.stack([torch.from_numpy(self.data[i : i + bs].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + bs].astype(np.int64)) for i in ix])
        if self.device.type == "cuda":
            # pinned memory + non_blocking overlaps H2D copy with compute
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    args = get_args()

    # ---- distributed setup -------------------------------------------------
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0

    if args.smoke_test:
        args.n_layer, args.n_head, args.n_embd, args.block_size = 2, 2, 64, 64
        args.micro_batch_size, args.grad_accum_steps = 2, 2
        args.max_iters, args.eval_interval, args.eval_iters = 10, 5, 2
        args.warmup_iters, args.ckpt_interval, args.log_interval = 2, 5, 1

    torch.manual_seed(1337 + rank)

    use_amp = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tokens_per_iter = args.micro_batch_size * args.block_size * args.grad_accum_steps * world_size
    if master:
        print(f"world_size={world_size} device={device}")
        print(f"tokens per optimizer step: {tokens_per_iter:,}")

    # ---- data --------------------------------------------------------------
    if args.smoke_test and not os.path.exists(os.path.join(args.data_dir, "train.bin")):
        # synthesize a random token file so the loop can be tested anywhere
        os.makedirs(args.data_dir, exist_ok=True)
        for split, n in [("train", 200_000), ("val", 20_000)]:
            arr = np.random.default_rng(0).integers(0, 50256, size=n, dtype=np.uint16)
            arr.tofile(os.path.join(args.data_dir, f"{split}.bin"))

    train_loader = BinDataLoader(os.path.join(args.data_dir, "train.bin"),
                                 args.block_size, args.micro_batch_size, device, seed=1337 + rank)
    val_loader = BinDataLoader(os.path.join(args.data_dir, "val.bin"),
                               args.block_size, args.micro_batch_size, device, seed=9999 + rank)

    # ---- model / optimizer -------------------------------------------------
    cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    if master:
        print(f"model parameters: {model.num_params() / 1e6:.1f}M")

    # weight decay on matrices only, not on layernorms/embedd... actually
    # standard practice: decay 2D+ params, no decay on 1D (norms, biases)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=use_amp,
    )

    # ---- resume ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.run_name}.pt")
    hf_token = os.environ.get("HF_TOKEN")

    if master and not os.path.exists(ckpt_path):
        pull_checkpoint(args.hf_repo, ckpt_path, hf_token)
    if is_ddp:
        dist.barrier()

    start_iter = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt["iter"] + 1
        torch.set_rng_state(ckpt["cpu_rng"].cpu())
        if use_amp and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu(), device)
        if master:
            print(f"resumed from {ckpt_path} at iter {start_iter}")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if is_ddp else model

    # ---- lr schedule -------------------------------------------------------
    def lr_at(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        if it >= args.max_iters:
            return args.min_lr
        ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - args.min_lr)

    # ---- eval --------------------------------------------------------------
    @torch.no_grad()
    def estimate_val_loss() -> float:
        model.eval()
        losses = torch.zeros(args.eval_iters, device=device)
        for k in range(args.eval_iters):
            x, y = val_loader.get_batch()
            with amp_ctx:
                _, loss = model(x, y)
            losses[k] = loss
        model.train()
        mean = losses.mean()
        if is_ddp:
            dist.all_reduce(mean, op=dist.ReduceOp.AVG)
        return mean.item()

    # ---- wandb -------------------------------------------------------------
    wb = None
    if args.wandb and master:
        import wandb as wb
        wb.init(project="mini-gpt-ddp", name=args.run_name, config=vars(args))

    # ---- training loop -----------------------------------------------------
    model.train()
    t0 = time.time()
    running_tps = None

    for it in range(start_iter, args.max_iters):
        lr = lr_at(it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(args.grad_accum_steps):
            x, y = train_loader.get_batch()
            # skip the all-reduce on all but the last micro-step
            sync = nullcontext() if (not is_ddp or micro == args.grad_accum_steps - 1) else model.no_sync()
            with sync:
                with amp_ctx:
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum_steps
                loss_accum += loss.item()
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ---- logging -------------------------------------------------------
        if it % args.log_interval == 0:
            if use_amp:
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_iter * args.log_interval / dt if it > start_iter else 0.0
            running_tps = tps if running_tps is None else 0.9 * running_tps + 0.1 * tps
            if master:
                print(f"iter {it:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                      f"grad_norm {grad_norm:.2f} | {tps:,.0f} tok/s | "
                      f"scale {scaler.get_scale():.0f}" if use_amp else
                      f"iter {it:5d} | loss {loss_accum:.4f} | lr {lr:.2e}")
                if wb:
                    wb.log({"train/loss": loss_accum, "train/lr": lr,
                            "train/grad_norm": grad_norm.item(),
                            "perf/tokens_per_sec": tps}, step=it)

        if it % args.eval_interval == 0 and it > 0:
            val_loss = estimate_val_loss()
            if master:
                print(f"iter {it:5d} | VAL loss {val_loss:.4f}")
                if wb:
                    wb.log({"val/loss": val_loss}, step=it)

        # ---- checkpoint ----------------------------------------------------
        if master and it % args.ckpt_interval == 0 and it > 0:
            tmp = ckpt_path + ".tmp"
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "iter": it,
                "config": vars(args),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if use_amp else None,
            }, tmp)
            os.replace(tmp, ckpt_path)  # atomic: a mid-save session kill can't corrupt the checkpoint
            push_checkpoint(args.hf_repo, ckpt_path, hf_token)
            push_checkpoint_meta(args.hf_repo, ckpt_path, it, args.max_iters, hf_token)

    # final checkpoint + sample
    if master:
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "iter": args.max_iters - 1,
                    "config": vars(args), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if use_amp else None}, ckpt_path)
        push_checkpoint(args.hf_repo, ckpt_path, hf_token)
        push_checkpoint_meta(args.hf_repo, ckpt_path, args.max_iters - 1, args.max_iters, hf_token)
        print("training complete")
        if not args.smoke_test:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            prompt = torch.tensor([enc.encode("Once upon a time")], device=device)
            out = raw_model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
            print("sample:", enc.decode(out[0].tolist()))

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

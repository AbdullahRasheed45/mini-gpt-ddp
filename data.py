"""
Tokenize TinyStories into flat uint16 binary files for memmap loading.

Run once (CPU is fine, ~10-15 min on Kaggle):
    python data.py

Produces train.bin / val.bin in --out_dir. Training then reads these with
np.memmap, which means zero tokenization cost during training and trivially
correct random access for both DDP ranks.

Why uint16: GPT-2 vocab (50257) fits in 16 bits, halving disk/IO vs int32.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--num_proc", type=int, default=4)
    args = ap.parse_args()

    # imports deferred so `python data.py --help` works without deps installed
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256, used as document separator

    ds = load_dataset("roneneldan/TinyStories")

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    tokenized = ds.map(
        tokenize,
        remove_columns=["text"],
        num_proc=args.num_proc,
        desc="tokenizing",
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for split, dset in tokenized.items():
        split_name = "val" if split == "validation" else split
        total = int(np.sum(dset["len"], dtype=np.uint64))
        path = os.path.join(args.out_dir, f"{split_name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        idx = 0
        n_shards = 512
        for shard_idx in range(n_shards):
            shard = dset.shard(num_shards=n_shards, index=shard_idx, contiguous=True)
            shard = shard.with_format("numpy")
            batch = np.concatenate(shard["ids"])
            arr[idx : idx + len(batch)] = batch
            idx += len(batch)
        arr.flush()
        print(f"{split_name}: {total:,} tokens -> {path}")


if __name__ == "__main__":
    main()

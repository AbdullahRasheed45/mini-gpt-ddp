import torch

from model import GPT, GPTConfig


def tiny_config(**overrides):
    cfg = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
               dropout=0.0, bias=False)
    cfg.update(overrides)
    return GPTConfig(**cfg)


def test_forward_returns_loss_when_targets_given():
    cfg = tiny_config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))

    logits, loss = model(idx, targets)

    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_forward_without_targets_only_computes_last_position():
    cfg = tiny_config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

    logits, loss = model(idx)

    assert logits.shape == (2, 1, cfg.vocab_size)
    assert loss is None


def test_weight_tying():
    model = GPT(tiny_config())
    assert model.lm_head.weight is model.tok_emb.weight


def test_generate_extends_sequence_by_requested_length():
    cfg = tiny_config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))

    out = model.generate(idx, max_new_tokens=5, temperature=0.8, top_k=10)

    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], idx)


def test_num_params_excludes_position_embedding_by_default():
    cfg = tiny_config()
    model = GPT(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert model.num_params(non_embedding=True) == total - model.pos_emb.weight.numel()
    assert model.num_params(non_embedding=False) == total

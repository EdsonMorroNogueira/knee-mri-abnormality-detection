"""MultiQueryAttentionPool plus an optional additive domain-knowledge bias on the attention logits.

The bias (slot_prior, shape (n_out, n_slot)) is added before the softmax, so a
strong prior steers which slots each class attends to. With slot_prior=None the
behaviour is identical to MultiQueryAttentionPool. Added in Stage 14.
"""
import torch
import torch.nn as nn


class MultiQueryAttentionPoolWithPrior(nn.Module):
    """Masked attention, one query per target class, with an optional additive prior on the logits."""

    def __init__(self, dim, n_slot, n_out, hidden=64, p=0.2, slot_prior=None):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

        if slot_prior is not None:
            assert slot_prior.shape == (n_out, n_slot), (
                f"slot_prior must have shape (n_out={n_out}, n_slot={n_slot}), "
                f"got {tuple(slot_prior.shape)}"
            )
            # Buffer, not Parameter: fixed knowledge, not trained.
            self.register_buffer("slot_prior", slot_prior)
        else:
            self.slot_prior = None

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5

        if self.slot_prior is not None:
            # (n_out, n_slot) -> (1, n_out, n_slot): same bias for the whole batch.
            att = att + self.slot_prior.unsqueeze(0)

        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))

        logits = (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        return logits, att


if __name__ == "__main__":
    torch.manual_seed(0)

    # Without a prior: only checks that shapes are right and it runs.
    model_no_prior = MultiQueryAttentionPoolWithPrior(dim=4, n_slot=6, n_out=12, hidden=16)
    x = torch.randn(3, 6, 4)
    mask = torch.ones(3, 6)
    mask[0, -2:] = 0.0
    logits, att = model_no_prior(x, mask)
    assert logits.shape == (3, 12), f"expected (3, 12), got {logits.shape}"
    assert att.shape == (3, 12, 6), f"expected (3, 12, 6), got {att.shape}"
    print("[OK] no prior: correct shapes")

    # Extreme prior (+100 on slot 0, -100 elsewhere): class 0 must attend only to slot 0.
    n_out, n_slot = 12, 6
    slot_prior = torch.zeros(n_out, n_slot)
    slot_prior[0, 0] = 100.0
    slot_prior[0, 1:] = -100.0
    model_with_prior = MultiQueryAttentionPoolWithPrior(dim=4, n_slot=n_slot, n_out=n_out, hidden=16, slot_prior=slot_prior)
    _, att_prior = model_with_prior(x, mask)
    assert att_prior[1, 0, 0] > 0.99, f"expected attention ~1.0 on slot 0 for class 0, got {att_prior[1, 0, 0]:.4f}"
    print(f"[OK] with prior: class 0 attention on slot 0 = {att_prior[1, 0, 0]:.4f} (expected ~1.0)")

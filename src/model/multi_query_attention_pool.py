"""Masked attention pooling over slot embeddings, with one learned query per target class.

Each of the 12 targets (ACL, Fracture, Effusion, ...) learns its own query, so
it can attend to different slots (e.g. ACL to sagittal, Fracture to axial).
Absent slots are masked out before the softmax so they get exactly zero weight.
"""
import torch
import torch.nn as nn


class MultiQueryAttentionPool(nn.Module):
    """Masked attention with one query per target class (n_out of them)."""

    def __init__(self, dim, n_slot, n_out, hidden=64, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        # Per-slot identity vector: lets the model tell slots apart beyond content.
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb                        # (B, S, hidden)
        # Scaled dot product of every query against every slot: (B, n_out, S).
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        # Slot presence is shared by all outputs: (B, S) -> (B, 1, S).
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))    # (B, n_out, hidden)

        # Per-class dot product: class o uses only row o of out.weight.
        # self.out(ctx) would give (B, n_out, n_out) and mix the classes.
        logits = (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias

        return logits, att


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MultiQueryAttentionPool(dim=4, n_slot=6, n_out=12, hidden=16)
    x = torch.randn(3, 6, 4)          # 3 studies, 6 slots, dim=4
    mask = torch.ones(3, 6)
    mask[0, -2:] = 0.0                # first study is missing its last 2 slots
    logits, att = model(x, mask)
    print("logits shape (expected (3, 12)):", logits.shape)
    print("att shape (expected (3, 12, 6)):", att.shape)
    print("attention sum per (study, target) - should be ~1.0:")
    print(att.sum(-1))

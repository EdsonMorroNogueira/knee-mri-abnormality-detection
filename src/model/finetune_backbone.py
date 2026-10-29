"""Fine-tunable DINOv2 backbone: only the last block (plus final layernorm) is unfrozen.

Unfreezing just the last block keeps the trainable parameter count small, which
matters with so few labelled studies. Unlike the frozen-encoder variant, the
forward pass here runs WITH gradient so it reaches the unfrozen block.
Introduced in Stage 4.
"""

import numpy as np
import torch

UNFREEZE_LAST = 1  # only the last block


def build_finetune_model(pretrained_path: str = "facebook/dinov2-small"):
    """Load a frozen DINOv2 and reopen the last block(s) and final layernorm for training.

    `pretrained_path` is a Hugging Face Hub id (default) or a local path, for
    offline inference notebooks where the weights ship as a separate dataset."""
    model = load_frozen_dinov2(pretrained_path)
    n_layer = len(model.encoder.layer)

    # The final layernorm is always unfrozen: it processes the last block's output.
    for blk in model.encoder.layer[max(0, n_layer - UNFREEZE_LAST) :]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.layernorm.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_layer} blocks total, last {UNFREEZE_LAST} trainable " f"({n_trainable / 1e6:.2f}M parameters)")
    return model


def forward_batch(model, pool, cache_uint8: np.ndarray, mask: np.ndarray):
    """Run cached pixels through DINOv2 (with gradient in unfrozen blocks), then through the pool."""
    n_studies, n_slot = mask.shape
    dim = model.config.hidden_size
    present = mask.reshape(-1) > 0.5

    flat_pixels = cache_uint8.reshape(n_studies * n_slot, 3, 224, 224)[present]
    batch = (flat_pixels.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    batch_tensor = torch.tensor(batch, dtype=torch.float32)

    cls_tokens = model(pixel_values=batch_tensor).last_hidden_state[:, 0, :]

    embeddings_flat = torch.zeros(n_studies * n_slot, dim)
    embeddings_flat[present] = cls_tokens
    embeddings = embeddings_flat.reshape(n_studies, n_slot, dim)

    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    logits, att = pool(embeddings, mask_tensor)
    return logits, att


if __name__ == "__main__":
    model = build_finetune_model()
    pool = MultiQueryAttentionPool(dim=model.config.hidden_size, n_slot=6, n_out=12, hidden=64)

    # Synthetic cache: checks that gradient reaches the encoder, no real DICOM needed.
    rng = np.random.default_rng(0)
    fake_cache = rng.integers(0, 256, size=(3, 6, 3, 224, 224), dtype=np.uint8)
    fake_mask = np.ones((3, 6), dtype=np.float32)
    fake_mask[0, -2:] = 0.0

    logits, att = forward_batch(model, pool, fake_cache, fake_mask)
    loss = logits.sum()
    loss.backward()

    last_block_grad = list(model.encoder.layer[-1].parameters())[0].grad
    first_block_grad = list(model.encoder.layer[0].parameters())[0].grad
    print(f"logits shape: {logits.shape} (expected (3, 12))")
    print(f"gradient reached LAST block? {last_block_grad is not None}")
    print(f"gradient reached FIRST block (should be False - frozen)? {first_block_grad is not None}")

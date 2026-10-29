"""ResNet50 backbone via timm (ImageNet-pretrained), partially unfrozen, for ensemble diversity.

A CNN is architecturally different from DINOv2 (ViT), so the ensemble gains
architecture diversity rather than only seed diversity. Requires `pip install timm`
and MultiQueryAttentionPool in scope. Added in Stage 16.
"""
import torch
import torch.nn as nn


def build_finetune_timm_backbone(model_name: str = "resnet50"):
    """Load a timm backbone with `num_classes=0` (global-average-pooled vector per image).

    Same output interface as DINOv2's CLS token: (n_images, dim)."""
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)

    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only layer4[-1] (~21% of params), mirroring DINOv2's last block
    # (~24%); all of layer4 would be ~65%. global_pool has no parameters.
    for p in model.layer4[-1].parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: {n_trainable / 1e6:.2f}M trainable parameters")
    return model


if __name__ == "__main__":
    model = build_finetune_timm_backbone()
    dim = model.num_features
    print(f"backbone output dim: {dim} (expected 2048 for resnet50)")

    # Smoke test: one synthetic forward pass yields a (B, dim) vector per image.
    fake_batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(fake_batch)
    assert out.shape == (2, dim), f"expected (2, {dim}), got {out.shape}"
    print(f"[OK] forward output shape {out.shape}")

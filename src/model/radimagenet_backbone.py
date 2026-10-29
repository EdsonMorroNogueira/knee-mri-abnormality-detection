"""ResNet50 pretrained on RadImageNet (~1.35M CT/MRI/ultrasound images), partially unfrozen.

Same architecture as the timm ResNet50; only the pretraining source changes to a
radiology-specific one. Weights: Lab-Rasool/RadImageNet on Hugging Face (MIT, no
sign-up; the official release needs approval and is TensorFlow/Keras).
Unverified risk: the checkpoint's original mean/std are unknown, so ImageNet
mean/std are used as a first approximation. Requires internet and
MultiQueryAttentionPool in scope. Added in Stage 16.
"""
import torch
import torch.nn as nn


class RadImageNetResNet50(nn.Module):
    """Mirror of the Lab-Rasool/RadImageNet checkpoint layout.

    Saved keys (`backbone.0.weight` ... `backbone.7.2.bn3...`) do not match a stock
    torchvision resnet50; they match nn.Sequential(*resnet50().children()) without
    the last two modules (avgpool, fc). The parameter-free avgpool is added outside."""

    def __init__(self):
        super().__init__()
        import torchvision

        base = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 2048

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def build_finetune_radimagenet_backbone():
    """Load the RadImageNet ResNet50 and unfreeze only the last Bottleneck of layer4.

    `backbone[-1]` is layer4 and its `[-1]` is the last block."""
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(repo_id="Lab-Rasool/RadImageNet", filename="ResNet50.pt")
    loaded = torch.load(weights_path, map_location="cpu", weights_only=False)

    model = RadImageNetResNet50()
    if isinstance(loaded, nn.Module):
        # Full module rather than a state_dict: reuse its weights if the layout matches.
        model.load_state_dict(loaded.state_dict())
    else:
        model.load_state_dict(loaded)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.backbone[-1][-1].parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet50 (RadImageNet): {n_trainable / 1e6:.2f}M trainable parameters")
    return model


if __name__ == "__main__":
    model = build_finetune_radimagenet_backbone()
    dim = model.num_features
    print(f"backbone output dim: {dim} (expected 2048)")

    fake_batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(fake_batch)
    assert out.shape == (2, dim), f"expected (2, {dim}), got {out.shape}"
    print(f"[OK] forward output shape {out.shape}")

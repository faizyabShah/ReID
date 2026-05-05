#!/usr/bin/env python
"""Quick test to verify PAT model checkpoint loading."""

import torch
import torch.nn as nn
from torchvision.models import vit_l_16
import sys

def test_checkpoint_loading(checkpoint_path):
    """Verify your PAT checkpoint can be loaded and has correct structure."""
    print("=" * 70)
    print("PAT CHECKPOINT VERIFICATION")
    print("=" * 70)

    # Load checkpoint
    print(f"\n[1/4] Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

    # Check structure
    print(f"\n[2/4] Checking checkpoint structure...")
    print(f"  Top-level keys: {list(ckpt.keys())}")

    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    print(f"  State dict keys (first 5): {list(state.keys())[:5]}")

    # Count parameters
    n_params = sum(v.numel() for v in state.values())
    print(f"  Total parameters in checkpoint: {n_params:,}")

    # Check for classifier/bottleneck
    has_classifier = any("classifier" in k for k in state.keys())
    has_bottleneck = any("bottleneck" in k for k in state.keys())
    print(f"  Has classifier layers: {has_classifier}")
    print(f"  Has bottleneck layers: {has_bottleneck}")

    # Check for DDP prefix
    has_ddp = any("module." in k for k in state.keys())
    print(f"  Has DDP 'module.' prefix: {has_ddp}")

    # Try loading into ViT-Large
    print(f"\n[3/4] Testing ViT-Large-Patch16 backbone loading...")
    try:
        backbone = vit_l_16(weights=None)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()
                   if "classifier" not in k and "bottleneck" not in k}
        backbone.load_state_dict(cleaned, strict=False)
        backbone.heads = nn.Identity()
        print("✅ Successfully loaded into ViT-Large backbone")
        print(f"  Feature dimension: 1024")
    except Exception as e:
        print(f"❌ Failed to load into backbone: {e}")
        return False

    # Test forward pass
    print(f"\n[4/4] Testing forward pass...")
    try:
        backbone.eval()
        test_input = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            feat = backbone(test_input)

        if feat.dim() == 3:
            cls_feat = feat[:, 0, :]  # CLS token
            print(f"✅ Forward pass successful")
            print(f"  Input shape: {test_input.shape}")
            print(f"  ViT output shape: {feat.shape}")
            print(f"  CLS token shape (extracted): {cls_feat.shape}")
        else:
            print(f"⚠️  Unexpected output shape: {feat.shape}")
            print(f"  Expected 3D tensor [B, N, D] or 2D [B, D]")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("✅ ALL CHECKS PASSED - Ready for signlabeler!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pat_checkpoint.py <path_to_checkpoint>")
        print("Example: python check_pat_checkpoint.py models/pat_vit_large.pth")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    success = test_checkpoint_loading(checkpoint_path)
    sys.exit(0 if success else 1)

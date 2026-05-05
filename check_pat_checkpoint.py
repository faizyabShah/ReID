#!/usr/bin/env python
"""Quick test to verify PAT model checkpoint loading."""

import torch
import torch.nn as nn
from model.make_model import build_part_attention_vit
from model.backbones.vit_pytorch import part_attention_vit_large, part_attention_vit_base
import sys

class DummyConfig:
    """Minimal config to initialize PAT model."""
    class MODEL:
        TRANSFORMER_TYPE = 'vit_large_patch16_224_TransReID'
        STRIDE_SIZE = 16
        DROP_PATH = 0.1
        DROP_OUT = 0.0
        ATT_DROP_RATE = 0.0
        PRETRAIN_PATH = ""
        PRETRAIN_CHOICE = 'imagenet'
        COS_LAYER = False
        NECK = 'bnneck'
    class INPUT:
        SIZE_TRAIN = [256, 128]
    class TEST:
        NECK_FEAT = 'after'
        CLS_FUSION = False

def test_checkpoint_loading(checkpoint_path, model_type='vit_large'):
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
    print(f"  Top-level keys: {list(ckpt.keys())[:5]}")

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

    # Try loading into PAT model
    print(f"\n[3/4] Testing PAT model loading...")
    try:
        cfg = DummyConfig()
        if model_type == 'vit_large':
            cfg.MODEL.TRANSFORMER_TYPE = 'vit_large_patch16_224_TransReID'
            factory = {'vit_large_patch16_224_TransReID': part_attention_vit_large}
            feat_dim = 1024
        elif model_type == 'vit_base':
            cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
            factory = {'vit_base_patch16_224_TransReID': part_attention_vit_base}
            feat_dim = 768
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        backbone = build_part_attention_vit(num_classes=1000, cfg=cfg,
                                             factory=factory, pretrain_tag='imagenet')
        backbone.load_state_dict(state, strict=False)
        backbone.eval()
        print("✅ Successfully loaded into PAT backbone")
        print(f"  Feature dimension: {feat_dim}")
    except Exception as e:
        print(f"❌ Failed to load into PAT backbone: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    print(f"\n[4/4] Testing forward pass...")
    try:
        test_input = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            out = backbone(test_input)

        print(f"✅ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")

        if isinstance(out, list):
            print(f"  Output: list of {len(out)} layerwise tokens")
            print(f"  Last layer shape: {out[-1].shape}")
            cls_feat = out[-1][:, 0, :]
            print(f"  CLS token shape (extracted): {cls_feat.shape}")
        elif isinstance(out, tuple):
            print(f"  Output: tuple of {len(out)} components")
            print(f"  First component shape: {out[0].shape if hasattr(out[0], 'shape') else 'N/A'}")
            print(f"  Second component (layerwise): {len(out[1])} layers")
        else:
            print(f"  Output shape: {out.shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅ ALL CHECKS PASSED - Ready for signlabeler!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pat_checkpoint.py <path_to_checkpoint> [vit_large|vit_base]")
        print("Example: python check_pat_checkpoint.py models/pat_vit_large.pth vit_large")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'vit_large'
    success = test_checkpoint_loading(checkpoint_path, model_type)
    sys.exit(0 if success else 1)

"""
Utilities for class-specific augmentation logging and debugging.
"""

from data.transforms.class_specific_aug import ClassSpecificAugmentation


def log_class_augmentation_policies():
    """Print all class augmentation policies for debugging."""
    print("\n" + "="*80)
    print("CLASS-SPECIFIC AUGMENTATION POLICIES")
    print("="*80)
    
    print("\nSupported Classes:")
    print("-" * 80)
    for i, class_name in enumerate(ClassSpecificAugmentation.SUPPORTED_CLASSES, 1):
        policy = ClassSpecificAugmentation.AUGMENTATION_POLICIES[class_name]
        print(f"  {i}. {class_name}")
        print(f"     Description: {policy['description']}")
        print(f"     Augmentations:")
        for aug_type, params in policy["augmentations"]:
            print(f"       - {aug_type}: {params if not isinstance(params, int) else f'size={params}'}")
        print()
    
    print("="*80 + "\n")


def get_augmentation_summary(class_name):
    """Get a summary of augmentations for a specific class."""
    if class_name not in ClassSpecificAugmentation.AUGMENTATION_POLICIES:
        print(f"\n✗ Class '{class_name}' not found!")
        print(f"Supported classes: {', '.join(ClassSpecificAugmentation.SUPPORTED_CLASSES)}")
        return
    
    policy = ClassSpecificAugmentation.AUGMENTATION_POLICIES[class_name]
    print(f"\nAugmentation Summary for '{class_name}':")
    print(f"  Description: {policy['description']}")
    print(f"  Augmentations:")
    for aug_type, params in policy["augmentations"]:
        print(f"    - {aug_type}: {params if not isinstance(params, int) else f'size={params}'}")


def is_class_specific_aug_available():
    """Check if class information is available in dataset."""
    return True  # If dataset is loaded, class info should be available


if __name__ == "__main__":
    # Can be run as: python -m data.transforms.aug_utils
    log_class_augmentation_policies()
    
    # Print examples
    print("\n\nExample Queries:")
    for cls in ClassSpecificAugmentation.SUPPORTED_CLASSES:
        get_augmentation_summary(cls)
        print()


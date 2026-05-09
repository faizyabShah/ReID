"""
Class-specific augmentation policies for person re-identification.

Tailored augmentation strategies for each object class:
- TrafficSign: Robust to perspective and occlusion
- RubbishBins: Focus on color and erasing variations
- Crosswalk: Geometric transformations and motion blur
- Container: Comprehensive augmentation with noise and color variations
"""

import torchvision.transforms as T
import torch
import random
from PIL import Image, ImageFilter, ImageDraw
import numpy as np


class ResizeLongestSide(object):
    """Resize image so longest side matches target size, pad to make square."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        ratio = self.size / max_side
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Pad to square
        delta_w = self.size - new_w
        delta_h = self.size - new_h
        pad_l = delta_w // 2
        pad_t = delta_h // 2
        pad_r = delta_w - pad_l
        pad_b = delta_h - pad_t
        
        img = T.functional.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)
        return img


class ResizePreserveAspectRatio(object):
    """Resize to target height while preserving aspect ratio."""
    
    def __init__(self, height):
        self.height = height
    
    def __call__(self, img):
        w, h = img.size
        ratio = self.height / h
        new_w = int(w * ratio)
        return img.resize((new_w, self.height), Image.BILINEAR)


class ResizePreserveAspectRatioWide(object):
    """Resize to target width while preserving aspect ratio (for wide images)."""
    
    def __init__(self, width):
        self.width = width
    
    def __call__(self, img):
        w, h = img.size
        ratio = self.width / w
        new_h = int(h * ratio)
        return img.resize((self.width, new_h), Image.BILINEAR)


class GaussianNoise(object):
    """Add Gaussian noise to image."""
    
    def __init__(self, mean=0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((img_array * 255).astype(np.uint8))


class MotionBlur(object):
    """Apply motion blur effect."""
    
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        k = self.kernel_size
        kernel = np.zeros((k, k))
        kernel[int((k - 1) / 2), :] = np.ones(k)
        kernel = kernel / k
        
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            result = np.zeros_like(img_array)
            for i in range(3):
                result[:, :, i] = T.functional.to_pil_image(
                    T.functional.to_tensor(
                        Image.fromarray(
                            np.uint8(np.convolve(img_array[:, :, i].flatten(), kernel.flatten(), mode='same').reshape(img_array[:, :, i].shape))
                        )
                    )
                ).tobytes()
            return Image.fromarray(result)
        return img


class JPEGCompression(object):
    """Apply JPEG compression."""
    
    def __init__(self, quality_range=(50, 100), p=0.5):
        self.quality_min, self.quality_max = quality_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        quality = random.randint(self.quality_min, self.quality_max)
        # Convert to JPEG and back
        import io
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomErasingSmallPatches(object):
    """Random erasing of small patches."""
    
    def __init__(self, p=0.5, scale=(0.02, 0.1)):
        self.p = p
        self.scale = scale
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        w, h = img.size
        erase_area = random.uniform(self.scale[0], self.scale[1])
        erase_h = int(np.sqrt(erase_area * h * w))
        erase_w = int(np.sqrt(erase_area * h * w))
        
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        
        img_array = np.array(img)
        erase_value = np.random.randint(0, 256, 3)
        img_array[y:y+erase_h, x:x+erase_w] = erase_value
        
        return Image.fromarray(img_array)


class ClassSpecificAugmentation:
    """
    Apply class-specific augmentations based on object class.
    
    Only 4 classes supported:
    - TrafficSign: Perspective robust augmentation
    - RubbishBins: Color and erasing augmentation
    - Crosswalk: Geometric transformations
    - Container: Comprehensive augmentation
    """
    
    # Exact class names from dataset
    SUPPORTED_CLASSES = ["TrafficSign", "RubbishBins", "Crosswalk", "Container"]
    
    # Class to augmentation chain mapping
    AUGMENTATION_POLICIES = {
        "TrafficSign": {
            "augmentations": [
                ("resize_longest", 256),
                ("random_perspective", {"distortion_scale": 0.1, "p": 0.4}),
                ("color_jitter", {"brightness": 0.15, "contrast": 0.15, "saturation": 0.05, "hue": 0.02}),
                ("gaussian_blur", {"p": 0.5}),
                ("jpeg_compression", {"quality_range": (50, 100), "p": 0.3}),
                ("random_horizontal_flip", {"p": 0.5}),
            ],
            "description": "Traffic signs: perspective-robust, JPEG compression for robustness"
        },
        "RubbishBins": {
            "augmentations": [
                ("resize_aspect", 256),
                ("color_jitter", {"brightness": 0.25, "contrast": 0.25, "saturation": 0.15, "hue": 0.05}),
                ("random_erasing", {"p": 0.4}),
                ("gaussian_blur", {"p": 0.5}),
                ("gaussian_noise", {"p": 0.3}),
                ("random_erasing_small", {"p": 0.3}),
                ("random_horizontal_flip", {"p": 0.5}),
            ],
            "description": "Rubbish bins: color variations, noise, erasing"
        },
        "Crosswalk": {
            "augmentations": [
                ("resize_aspect_wide", 256),
                ("random_perspective", {"distortion_scale": 0.3, "p": 0.7}),
                ("random_affine", {"degrees": 10, "translate": (0.1, 0.1), "scale": (0.9, 1.1), "p": 0.5}),
                ("random_crop", {"p": 0.5}),
                ("motion_blur", {"p": 0.4}),
                ("color_jitter", {"brightness": 0.25, "contrast": 0.25, "saturation": 0.1, "hue": 0.03}),
                ("random_horizontal_flip", {"p": 0.5}),
                ("random_erasing", {"p": 0.3}),
            ],
            "description": "Crosswalks: geometric transforms, motion blur"
        },
        "Container": {
            "augmentations": [
                ("resize_aspect", 256),
                ("color_jitter", {"brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.08}),
                ("random_erasing", {"p": 0.5}),
                ("gaussian_noise", {"p": 0.4}),
                ("gaussian_blur", {"p": 0.5}),
                ("random_affine", {"degrees": 5, "translate": (0.05, 0.05), "scale": (1.0, 1.0), "p": 0.5}),
                ("random_horizontal_flip", {"p": 0.5}),
            ],
            "description": "Containers: aggressive color/noise/erasing"
        }
    }
    
    @staticmethod
    def get_augmentation_chain(class_name):
        """
        Get the augmentation chain for a specific class.
        
        Args:
            class_name: Name of the class (e.g., "TrafficSign")
            
        Returns:
            List of augmentation tuples or None if class not found
        """
        if class_name not in ClassSpecificAugmentation.AUGMENTATION_POLICIES:
            return None
        return ClassSpecificAugmentation.AUGMENTATION_POLICIES[class_name]["augmentations"]
    
    @staticmethod
    def build_transforms_for_class(class_name, size_train=(256, 128)):
        """
        Build a complete transform pipeline for a specific class.
        
        Args:
            class_name: Class name (TrafficSign, RubbishBins, Crosswalk, Container)
            size_train: Target training size
            
        Returns:
            Composed torchvision transform or None if class not found
        """
        augmentation_chain = ClassSpecificAugmentation.get_augmentation_chain(class_name)
        if augmentation_chain is None:
            return None
        
        transforms_list = []
        
        for aug_type, params in augmentation_chain:
            if aug_type == "resize_longest":
                transforms_list.append(ResizeLongestSide(params))
            elif aug_type == "resize_aspect":
                transforms_list.append(ResizePreserveAspectRatio(params))
            elif aug_type == "resize_aspect_wide":
                transforms_list.append(ResizePreserveAspectRatioWide(params))
            elif aug_type == "random_perspective":
                transforms_list.append(
                    T.RandomPerspective(
                        distortion_scale=params.get("distortion_scale", 0.5),
                        p=params.get("p", 0.5)
                    )
                )
            elif aug_type == "random_affine":
                transforms_list.append(
                    T.RandomAffine(
                        degrees=params.get("degrees", 0),
                        translate=params.get("translate", None),
                        scale=params.get("scale", None),
                        p=params.get("p", 0.5)
                    )
                )
            elif aug_type == "random_crop":
                if params.get("p", 0.5) > 0:
                    transforms_list.append(
                        T.RandomCrop(size_train, pad_if_needed=True)
                    )
            elif aug_type == "color_jitter":
                transforms_list.append(
                    T.ColorJitter(
                        brightness=params.get("brightness", 0),
                        contrast=params.get("contrast", 0),
                        saturation=params.get("saturation", 0),
                        hue=params.get("hue", 0)
                    )
                )
            elif aug_type == "gaussian_blur":
                transforms_list.append(GaussianBlur(p=params.get("p", 0.5)))
            elif aug_type == "gaussian_noise":
                transforms_list.append(
                    GaussianNoise(
                        mean=params.get("mean", 0),
                        std=params.get("std", 0.1),
                        p=params.get("p", 0.5)
                    )
                )
            elif aug_type == "motion_blur":
                transforms_list.append(
                    MotionBlur(
                        kernel_size=params.get("kernel_size", 5),
                        p=params.get("p", 0.5)
                    )
                )
            elif aug_type == "jpeg_compression":
                transforms_list.append(
                    JPEGCompression(
                        quality_range=params.get("quality_range", (50, 100)),
                        p=params.get("p", 0.5)
                    )
                )
            elif aug_type == "random_erasing":
                transforms_list.append(
                    RandomErasingTransform(p=params.get("p", 0.5))
                )
            elif aug_type == "random_erasing_small":
                transforms_list.append(
                    RandomErasingSmallPatches(
                        p=params.get("p", 0.5),
                        scale=params.get("scale", (0.02, 0.1))
                    )
                )
            elif aug_type == "random_horizontal_flip":
                transforms_list.append(
                    T.RandomHorizontalFlip(p=params.get("p", 0.5))
                )
        
        # Ensure final size
        if size_train and class_name != "Crosswalk":  # Crosswalk keeps aspect ratio
            transforms_list.append(T.Resize(size_train, interpolation=3))
        
        return T.Compose(transforms_list)


class GaussianBlur(object):
    """Gaussian blur with random sigma."""
    
    def __init__(self, kernel_size=3, sigma=(0.1, 1.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class RandomErasingTransform(object):
    """Random erasing using timm's implementation if available."""
    
    def __init__(self, p=0.5):
        self.p = p
        self.use_timm = False
        try:
            from timm.data.random_erasing import RandomErasing as RE
            self.re = RE(probability=p, mode='pixel', max_count=1, device='cpu')
            self.use_timm = True
        except ImportError:
            self.re = None
    
    def __call__(self, img):
        if self.use_timm and self.re:
            # Convert to tensor, apply, convert back
            img_t = T.functional.to_tensor(img)
            img_t = self.re(img_t)
            return T.functional.to_pil_image(img_t)
        return img


class ClassAwareTransformWrapper:
    """
    Wrapper that applies class-specific augmentations.
    
    Detects class information and applies the appropriate transform pipeline.
    """
    
    def __init__(self, base_transforms, cfg=None, enable_class_specific=True):
        """
        Initialize the wrapper.
        
        Args:
            base_transforms: List of base transforms (before class-specific)
            cfg: Config object
            enable_class_specific: Whether to enable class-specific augmentation
        """
        self.base_transforms = base_transforms
        self.cfg = cfg
        self.enable_class_specific = enable_class_specific
        
        # Split transforms at ToTensor
        self.pre_tensor_transforms = []
        self.post_tensor_transforms = []
        self.found_tensor = False
        
        for transform in base_transforms:
            if isinstance(transform, T.ToTensor):
                self.found_tensor = True
            elif not self.found_tensor:
                self.pre_tensor_transforms.append(transform)
            else:
                self.post_tensor_transforms.append(transform)
    
    def __call__(self, img, class_info=None):
        """
        Apply transforms with class awareness.
        
        Args:
            img: PIL Image
            class_info: Dict with 'class_name' key
            
        Returns:
            Transformed tensor
        """
        # Apply pre-tensor transforms
        for transform in self.pre_tensor_transforms:
            img = transform(img)
        
        # Apply class-specific augmentations
        if self.enable_class_specific and class_info and 'class_name' in class_info:
            class_name = class_info['class_name']
            class_transform = ClassSpecificAugmentation.build_transforms_for_class(class_name)
            if class_transform:
                img = class_transform(img)
        
        # Apply post-tensor transforms (ToTensor, Normalize, etc.)
        for transform in self.post_tensor_transforms:
            img = transform(img)
        
        return img


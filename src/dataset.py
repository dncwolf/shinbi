"""
ImageFolder ベースの Dataset。
extract_features.py から呼び出すための生画像データセット（正規化なし）を提供する。
"""

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_raw_dataset(
    processed_dir: str | Path,
    split: str,
    image_size: int = 224,
) -> ImageFolder:
    """正規化なし（ToTensor のみ）の Dataset を返す。バックボーン別正規化を呼び出し側で行う。"""
    root = Path(processed_dir) / split
    transform = transforms.Compose([transforms.ToTensor()])
    return ImageFolder(root=str(root), transform=transform, target_transform=lambda y: 1 - y)


def get_dataset(
    processed_dir: str | Path,
    split: str,
    image_size: int = 224,
    pre_resized: bool = True,
) -> ImageFolder:
    """正規化済み Dataset（NIMA 統計）。train 時は Augmentation 付き。"""
    NIMA_MEAN = [0.5, 0.5, 0.5]
    NIMA_STD = [0.5, 0.5, 0.5]

    resize = [] if pre_resized else [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    ]

    if split == "train":
        t = transforms.Compose(resize + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NIMA_MEAN, NIMA_STD),
        ])
    else:
        t = transforms.Compose(resize + [
            transforms.ToTensor(),
            transforms.Normalize(NIMA_MEAN, NIMA_STD),
        ])

    root = Path(processed_dir) / split
    return ImageFolder(root=str(root), transform=t, target_transform=lambda y: 1 - y)

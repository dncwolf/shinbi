"""
ImageFolder ベースの Dataset。
train / val / test で transform を切り替える。
NIMA（InceptionResNetV2）の正規化統計を使用: mean=0.5, std=0.5。
preprocess_images.py 実行済みの場合は Resize をスキップ。
"""

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder

NIMA_MEAN = [0.5, 0.5, 0.5]
NIMA_STD = [0.5, 0.5, 0.5]


def get_transform(split: str, image_size: int = 224, pre_resized: bool = True) -> transforms.Compose:
    resize = [] if pre_resized else [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    ]

    if split == "train":
        return transforms.Compose(resize + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NIMA_MEAN, NIMA_STD),
        ])
    else:
        return transforms.Compose(resize + [
            transforms.ToTensor(),
            transforms.Normalize(NIMA_MEAN, NIMA_STD),
        ])


def get_dataset(
    processed_dir: str | Path,
    split: str,
    image_size: int = 224,
    pre_resized: bool = True,
) -> ImageFolder:
    root = Path(processed_dir) / split
    transform = get_transform(split, image_size, pre_resized)
    # ImageFolder はアルファベット順で favorite=0, not_favorite=1 を割り当てる。
    # BCEWithLogitsLoss の pos_weight は label=1 のクラスに作用するため、
    # favorite を label=1 に反転させて minority class を正例にする。
    return ImageFolder(root=str(root), transform=transform, target_transform=lambda y: 1 - y)

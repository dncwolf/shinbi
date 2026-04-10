"""
ImageFolder ベースの Dataset。
train / val / test で transform を切り替える。
"""

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(split: str, image_size: int = 224) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


def get_dataset(processed_dir: str | Path, split: str, image_size: int = 224) -> ImageFolder:
    root = Path(processed_dir) / split
    transform = get_transform(split, image_size)
    return ImageFolder(root=str(root), transform=transform)

"""
ImageFolder ベースの Dataset。
train / val / test で transform を切り替える。
preprocess_images.py 実行済みの場合は Resize をスキップ。
"""

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(split: str, image_size: int = 224, pre_resized: bool = True) -> transforms.Compose:
    resize = [] if pre_resized else [transforms.Resize((image_size, image_size))]

    if split == "train":
        return transforms.Compose(
            resize + [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            resize + [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


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

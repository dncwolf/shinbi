"""
data/raw/ を読み込み、非お気に入りをランダム間引きして
train / val / test に分割し data/processed/ に保存する。
"""

import random
import shutil
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    return [p for p in directory.iterdir() if p.suffix.lower() in exts]


def split_files(files: list[Path], train: float, val: float) -> tuple[list, list, list]:
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train)
    n_val = int(n * val)
    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def copy_files(files: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    fav_dir = raw_dir / cfg["data"]["raw_fav_dir"]
    not_fav_dir = raw_dir / cfg["data"]["raw_not_fav_dir"]
    target_not_fav = cfg["data"]["not_fav_target_count"]
    split = cfg["data"]["split"]

    fav_files = collect_images(fav_dir)
    not_fav_files = collect_images(not_fav_dir)

    print(f"お気に入り: {len(fav_files)}枚")
    print(f"非お気に入り（元）: {len(not_fav_files)}枚")

    # 非お気に入りを間引き
    random.seed(42)
    if len(not_fav_files) > target_not_fav:
        not_fav_files = random.sample(not_fav_files, target_not_fav)
    print(f"非お気に入り（間引き後）: {len(not_fav_files)}枚")
    print(f"合計: {len(fav_files) + len(not_fav_files)}枚")

    # 分割
    random.seed(42)
    fav_train, fav_val, fav_test = split_files(fav_files, split["train"], split["val"])
    not_fav_train, not_fav_val, not_fav_test = split_files(
        not_fav_files, split["train"], split["val"]
    )

    # processed/ を一度クリア
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    # コピー
    for subset, fav, not_fav in [
        ("train", fav_train, not_fav_train),
        ("val", fav_val, not_fav_val),
        ("test", fav_test, not_fav_test),
    ]:
        copy_files(fav, processed_dir / subset / "favorite")
        copy_files(not_fav, processed_dir / subset / "not_favorite")
        print(
            f"{subset}: favorite={len(fav)}, not_favorite={len(not_fav)}, "
            f"計={len(fav) + len(not_fav)}"
        )

    print("データ分割完了")


if __name__ == "__main__":
    main()

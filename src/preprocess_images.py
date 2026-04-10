"""
data/processed/ 以下の画像を 224x224 JPEG に一括変換して軽量化する。
split_data.py の後に一度だけ実行する。

効果:
  - 毎回の DataLoader での Resize 処理を削除
  - ファイルサイズ削減による I/O 高速化
"""

import os
from pathlib import Path

import yaml
from PIL import Image


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def convert_image(src: Path, size: int, quality: int) -> tuple[int, int]:
    """画像を resize して .jpg で上書き保存。(元サイズ, 新サイズ) をバイト数で返す。"""
    orig_size = src.stat().st_size
    img = Image.open(src).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    out_path = src.with_suffix(".jpg")
    img.save(out_path, format="JPEG", quality=quality, optimize=True)
    # 拡張子が .jpg 以外（.jpeg / .png 等）なら元ファイルを削除
    if src != out_path:
        src.unlink()
    new_size = out_path.stat().st_size
    return orig_size, new_size


def main() -> None:
    cfg = load_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    image_size = cfg["data"]["image_size"]
    quality = cfg["data"]["jpeg_quality"]

    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    image_files = [
        p for p in processed_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()
    ]

    if not image_files:
        print("画像ファイルが見つかりません。split_data.py を先に実行してください。")
        return

    print(f"変換対象: {len(image_files)}枚 → {image_size}x{image_size} JPEG (quality={quality})")

    total_orig = 0
    total_new = 0
    for i, path in enumerate(image_files, 1):
        orig, new = convert_image(path, image_size, quality)
        total_orig += orig
        total_new += new
        if i % 100 == 0 or i == len(image_files):
            print(f"  {i}/{len(image_files)} 完了")

    ratio = (1 - total_new / total_orig) * 100
    print(
        f"完了: {total_orig / 1e6:.1f} MB → {total_new / 1e6:.1f} MB "
        f"（{ratio:.1f}% 削減）"
    )


if __name__ == "__main__":
    main()

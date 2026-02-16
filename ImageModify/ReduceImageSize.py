import argparse
import os
from datetime import datetime
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def resize_images(folder_path: str, percentage: float) -> None:
    source = Path(folder_path)
    if not source.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = source / f"Reduced_{date_str}"
    output_dir.mkdir(exist_ok=True)

    scale = percentage / 100.0
    processed = 0

    for file in source.iterdir():
        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            print(f"Processing: {file.name}")
            with Image.open(file) as img:
                new_width = max(1, int(img.width * scale))
                new_height = max(1, int(img.height * scale))
                resized = img.resize((new_width, new_height), Image.LANCZOS)
                out_name = f"Modified_{processed + 1}_{new_width}x{new_height}{file.suffix}"
                resized.save(output_dir / out_name)
                processed += 1
                print(f"Resized: {file.name} -> {out_name} ({img.width}x{img.height} -> {new_width}x{new_height})")
        except Exception as e:
            print(f"Skipped: {file.name} ({e})")

    print(f"\nDone. {processed} image(s) saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a folder by a given percentage.")
    parser.add_argument("folder", help="Path to the folder containing images.")
    parser.add_argument("percentage", type=float, help="Resize percentage (e.g. 50 to halve, 200 to double).")
    args = parser.parse_args()

    resize_images(args.folder, args.percentage)

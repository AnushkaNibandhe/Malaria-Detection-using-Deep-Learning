import os
import zipfile
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image


random.seed(42)
np.random.seed(42)


def download_from_kaggle(output_dir: Path) -> None:
    """
    Download the Malaria Cell Images Dataset from Kaggle using Kaggle API.

    Dataset URL:
    https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

    This function assumes that the Kaggle API is installed and configured
    (kaggle.json with credentials is available).
    It will create parasitized.zip and uninfected.zip inside output_dir if not present.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError(
            "kaggle package is required to download the dataset.\n"
            "Install it with: pip install kaggle\n"
            "Or manually place parasitized.zip and uninfected.zip in the project directory."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset = "iarunava/cell-images-for-detecting-malaria"
    print(f"Downloading Kaggle dataset '{dataset}' ...")
    api.dataset_download_files(dataset, path=str(output_dir), unzip=True)
    print("Download complete.")


def collect_project_subset(source_dir: Path, work_dir: Path) -> None:
    """
    From the full NIH dataset under source_dir, create two zip files:
    parasitized.zip and uninfected.zip each containing 499 PNG images.

    This matches the project subset specification:
      - 499 Parasitized images (label 0)
      - 499 Uninfected images (label 1)
    """
    parasitized_source = source_dir / "cell_images" / "Parasitized"
    uninfected_source = source_dir / "cell_images" / "Uninfected"

    if not parasitized_source.exists() or not uninfected_source.exists():
        raise FileNotFoundError(
            f"Expected 'cell_images/Parasitized' and 'cell_images/Uninfected' "
            f"inside {source_dir}. Please ensure the Kaggle dataset is extracted there."
        )

    work_dir.mkdir(parents=True, exist_ok=True)

    parasitized_images = sorted(
        [p for p in parasitized_source.iterdir() if p.suffix.lower() == ".png"]
    )
    uninfected_images = sorted(
        [p for p in uninfected_source.iterdir() if p.suffix.lower() == ".png"]
    )

    if len(parasitized_images) < 499 or len(uninfected_images) < 499:
        raise ValueError(
            "Not enough images to form the 499/499 subset. "
            f"Found {len(parasitized_images)} parasitized and "
            f"{len(uninfected_images)} uninfected images."
        )

    random.shuffle(parasitized_images)
    random.shuffle(uninfected_images)

    subset_parasitized = parasitized_images[:499]
    subset_uninfected = uninfected_images[:499]

    parasitized_zip_path = work_dir / "parasitized.zip"
    uninfected_zip_path = work_dir / "uninfected.zip"

    print("Creating parasitized.zip with 499 images ...")
    with zipfile.ZipFile(parasitized_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in subset_parasitized:
            zf.write(img_path, arcname=img_path.name)

    print("Creating uninfected.zip with 499 images ...")
    with zipfile.ZipFile(uninfected_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in subset_uninfected:
            zf.write(img_path, arcname=img_path.name)

    print("Subset zips created at:", work_dir)


def extract_zips(parasitized_zip: Path, uninfected_zip: Path, extract_dir: Path) -> None:
    """Extract parasitized.zip and uninfected.zip into temporary folders."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(parasitized_zip, "r") as zf:
        zf.extractall(extract_dir / "Parasitized_all")
    with zipfile.ZipFile(uninfected_zip, "r") as zf:
        zf.extractall(extract_dir / "Uninfected_all")

    print("Extraction complete.")


def split_train_validation(extract_dir: Path, target_root: Path) -> None:
    """
    Organize images into:

    cell_images/
    |-- train/
    |   |-- Parasitized/   (399 images)
    |   |-- Uninfected/    (399 images)
    |-- validation/
        |-- Parasitized/   (100 images)
        |-- Uninfected/    (100 images)
    """
    train_par = target_root / "train" / "Parasitized"
    train_uninf = target_root / "train" / "Uninfected"
    val_par = target_root / "validation" / "Parasitized"
    val_uninf = target_root / "validation" / "Uninfected"

    for d in [train_par, train_uninf, val_par, val_uninf]:
        d.mkdir(parents=True, exist_ok=True)

    src_par = extract_dir / "Parasitized_all"
    src_uninf = extract_dir / "Uninfected_all"

    par_images = sorted([p for p in src_par.iterdir() if p.suffix.lower() == ".png"])
    uninf_images = sorted([p for p in src_uninf.iterdir() if p.suffix.lower() == ".png"])

    if len(par_images) != 499 or len(uninf_images) != 499:
        raise AssertionError(
            "Expected exactly 499 images in each extracted class subset.\n"
            f"Got {len(par_images)} Parasitized and {len(uninf_images)} Uninfected."
        )

    random.shuffle(par_images)
    random.shuffle(uninf_images)

    train_par_images = par_images[:399]
    val_par_images = par_images[399:]
    train_uninf_images = uninf_images[:399]
    val_uninf_images = uninf_images[399:]

    def copy_images(img_list, dest_dir):
        for img_path in img_list:
            shutil.copy2(img_path, dest_dir / img_path.name)

    copy_images(train_par_images, train_par)
    copy_images(val_par_images, val_par)
    copy_images(train_uninf_images, train_uninf)
    copy_images(val_uninf_images, val_uninf)

    print("Train/validation split complete.")


def split_full_dataset(source_root: Path, target_root: Path, train_ratio: float = 0.8) -> None:
    """
    From the full Kaggle dataset under source_root/cell_images, create
    train/validation splits using ALL available images.

    Final structure:
      cell_images/
      |-- train/
      |   |-- Parasitized/
      |   |-- Uninfected/
      |-- validation/
          |-- Parasitized/
          |-- Uninfected/

    The number of images per split is determined by train_ratio for each class.
    """
    parasitized_source = source_root / "cell_images" / "Parasitized"
    uninfected_source = source_root / "cell_images" / "Uninfected"

    if not parasitized_source.exists() or not uninfected_source.exists():
        raise FileNotFoundError(
            f"Expected 'cell_images/Parasitized' and 'cell_images/Uninfected' "
            f"inside {source_root}. Please ensure the Kaggle dataset is extracted there."
        )

    train_par = target_root / "train" / "Parasitized"
    train_uninf = target_root / "train" / "Uninfected"
    val_par = target_root / "validation" / "Parasitized"
    val_uninf = target_root / "validation" / "Uninfected"

    for d in [train_par, train_uninf, val_par, val_uninf]:
        d.mkdir(parents=True, exist_ok=True)

    par_images = sorted([p for p in parasitized_source.iterdir() if p.suffix.lower() == ".png"])
    uninf_images = sorted([p for p in uninfected_source.iterdir() if p.suffix.lower() == ".png"])

    random.shuffle(par_images)
    random.shuffle(uninf_images)

    def split_class(imgs):
        n_total = len(imgs)
        n_train = int(n_total * train_ratio)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]
        return train_imgs, val_imgs

    train_par_images, val_par_images = split_class(par_images)
    train_uninf_images, val_uninf_images = split_class(uninf_images)

    def copy_images(img_list, dest_dir):
        for img_path in img_list:
            shutil.copy2(img_path, dest_dir / img_path.name)

    copy_images(train_par_images, train_par)
    copy_images(val_par_images, val_par)
    copy_images(train_uninf_images, train_uninf)
    copy_images(val_uninf_images, val_uninf)

    print("Full dataset train/validation split complete.")


def compute_image_stats(root_dir: Path):
    """Compute and print dataset statistics for all PNG images under root_dir."""
    class_counts = {"Parasitized": 0, "Uninfected": 0}
    sizes = []
    channels_set = set()

    for split in ["train", "validation"]:
        for cls in ["Parasitized", "Uninfected"]:
            cls_path = root_dir / split / cls
            images = [p for p in cls_path.iterdir() if p.suffix.lower() == ".png"]
            class_counts[cls] += len(images)

            for img_path in images:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    w, h = img.size
                    sizes.append((w, h))
                    channels_set.add(len(img.getbands()))

    total_images = sum(class_counts.values())
    avg_w = int(np.mean([s[0] for s in sizes]))
    avg_h = int(np.mean([s[1] for s in sizes]))
    min_w = min(s[0] for s in sizes)
    min_h = min(s[1] for s in sizes)
    max_w = max(s[0] for s in sizes)
    max_h = max(s[1] for s in sizes)

    print("\n=== DATASET STATISTICS ===")
    print(f"Total images: {total_images}")
    for cls, count in class_counts.items():
        pct = (count / total_images) * 100 if total_images > 0 else 0.0
        print(f"  {cls}: {count} images ({pct:.2f}%)")

    print("Expected approximate 50/50 class balance.")
    print("Image format: PNG")
    print(f"Color mode: RGB, channels encountered: {channels_set}")
    print(f"Average image size: {avg_w}x{avg_h} pixels")
    print(f"Min size: {min_w}x{min_h} pixels")
    print(f"Max size: {max_w}x{max_h} pixels")
    print("Staining method: Giemsa staining (as per dataset description)")

    print("Dataset integrity check complete.")


def verify_splits(root_dir: Path) -> None:
    """Print basic train/validation split statistics for each class."""
    train_par = len(
        [p for p in (root_dir / "train" / "Parasitized").iterdir() if p.suffix.lower() == ".png"]
    )
    train_uninf = len(
        [p for p in (root_dir / "train" / "Uninfected").iterdir() if p.suffix.lower() == ".png"]
    )
    val_par = len(
        [p for p in (root_dir / "validation" / "Parasitized").iterdir() if p.suffix.lower() == ".png"]
    )
    val_uninf = len(
        [p for p in (root_dir / "validation" / "Uninfected").iterdir() if p.suffix.lower() == ".png"]
    )

    print("\n=== SPLIT COUNTS (Full Dataset) ===")
    print(f"Train - Parasitized: {train_par}")
    print(f"Train - Uninfected: {train_uninf}")
    print(f"Validation - Parasitized: {val_par}")
    print(f"Validation - Uninfected: {val_uninf}")

    total_par = train_par + val_par
    total_uninf = train_uninf + val_uninf

    if total_par > 0:
        print(f"Parasitized train ratio: {train_par / total_par:.3f}")
    if total_uninf > 0:
        print(f"Uninfected train ratio: {train_uninf / total_uninf:.3f}")

    print("Train/validation split check complete.")


def main():
    project_dir = Path(__file__).resolve().parent
    raw_data_dir = project_dir / "raw_kaggle_data"
    target_root = project_dir / "cell_images"

    source_root = raw_data_dir

    if not (source_root / "cell_images").exists():
        print("Full Kaggle data not found locally. Downloading from Kaggle...")
        download_from_kaggle(raw_data_dir)

    print("Creating train/validation splits under 'cell_images/' using full dataset ...")
    if target_root.exists():
        shutil.rmtree(target_root)
    split_full_dataset(source_root, target_root, train_ratio=0.8)

    compute_image_stats(target_root)
    verify_splits(target_root)


if __name__ == "__main__":
    main()


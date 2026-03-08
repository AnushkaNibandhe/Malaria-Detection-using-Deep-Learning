import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator


TARGET_SIZE = (64, 64)
BATCH_SIZE = 32
COLOR_MODE = "rgb"
CLASS_MODE = "binary"
RANDOM_STATE = 42


def create_generators(data_root: Path, target_size=None):
    """
    Create train and validation generators with the given target_size.
    Both use the same augmentation settings for training.
    target_size: tuple (H, W), e.g. (64, 64) for Custom CNN or (224, 224) for MobileNetV2/EfficientNetB0.
    """
    if target_size is None:
        target_size = TARGET_SIZE

    train_dir = data_root / "train"
    val_dir = data_root / "validation"

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    classes = ["Parasitized", "Uninfected"]

    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=target_size,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        classes=classes,
        shuffle=True,
        seed=RANDOM_STATE,
    )

    validation_generator = val_datagen.flow_from_directory(
        directory=str(val_dir),
        target_size=target_size,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        classes=classes,
        shuffle=False,
        seed=RANDOM_STATE,
    )

    train_samples = train_generator.samples
    val_samples = validation_generator.samples

    print(f"\n=== GENERATOR INFO (target_size={target_size}) ===")
    print(f"Number of training samples: {train_samples}")
    print(f"Number of validation samples: {val_samples}")
    print(f"Class indices: {train_generator.class_indices}")

    steps_per_epoch = train_samples // BATCH_SIZE
    val_steps = val_samples // BATCH_SIZE

    print(f"Steps per epoch (train): {steps_per_epoch}")
    print(f"Validation steps: {val_steps}")

    return train_generator, validation_generator, steps_per_epoch, val_steps


def visualize_augmentation(generator, n=5):
    batch_x, batch_y = next(generator)
    class_indices = {v: k for k, v in generator.class_indices.items()}

    plt.figure(figsize=(15, 6))
    for i in range(n):
        idx = i
        img = batch_x[idx]
        label = int(batch_y[idx])
        class_name = class_indices[label]

        # Original-style (approximation by undoing augmentation is not trivial),
        # so we simply show two augmented views of the same batch instance.

        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Augmented ({class_name})")

        # Second augmented sample
        plt.subplot(2, n, n + i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Augmented 2 ({class_name})")

    plt.tight_layout()
    plt.show()


def _get_all_images_by_class(root: Path):
    data = {"Parasitized": [], "Uninfected": []}
    for split in ["train", "validation"]:
        for cls in ["Parasitized", "Uninfected"]:
            cls_dir = root / split / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() == ".png":
                    data[cls].append(p)
    return data


def plot_sample_grid(root: Path):
    data = _get_all_images_by_class(root)
    samples = []
    for cls in ["Parasitized", "Uninfected"]:
        imgs = data[cls]
        random.shuffle(imgs)
        samples.extend([(p, cls) for p in imgs[:10]])

    plt.figure(figsize=(20, 16))
    for idx, (img_path, cls) in enumerate(samples):
        plt.subplot(4, 5, idx + 1)
        img = Image.open(img_path).convert("RGB")
        size_kb = os.path.getsize(img_path) / 1024.0
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{cls}\n{size_kb:.2f} KB")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(root: Path):
    counts = []
    for split in ["train", "validation"]:
        for cls in ["Parasitized", "Uninfected"]:
            cls_dir = root / split / cls
            n = len([p for p in cls_dir.iterdir() if p.suffix.lower() == ".png"])
            counts.append({"split": split, "class": cls, "count": n})

    import pandas as pd

    df = pd.DataFrame(counts)
    plt.figure(figsize=(8, 6))
    colors = {"Parasitized": "red", "Uninfected": "green"}
    sns.barplot(
        data=df,
        x="split",
        y="count",
        hue="class",
        palette=[colors["Parasitized"], colors["Uninfected"]],
    )

    for i, row in df.iterrows():
        plt.text(
            x=i // 2 + (-0.15 if row["class"] == "Parasitized" else 0.15),
            y=row["count"] + 1,
            s=str(row["count"]),
            ha="center",
        )

    plt.title("Class Distribution across Train and Validation Sets")
    plt.ylabel("Count")
    plt.xlabel("Split")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()


def plot_image_dimension_analysis(root: Path, sample_size: int = 100):
    data = _get_all_images_by_class(root)
    all_paths = []
    for cls, paths in data.items():
        for p in paths:
            all_paths.append((p, cls))

    random.shuffle(all_paths)
    sample_paths = all_paths[:sample_size]

    widths = []
    heights = []
    classes = []
    for img_path, cls in sample_paths:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            widths.append(w)
            heights.append(h)
            classes.append(cls)

    colors = {"Parasitized": "red", "Uninfected": "green"}
    plt.figure(figsize=(8, 6))
    for cls in ["Parasitized", "Uninfected"]:
        idxs = [i for i, c in enumerate(classes) if c == cls]
        plt.scatter(
            [widths[i] for i in idxs],
            [heights[i] for i in idxs],
            c=colors[cls],
            label=cls,
            alpha=0.7,
        )

    # Reference lines
    avg_w, avg_h = 141, 143
    min_w, min_h = 79, 82
    max_w, max_h = 247, 226

    plt.axvline(avg_w, color="blue", linestyle="--", label="Avg Width 141")
    plt.axhline(avg_h, color="purple", linestyle="--", label="Avg Height 143")
    plt.axvline(min_w, color="gray", linestyle=":", label="Min Width 79")
    plt.axhline(min_h, color="gray", linestyle=":", label="Min Height 82")
    plt.axvline(max_w, color="black", linestyle=":", label="Max Width 247")
    plt.axhline(max_h, color="black", linestyle=":", label="Max Height 226")

    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Image Dimension Analysis (Sample)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pixel_intensity_histogram(root: Path):
    data = _get_all_images_by_class(root)

    def get_samples(cls, n):
        paths = data[cls]
        random.shuffle(paths)
        return paths[:n]

    par_paths = get_samples("Parasitized", 3)
    uninf_paths = get_samples("Uninfected", 3)

    plt.figure(figsize=(15, 5))
    colors = {"Parasitized": "red", "Uninfected": "green"}
    channels = ["R", "G", "B"]

    for idx, channel in enumerate(channels):
        plt.subplot(1, 3, idx + 1)

        for cls, paths in [("Parasitized", par_paths), ("Uninfected", uninf_paths)]:
            channel_values = []
            for img_path in paths:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)
                channel_idx = idx
                channel_values.extend(arr[:, :, channel_idx].flatten())

            plt.hist(
                channel_values,
                bins=50,
                alpha=0.5,
                color=colors[cls],
                label=cls,
                density=True,
            )

        plt.title(f"{channel} Channel Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    project_root = Path(__file__).resolve().parent
    data_root = project_root / "cell_images"

    train_gen, val_gen, steps_per_epoch, val_steps = create_generators(data_root)
    visualize_augmentation(train_gen, n=5)
    plot_sample_grid(data_root)
    plot_class_distribution(data_root)
    plot_image_dimension_analysis(data_root, sample_size=100)
    plot_pixel_intensity_histogram(data_root)


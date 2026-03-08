import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image


def prepare_image(image_path_or_file, model=None):
    """
    Accepts a file path (string/Path) or a file-like object (e.g., Flask upload).
    Returns a preprocessed array of shape (1, H, W, 3) where H=W=64.
    """
    if isinstance(image_path_or_file, (str, Path)):
        img = Image.open(image_path_or_file)
    else:
        img = Image.open(image_path_or_file)

    img = img.convert("RGB")

    target_h, target_w = 64, 64
    if model is not None and hasattr(model, "input_shape"):
        # model.input_shape is typically (None, H, W, C)
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 3:
            target_h, target_w = shape[1], shape[2]

    img = img.resize((target_w, target_h), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_malaria(image_input, model):
    preprocessed = prepare_image(image_input, model=model)
    prediction = model.predict(preprocessed, verbose=0)
    pred_prob = float(prediction[0][0])

    if pred_prob < 0.5:
        class_idx = 0
        class_name = "Parasitized"
    else:
        class_idx = 1
        class_name = "Uninfected"

    parasitized_conf = round((1 - pred_prob) * 100, 2)
    uninfected_conf = round(pred_prob * 100, 2)
    confidence = max(parasitized_conf, uninfected_conf)

    result = {
        "success": True,
        "predicted_class": class_name,
        "class_index": class_idx,
        "confidence": confidence,
        "parasitized_probability": parasitized_conf,
        "uninfected_probability": uninfected_conf,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


def batch_predict(image_folder, model):
    image_folder = Path(image_folder)
    records = []

    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith(".png"):
            continue

        fpath = image_folder / fname
        start = time.time()
        result = predict_malaria(str(fpath), model)
        end = time.time()

        records.append(
            {
                "filename": fname,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "processing_time_s": round(end - start, 4),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        print("\nBatch prediction summary:")
        print(df.head())
        print("\nStatistics:")
        print(df.describe(include="all"))
    else:
        print("No PNG images found for batch prediction.")

    return df


if __name__ == "__main__":
    print(
        "This module provides prepare_image, predict_malaria, and batch_predict "
        "for use with the trained malaria detection models."
    )


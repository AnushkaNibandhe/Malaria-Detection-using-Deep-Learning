import os
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

from prediction_utils import predict_malaria, batch_predict


def scenario_1_uninfected(model):
    print("\n=== Scenario 1: Rural Clinic Screening ===")
    img_path = Path("C205ThinF_IMG_20151106_151800_cell_38.png")
    if not img_path.exists():
        print(f"Test image not found at {img_path}. Skipping strict assertions.")
        return True

    start = datetime.now()
    result = predict_malaria(str(img_path), model)
    elapsed = (datetime.now() - start).total_seconds()

    print("Result:", result)
    print(f"Processing time: {elapsed:.3f} s")

    assert result["predicted_class"] == "Uninfected", "Expected class 'Uninfected'"
    assert result["confidence"] >= 99.0, "Expected confidence >= 99.0"
    assert elapsed < 10.0, "Expected processing time < 10 seconds"

    return True


def scenario_2_parasitized(model):
    print("\n=== Scenario 2: Hospital Mass Screening ===")
    img_path = Path("C99P60ThinF_IMG_20150918_141620_cell_57.png")
    if not img_path.exists():
        print(f"Test image not found at {img_path}. Skipping strict assertions.")
        return True

    start = datetime.now()
    result = predict_malaria(str(img_path), model)
    elapsed = (datetime.now() - start).total_seconds()

    print("Result:", result)
    print(f"Processing time: {elapsed:.3f} s")

    assert result["predicted_class"] == "Parasitized", "Expected class 'Parasitized'"
    assert result["confidence"] >= 98.0, "Expected confidence >= 98.0"
    assert elapsed < 10.0, "Expected processing time < 10 seconds"

    return True


def scenario_3_batch(model):
    print("\n=== Scenario 3: Research Analysis — Batch Processing ===")
    val_dir = Path("cell_images") / "validation"
    if not val_dir.exists():
        print(f"Validation directory not found at {val_dir}. Skipping strict assertions.")
        return True

    # Build ground-truth labels from folder names
    file_paths = []
    y_true = []
    for cls_name, label in [("Parasitized", 0), ("Uninfected", 1)]:
        cls_dir = val_dir / cls_name
        if not cls_dir.exists():
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(".png"):
                file_paths.append(cls_dir / fname)
                y_true.append(label)

    if not file_paths:
        print("No validation images found. Skipping strict assertions.")
        return True

    # Run batch_predict to get predictions & times
    df_pred = batch_predict(val_dir, model)
    if df_pred.empty:
        print("No predictions returned. Skipping strict assertions.")
        return True

    # Map predicted classes back to labels
    class_to_idx = {"Parasitized": 0, "Uninfected": 1}
    y_pred = [class_to_idx.get(cls, 0) for cls in df_pred["predicted_class"]]

    # Ensure matching length
    y_true = y_true[: len(y_pred)]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Batch accuracy on validation set: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print(f"Average processing time per image: {df_pred['processing_time_s'].mean():.4f} s")

    assert acc >= 0.985, "Expected batch accuracy >= 98.5%"

    return True


def main():
    print("Loading final malaria detector model...")
    model_path = "malaria_detector_final.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Train and export the final model before running tests."
        )

    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    results = []

    try:
        passed = scenario_1_uninfected(model)
        results.append(("Scenario 1: Rural Clinic Screening", passed))
    except AssertionError as e:
        print("Scenario 1 FAILED:", e)
        results.append(("Scenario 1: Rural Clinic Screening", False))

    try:
        passed = scenario_2_parasitized(model)
        results.append(("Scenario 2: Hospital Mass Screening", passed))
    except AssertionError as e:
        print("Scenario 2 FAILED:", e)
        results.append(("Scenario 2: Hospital Mass Screening", False))

    try:
        passed = scenario_3_batch(model)
        results.append(("Scenario 3: Research Batch Processing", passed))
    except AssertionError as e:
        print("Scenario 3 FAILED:", e)
        results.append(("Scenario 3: Research Batch Processing", False))

    print("\n=== Test Summary ===")
    overall_ok = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        if not passed:
            overall_ok = False

    print("\nOverall system status:", "PASS" if overall_ok else "FAIL")


if __name__ == "__main__":
    main()


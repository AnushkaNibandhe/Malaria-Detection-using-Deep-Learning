import warnings
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

import zipfile
import shutil
import joblib
from datetime import datetime
from PIL import Image


def configure_environment():
    """Configure warnings, seeds, and print environment info."""
    warnings.filterwarnings("ignore")
    tf.random.set_seed(42)
    np.random.seed(42)
    random_state = 42

    print("=== Library Imports Successful ===")
    print("Data Manipulation: numpy, pandas")
    print("Visualization: matplotlib, seaborn")
    print("Deep Learning: tensorflow / keras")
    print("Metrics: sklearn.metrics")
    print("Utilities: os, zipfile, shutil, warnings, joblib, datetime, PIL.Image")

    print("\nTensorFlow configuration:")
    print(f"TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU(s) available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("No GPU detected. Using CPU.")

    print(f"\nRANDOM_STATE set to: {random_state}")
    return random_state


if __name__ == "__main__":
    configure_environment()


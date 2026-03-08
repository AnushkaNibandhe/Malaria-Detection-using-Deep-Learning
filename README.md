# Malaria Detection System

A deep learning and computer vision project designed to automatically classify cell images as either **Parasitized** or **Uninfected** with malaria. By leveraging powerful convolutional neural networks (CNNs), the system provides a fast, reliable, and scalable automated screening tool to assist in medical diagnostics. 

The project includes scripts for end-to-end processing: down to downloading and preparing the dataset, building and comparing custom CNN architectures against transfer learning models (MobileNetV2, EfficientNetB0), training pipelines, and a Flask-based web interface for real-time model inference.

## 🗂️ Project Structure

- `app.py`: The entry point for the Flask web application. It serves the trained model (`.h5` files) and exposes a clean user interface and an API endpoint (`/predict`) for classification.
- `models_training.py`: The core model training pipeline. It builds, trains, evaluates, and compares multiple models (Custom CNN, MobileNetV2, EfficientNetB0) using data generators. Evaluates performance using ROC-AUC, Precision, Recall, F1-score, and Accuracy, and automatically saves the best performing model as `malaria_detector_final.h5`.
- `data_preprocessing.py`: Utilities for setting up image data generators (e.g., resizing to `64x64` for the custom CNN and `224x224` for transfer learning models) and applying data augmentation techniques.
- `dataset_setup.py`: Automates the downloading, extracting, and directory preparation of the Kaggle Malaria Cell Images Dataset into structured `train` and `val` directories.
- `env_setup.py`: Environment configuration script ensuring reproducible random seeds across TensorFlow and NumPy and verifying required libraries and physical GPU devices are available.
- `prediction_utils.py`: Helper functions for standardizing, resizing, and preparing singular images for accurate inference via the model.
- `test_system.py`: Script to systematically test the Flask system's API endpoints and functionalities.
- `templates/`: Contains the `index.html` frontend for the Flask web application.

---

## ⚙️ Dependencies and Compatible Versions

This project relies on several data science, machine learning, and web framework libraries. The system is designed to run efficiently on **Python 3.9 to 3.11**. 

Below are the carefully selected, stable, and compatible versions for the libraries used in this project:

- **Flask (3.0.0)** - For serving the web application.
- **numpy (1.24.3)** - For efficient array and matrix operations (compatible with TF 2.15).
- **pandas (2.1.1)** - For data manipulation and metric formatting.
- **matplotlib (3.8.0)** - For plotting training history, ROC curves, and other visual metrics.
- **seaborn (0.13.0)** - For generating visually appealing confusion matrix heatmaps.
- **tensorflow (2.15.0)** - Core deep learning framework used for constructing the Custom CNN, MobileNetV2, and EfficientNetB0 models.
- **scikit-learn (1.3.1)** - Utilized for precision, recall, F1, ROC-AUC calculations, and computing classification reports.
- **Pillow (10.0.1)** - Essential for image opening, resizing, and array conversion in the Flask application.
- **joblib (1.3.2)** - Standard utility library (often a dependency for scikit-learn).

You can automatically install all compatible requirements using the given `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started & How to Run

### 1. Preparing and Training the Model
If you want to prepare the dataset and train the models from scratch, execute:
```bash
python models_training.py
```
**What this script does:**
1. **Dataset Prep:** Automatically handles the Kaggle malaria cell dataset.
2. **Model Training:** Trains a Custom CNN, MobileNetV2, and EfficientNetB0.
3. **Evaluation:** Evaluates their performance via ROC-AUC and Accuracy, generating plots (`model_comparison_roc.png`, `confusion_matrices.png`, and training histories).
4. **Export:** Compares metric scores and saves the highest-performing model as `malaria_detector_final.h5` in the root directory.

### 2. Running the Web Application
Once the `malaria_detector_final.h5` (or `best_malaria_cnn.h5`) model is generated or placed in the root folder, launch the Flask application:
```bash
python app.py
```
- Open your terminal and wait for the application to output that it is running on port 5000.
- Navigate to `http://127.0.0.1:5000/` in your standard web browser.
- You will be presented with the Malaria Detection System interface allowing you to verify the application health and upload cell images for real-time predictions.

---

## 🌐 API Endpoints documentation

The web application exposes a standard POST endpoint for programmatic access:

### `POST /predict`
Uploads an image file and returns a JSON payload with the classification result and confidence.

**Request:** 
Form-data containing a file under the key `file`.

**Successful Response Example:**
```json
{
  "success": true,
  "predicted_class": "Parasitized",
  "confidence": 98.45,
  "parasitized_probability": 98.45,
  "uninfected_probability": 1.55,
  "processing_time": 0.045,
  "timestamp": "2026-03-08 20:05:12"
}
```

### `GET /health`
Returns the status of the server and whether the deep learning model has loaded successfully into memory.

**Response Example:**
```json
{
  "status": "running",
  "model_loaded": true
}
```

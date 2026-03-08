from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from PIL import Image
import io

from prediction_utils import prepare_image


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

model = None
model_error = None
CLASS_NAMES = ["Parasitized", "Uninfected"]


def load_model_once():
    global model, model_error
    if model is not None or model_error is not None:
        return

    try:
        candidates = []
        cwd_files = os.listdir(".")

        priority_order = ["malaria_detector_final.h5", "best_malaria_cnn.h5"]
        for fname in priority_order:
            if fname in cwd_files:
                candidates.append(fname)

        if not candidates:
            for fname in cwd_files:
                if fname.lower().endswith(".h5"):
                    candidates.append(fname)

        if not candidates:
            raise FileNotFoundError("No .h5 model files found in current directory.")

        model_path = candidates[0]
        print(f"Loading model from: {model_path}")
        loaded_model = tf.keras.models.load_model(model_path)

        loaded_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        print("Model loaded successfully.")
        print(f"Model input shape: {loaded_model.input_shape}")
        print(f"Model output shape: {loaded_model.output_shape}")

        model = loaded_model
        model_error = None
    except Exception as e:
        model = None
        model_error = str(e)
        print(f"Error loading model: {model_error}")


load_model_once()


def prepare_image_flask(file_storage):
    img_bytes = file_storage.read()
    file_storage.seek(0)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")

    target_h, target_w = 64, 64
    if model is not None and hasattr(model, "input_shape"):
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 3:
            target_h, target_w = shape[1], shape[2]

    img = img.resize((target_w, target_h), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/predict", methods=["POST"])
def predict():
    global model, model_error

    if model is None:
        load_model_once()
    if model is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Model not loaded: {model_error}",
                }
            ),
            400,
        )

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    try:
        start_time = datetime.now()
        preprocessed_img = prepare_image_flask(file)
        pred = model.predict(preprocessed_img, verbose=0)
        print(f"Prediction array: {pred}, shape: {pred.shape}")

        pred_prob = float(pred[0][0])
        if pred_prob < 0.5:
            class_name = "Parasitized"
        else:
            class_name = "Uninfected"

        parasitized_prob = round((1 - pred_prob) * 100, 2)
        uninfected_prob = round(pred_prob * 100, 2)
        confidence = max(parasitized_prob, uninfected_prob)

        processing_time = (datetime.now() - start_time).total_seconds()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify(
            {
                "success": True,
                "predicted_class": class_name,
                "confidence": confidence,
                "parasitized_probability": parasitized_prob,
                "uninfected_probability": uninfected_prob,
                "processing_time": processing_time,
                "timestamp": timestamp,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    template_path = os.path.join("templates", "index.html")
    model_loaded = model is not None

    if os.path.exists(template_path):
        return render_template(
            "index.html",
            model_loaded=model_loaded,
            model_error=model_error,
        )

    status_color = "#4caf50" if model_loaded else "#f44336"
    status_text = "Model Loaded" if model_loaded else "Model Load Failed"
    error_html = f"<p style='color:#ffebee'>{model_error}</p>" if model_error else ""

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Malaria Detection System</title>
      <style>
        body {{
          margin: 0;
          padding: 0;
          font-family: Arial, Helvetica, sans-serif;
          height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, #667eea, #764ba2);
          color: #fff;
        }}
        .card {{
          background: #ffffff;
          color: #333;
          padding: 24px 32px;
          border-radius: 12px;
          box-shadow: 0 10px 25px rgba(0,0,0,0.2);
          max-width: 480px;
          width: 100%;
          text-align: center;
        }}
        .status-badge {{
          display: inline-block;
          padding: 6px 12px;
          border-radius: 999px;
          background: {status_color};
          color: #fff;
          font-weight: bold;
          margin-top: 12px;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Malaria Detection System</h1>
        <p>Flask server running on port 5000.</p>
        <div class="status-badge">{status_text}</div>
        {error_html}
      </div>
    </body>
    </html>
    """


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "model_loaded": model is not None})


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    if model is None:
        load_model_once()

    if model is not None:
        print("Model is loaded and ready.")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    else:
        print(f"Model failed to load: {model_error}")

    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

# === Initialize Flask app ===
app = Flask(__name__, static_folder='build/static', template_folder='build')
CORS(app)

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_classifier (1).tflite")
H5_MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_classifier.h5")

# === Load TFLite model ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# === Load H5 model for Grad-CAM ===
h5_model = tf.keras.models.load_model(H5_MODEL_PATH)

# === Find last conv layer automatically ===
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")

# === Image Preprocessing ===
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_resized = image.resize((width, height))
    image_array = np.array(image_resized).astype(np.float32)
    image_array = (image_array / 127.5) - 1.0
    processed = np.expand_dims(image_array, axis=0)
    return image, processed

# === Grad-CAM Generator ===
def generate_gradcam(img_array, original_image):
    layer_name = get_last_conv_layer_name(h5_model)
    grad_model = tf.keras.models.Model(
        [h5_model.inputs],
        [h5_model.get_layer(layer_name).output, h5_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_array = np.array(original_image)
    overlayed = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
    return overlayed

# === Prediction API ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img_bytes = file.read()
        original_img, processed_img = preprocess_image(img_bytes)

        # Predict with TFLite
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction_score = float(output[0][0])
        threshold = 0.42
        result = 'Tumor' if prediction_score > threshold else 'No Tumor'

        # Grad-CAM with H5 model
        processed_for_h5 = np.array(original_img.resize((width, height)))
        processed_for_h5 = (processed_for_h5 / 127.5) - 1.0
        processed_for_h5 = np.expand_dims(processed_for_h5, axis=0).astype(np.float32)

        gradcam_image = generate_gradcam(processed_for_h5, original_img)
        _, buffer = cv2.imencode('.png', gradcam_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'prediction': result,
            'confidence': round(prediction_score * 100, 2),
            'explanation': encoded_image,
            'reason': f"The model focused on the highlighted areas to predict: '{result}'."
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Optional: List model layer names ===
@app.route('/layers', methods=['GET'])
def get_layers():
    layers = [layer.name for layer in h5_model.layers]
    return jsonify({'layers': layers})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    build_dir = os.path.join(BASE_DIR, 'build')
    file_path = os.path.join(build_dir, path)
    if path and os.path.exists(file_path):
        return send_from_directory(build_dir, path)
    else:
        return send_from_directory(build_dir, 'index.html')

# === Run app ===
if __name__ == '__main__':
    app.run(debug=True)

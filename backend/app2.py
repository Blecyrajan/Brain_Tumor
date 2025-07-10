from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

# === Initialize Flask app ===
app = Flask(__name__)
CORS(app)

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_classifier (1).tflite")
H5_MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_classifier.h5")

# === Load TFLite model for prediction ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# === Global H5 model loader for Grad-CAM ===
h5_model = None
def load_h5_model():
    global h5_model
    if h5_model is None:
        h5_model = tf.keras.models.load_model(H5_MODEL_PATH)
    return h5_model

# === Image Preprocessing for TFLite ===
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_resized = image.resize((width, height))
    image_array = np.array(image_resized).astype(np.float32)
    image_array = (image_array / 127.5) - 1.0  # match training normalization
    processed = np.expand_dims(image_array, axis=0)
    return image, processed

# === Grad-CAM Generator ===
def generate_gradcam(img_array, original_image, layer_name="Conv_1"):
    model = load_h5_model()
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
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

# === Prediction Route ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()

    # === Preprocess image ===
    original_img, processed_img = preprocess_image(img_bytes)

    # === Predict with TFLite model ===
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction_score = float(output[0][0])

    # === Determine result based on threshold ===
    threshold = 0.42  # ← Use threshold from training script
    result = 'Tumor' if prediction_score > threshold else 'No Tumor'

    # === Grad-CAM explanation using h5 model ===
    processed_for_h5 = (np.array(original_img.resize((width, height))) / 127.5) - 1.0
    processed_for_h5 = np.expand_dims(processed_for_h5, axis=0).astype(np.float32)
    gradcam_image = generate_gradcam(processed_for_h5, original_img)

    # === Encode Grad-CAM overlay ===
    _, buffer = cv2.imencode('.png', gradcam_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # === Explanation text ===
    reason = f"The model focused on the highlighted regions to predict: '{result}'."

    return jsonify({
        'prediction': result,
        'confidence': round(prediction_score * 100, 2),
        'explanation': encoded_image,
        'reason': reason
    })

# === Run server ===
if __name__ == '__main__':
    app.run(debug=True)

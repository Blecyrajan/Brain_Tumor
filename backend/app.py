from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

# Set up Flask app and allow CORS
app = Flask(__name__)
CORS(app)

# === Correct path handling ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_classifier.tflite")

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Automatically determine required input size from the model
_, height, width, _ = input_details[0]['shape']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((width, height))  # Resize to model input size (64x64)
    image_array = np.array(image) / 255.0
    return image, np.expand_dims(image_array, axis=0).astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    original_img, processed_img = preprocess_image(img_bytes)

    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = float(output[0][0])
    result = 'Tumor' if pred > 0.5 else 'No Tumor'
    print("Raw prediction value:", pred)

    # Create dummy heatmap overlay at correct size
    heatmap = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(heatmap, (width // 2, height // 2), int(min(width, height) * 0.4), 255, -1)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(np.array(original_img.resize((width, height))), 0.6, heatmap_color, 0.4, 0)

    _, buffer = cv2.imencode('.png', overlayed)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'prediction': result,
        'confidence': pred,
        'explanation': encoded_image
    })

if __name__ == '__main__':
    app.run(debug=True)

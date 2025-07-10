# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_classifier (1).tflite")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((width, height))
    image_array = np.array(image).astype(np.float32)
    image_array = (image_array / 127.5) - 1.0
    return image, np.expand_dims(image_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    original_img, processed_img = preprocess_image(file.read())

    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_score = float(output_data[0][0])
    threshold = 0.4
    result = 'Tumor' if prediction_score > threshold else 'No Tumor'

    # Heatmap placeholder (replace with Grad-CAM later)
    heatmap = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(heatmap, (width // 2, height // 2), int(min(width, height) * 0.4), 255, -1)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(np.array(original_img.resize((width, height))), 0.6, heatmap_color, 0.4, 0)

    _, buffer = cv2.imencode('.png', overlayed)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'prediction': result,
        'confidence': round(prediction_score, 4),
        'explanation': encoded_image
    })

if __name__ == '__main__':
    app.run(debug=True)

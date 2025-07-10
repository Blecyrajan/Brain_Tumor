import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# === CONFIG ===
DATASET_DIR = "/kaggle/input/brain-tumor-classification-dataset/dataset2"  # Folder with 'yes' and 'no' subfolders
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# === Load the dataset ===
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=None,  # Load all images as individual tensors
    shuffle=True,
    seed=SEED
)

# === Extract images and scalar labels ===
images = []
labels = []

for img, label in dataset:
    images.append(img.numpy())
    labels.append(label.numpy().item())  # ✅ scalar (float), not array

# === Convert to numpy arrays ===
images = np.array(images)
labels = np.array(labels)

# === Train-validation split ===
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# ✅ Ensure arrays for preprocessing and training
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# ✅ Check validation class balance
unique, counts = np.unique(y_val, return_counts=True)
print("Validation set class distribution:", dict(zip(unique, counts)))

# === Preprocessing: match app.py normalization ([-1, 1]) ===
X_train = (X_train / 127.5) - 1.0
X_val = (X_val / 127.5) - 1.0

# === Create tf.data.Dataset objects ===
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Compute class weights ===
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

# === Define the model (MobileNetV2) ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze feature extractor

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === Train the model ===
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights)

# === Predict on validation set ===
y_val_pred_probs = model.predict(X_val).ravel()

# === Find best threshold using F1-score ===
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_val, y_val_pred_probs > t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = np.max(f1_scores)
print(f"[INFO] Best threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")

# === Final predictions using best threshold ===
y_pred_final = (y_val_pred_probs > best_threshold).astype(int)

# === Evaluation metrics ===
accuracy = accuracy_score(y_val, y_pred_final)
precision = precision_score(y_val, y_pred_final)
recall = recall_score(y_val, y_pred_final)
conf_matrix = confusion_matrix(y_val, y_pred_final)

print("\n[MODEL EVALUATION ON VALIDATION SET]")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {best_f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# === Save model as H5 ===
model.save("brain_tumor_classifier.h5")

# === Convert to TFLite ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("brain_tumor_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("\n[INFO] Model saved and converted to brain_tumor_classifier.tflite")

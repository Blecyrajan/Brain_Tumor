import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from sklearn.utils import class_weight
import numpy as np
import os

# === DATASET LOADING ===
dataset_path = "./dataset2"
img_size = (224, 224)
batch_size = 40

# Raw dataset (no augmentation) for class weight calculation
raw_train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Extract labels
label_list = []
for images, labels in raw_train_dataset:
    label_list.extend(labels.numpy())

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(label_list),
    y=label_list
)
class_weights = dict(enumerate(class_weights))

# Enhanced data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
])

# Apply augmentation + rescaling
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(Rescaling(1./255)(x)), y)).prefetch(tf.data.AUTOTUNE)

val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = val_dataset.map(lambda x, y: (Rescaling(1./255)(x), y)).prefetch(tf.data.AUTOTUNE)

# === MODEL BUILDING ===
def create_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True  # Fine-tune entire model

    model = Sequential([
        Rescaling(1./255),  # Safety
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()

# === TRAINING ===
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# === SAVE MODEL (.h5) ===
model.save('brain_tumor_detection.h5')

# === CONVERT TO TFLITE ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Training complete and models saved: brain_tumor_detection.h5 & model.tflite")

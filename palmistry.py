# -*- coding: utf-8 -*-
"""Palmistry Recognition CNN + Gradio UI"""

# 1) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2) Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import math
import random
import gradio as gr
from PIL import Image

# 3) Config (update dataset path to palmistry dataset!)
data_dir = "/content/drive/MyDrive/cole_palmer"   # <-- change to your palmistry dataset
img_size = (128, 128)   # palmistry can work with grayscale small size
batch_size = 16
epochs = 20   # you may want to train longer for palmistry

# 4) Load dataset (grayscale)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

# 5) Class names
class_names = train_ds.class_names
print("Palmistry classes:", class_names)
idx_to_class = {i: name for i, name in enumerate(class_names)}

# 6) Preprocess
def preprocess(image, label):
    image = image / 255.0
    num_classes = len(class_names)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# 7) CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 8) Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 9) Plot accuracy/loss
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.legend(); plt.title('Loss')
plt.show()

# 10) Save palmistry model
save_path = "/content/drive/MyDrive/palmistry_model_gray.keras"
model.save(save_path)
print("Palmistry model saved to:", save_path)

# 11) Gradio inference
def predict_gradio(pil_img):
    if pil_img is None:
        return {}, None
    # convert to grayscale, resize, normalize
    img_gray = pil_img.convert("L").resize(img_size)
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))

    preds = model.predict(arr, verbose=0)[0]
    if preds.sum() != 0:
        preds = preds / preds.sum()
    out = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return out, img_gray

iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil", label="Upload palm image"),
    outputs=[gr.Label(num_top_classes=len(class_names), label="Prediction"),
             gr.Image(type="pil", label="Processed (grayscale)")],
    title="Palmistry Recognition (Grayscale 60x60)",
    description=f"Upload a palm image, model trained from: {data_dir}"
)

iface.launch(share=True)

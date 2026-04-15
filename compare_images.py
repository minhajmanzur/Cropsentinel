import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# =========================
# Load model
# =========================
model = tf.keras.models.load_model("CropSentinel_Model.h5")

# =========================
# Load class names
# =========================
with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# =========================
# Prediction function
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr, verbose=0)
    return class_names[np.argmax(preds)]

# =========================
# Image paths
# =========================
healthy_img      = "test_images/healthy.jpg"
diseased_img     = "test_images/pest.jpg"
non_flooded_img  = "test_images/non_flooded.jpg"
flooded_img      = "test_images/flooded.jpg"

# ====================================================
# TAB 1 → Healthy vs Diseased Plants
# ====================================================
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))

# Healthy Plant
ax1[0].imshow(image.load_img(healthy_img, target_size=(256, 256)))
ax1[0].set_title(
    f"Healthy Plant\nPrediction: {predict_image(healthy_img)}",
    fontsize=11
)
ax1[0].axis("off")

# Diseased Plant
ax1[1].imshow(image.load_img(diseased_img, target_size=(256, 256)))
ax1[1].set_title(
    f"Diseased Plant\nPrediction: {predict_image(diseased_img)}",
    fontsize=11
)
ax1[1].axis("off")

fig1.suptitle("CropSentinel – Plant Health Detection", fontsize=14)
plt.tight_layout()
plt.show()

# ====================================================
# TAB 2 → Flood Detection
# ====================================================
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))

# Non-Flooded Area
ax2[0].imshow(image.load_img(non_flooded_img, target_size=(256, 256)))
ax2[0].set_title(
    f"Non-Flooded Area\nPrediction: {predict_image(non_flooded_img)}",
    fontsize=11
)
ax2[0].axis("off")

# Flooded Area
ax2[1].imshow(image.load_img(flooded_img, target_size=(256, 256)))
ax2[1].set_title(
    f"Flooded Area\nPrediction: {predict_image(flooded_img)}",
    fontsize=11
)
ax2[1].axis("off")

fig2.suptitle("CropSentinel – Flood Detection", fontsize=14)
plt.tight_layout()
plt.show()

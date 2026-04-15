import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("CropSentinel_Model.h5")

# Validation directory
val_dir = r"C:\Programming\Final Year Project\CropSentinel Ai Based Flood and Crop detection system using remote sensing\CropSentinel Dataset\val"

# Generator (IMPORTANT: shuffle=False)
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load class names
with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# Predictions
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("CropSentinel Confusion Matrix")
plt.show()

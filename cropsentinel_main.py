# CropSentinel - Flood and Pest Detection using CNN (VGG16/ResNet50)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Dataset path
train_dir = r"C:\Programming\Final Year Project\CropSentinel Ai Based Flood and Crop detection system using remote sensing\CropSentinel Dataset\train"
val_dir   = r"C:\Programming\Final Year Project\CropSentinel Ai Based Flood and Crop detection system using remote sensing\CropSentinel Dataset\val"


# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model selection (choose one)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model
base_model.trainable = False

# Add classification layers
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save model
model.save("cropsentinel_vgg16_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()

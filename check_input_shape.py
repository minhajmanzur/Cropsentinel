from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("CropSentinel_Model.h5")

# Print basic input information
print("\n==============================")
print("MODEL INPUT SHAPE:", model.input_shape)
print("MODEL OUTPUT SHAPE:", model.output_shape)
print("==============================\n")

# Print full model architecture
model.summary()
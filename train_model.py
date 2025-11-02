import os
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.preprocessing import image_dataset_from_directory

# Set dataset paths
train_dir = os.path.join("dataset", "train")
val_dir = os.path.join("dataset", "va=-09iop")

# Load datasets
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary',
    shuffle=True
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

# Normalize and prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)

# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save the trained model
model.save("brain_tumor_model.keras")
print("✅ Model saved as brain_tumor_model.keras")

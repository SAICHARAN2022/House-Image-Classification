import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

dataset_path = "/home/houzeo/Documents/House-image-classification/Dataset/room-dataset"
test_path = "/home/houzeo/Documents/House-image-classification/test/room-dataset"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

for p in (dataset_path, test_path):
    if not os.path.exists(p):
        sys.exit(f"Path not found: {p}")

# Train / validation split from the training folder; test set is kept untouched
train_ds = keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_ds = keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_ds = keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
assert class_names == test_ds.class_names, (
    f"Class mismatch: train={class_names}, test={test_ds.class_names}"
)
num_classes = len(class_names)
print("Classes:", class_names)

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# CNN model (rescaling + augmentation live inside the model)
model = Sequential([
    keras.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1.0 / 255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
model.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=[early_stop])

# Final, unbiased evaluation on the held-out test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2%}")

model.save("intel_cnn.keras")

# Prediction on a single image: python house_image_classification.py path/to/image.jpg
img_path = sys.argv[1] if len(sys.argv) > 1 else None
if img_path:
    if not os.path.exists(img_path):
        sys.exit(f"Image not found: {img_path}")
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    x = np.expand_dims(keras.utils.img_to_array(img), axis=0)  # model rescales internally
    probs = model.predict(x)[0]
    print("prediction:", class_names[np.argmax(probs)], f"{probs.max():.2%}")

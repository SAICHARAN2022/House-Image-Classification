import os
import sys

import numpy as np
from tensorflow import keras

MODEL_PATH = "house_mobilenet.keras"
TRAIN_DIR = "/home/houzeo/Documents/House-image-classification/Dataset/room-dataset"
IMG_SIZE = (150, 150)

if len(sys.argv) < 2:
    sys.exit("Usage: python predict.py image1.jpg [image2.jpg ...]")
img_paths = sys.argv[1:]
if not os.path.exists(MODEL_PATH):
    sys.exit(f"{MODEL_PATH} not found - run house_image_classification.py first to train and save it")

# Same order image_dataset_from_directory uses (sorted folder names)
class_names = sorted(
    d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
)

model = keras.models.load_model(MODEL_PATH)

for img_path in img_paths:
    if not os.path.exists(img_path):
        print(f"\n{img_path}\n  Image not found, skipping")
        continue
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    x = np.expand_dims(keras.utils.img_to_array(img), axis=0)  # model rescales internally
    probs = model.predict(x, verbose=0)[0]

    print(f"\n{img_path}")
    print(f"Prediction: {class_names[np.argmax(probs)]} ({probs.max():.2%})")
    for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1]):
        print(f"  {name}: {p:.2%}")

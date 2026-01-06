import os
import json
import numpy as np
from flask import Flask, render_template, request
from keras.saving import load_model          # <-- correct loader
from tensorflow.keras.preprocessing import image

IMG_SIZE = (160, 160)

app = Flask(__name__)

# Global resources
model = None
class_names = []
calorie_map = {}


# ---------- Lazy Load Model (first request only) ----------
def load_resources():

    global model, class_names, calorie_map

    if model is not None:
        return

    print("🔹 Loading model (Keras 3 compatible)...")

    # Keras 3 model loader (required for your .h5 format)
    model = load_model(
        "food101_lightweight_mobilenet.h5",
        compile=False,
        safe_mode=True
    )

    # Load class labels
    with open("class_names.json") as f:
        class_names = json.load(f)

    # Load calorie mapping
    with open("calories.json") as f:
        calorie_map = json.load(f)

    print("✅ Model Ready")


# ---------- Prediction ----------
def predict_food(img_path):

    load_resources()   # loads only once

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)

    index = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    food_name = class_names[index]

    calories = calorie_map.get(food_name, 250)

    return food_name, round(calories), round(confidence, 2)


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def upload_image():

    file = request.files["image"]

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    upload_path = os.path.join(upload_dir, "food.jpg")
    file.save(upload_path)

    food, calories, confidence = predict_food(upload_path)

    return render_template(
        "index.html",
        uploaded=True,
        img_path="/" + upload_path,
        food=food,
        calories=calories,
        confidence=confidence
    )


# ---------- Local Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

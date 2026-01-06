from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing import image

# ---------- Load Model ----------
model = tf.keras.models.load_model("food101_lightweight_mobilenet.h5")

# ---------- Load Class Names ----------
with open("class_names.json") as f:
    class_names = json.load(f)

# ---------- Load Calorie Map ----------
with open("calories.json") as f:
    calorie_map = json.load(f)

IMG_SIZE = (160, 160)

app = Flask(__name__)


# ---------- Prediction Function ----------
def predict_food(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)

    index = np.argmax(preds)
    confidence = float(np.max(preds) * 100)

    # food label
    food_name = class_names[index]

    # get calories (fallback = 250 kcal)
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

    # save uploaded image
    file = request.files["image"]

    upload_path = os.path.join("static", "uploaded.jpg")
    file.save(upload_path)

    food, calories, confidence = predict_food(upload_path)

    return render_template(
        "index.html",
        food=food,
        calories=calories,
        confidence=confidence,
        uploaded=True,
        img_path=upload_path
    )



# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)

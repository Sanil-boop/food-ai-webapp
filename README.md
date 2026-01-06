🍽️ Food AI — Food Recognition & Calorie Estimator

This project uses a deep-learning model to recognize food items from images and estimate their calorie content.
The system is built using MobileNetV2, optimized for fast & lightweight inference, and deployed as a Flask web app.

Users can upload a food image and instantly receive:

✅ Detected Food Name

🔥 Estimated Calories

🎯 Model Confidence Score

🖼️ Image Preview


🧾 Features

✔ Upload food image
✔ Predict food item
✔ Estimate calories
✔ Show confidence score
✔ Mobile-friendly UI
✔ Fast CPU inference
✔ Works on Render & local machine


🏗️ Tech Stack
| Component  | Technology                 |
| ---------- | -------------------------- |
| Model      | MobileNetV2                |
| Framework  | TensorFlow / Keras         |
| Backend    | Flask                      |
| Frontend   | HTML + CSS                 |
| Deployment | Render                     |
| Format     | `.keras` compatible export |

📂 Project Structure
food-ai-webapp/
│
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
│
├── food101_mobilenet_compat.keras  # Model file
├── class_names.json                # Labels
├── calories.json                   # Calorie mapping
│
├── templates/
│   ├── index.html                  # Main UI
│   └── about.html                  # About page
│
├── static/
│   ├── style.css                   # Styling
│   └── uploads/                    # Uploaded images

🖥️ Run Locally (VS Code / Jupyter)
1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
python app.py

Open in browser:
http://127.0.0.1:5000


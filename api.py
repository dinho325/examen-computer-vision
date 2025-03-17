import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Charger le modèle entraîné
model = tf.keras.models.load_model("model_fruits360.h5")  

# Fonction de prétraitement de l'image
def preprocess_image(image):
    image = image.resize((100, 100))  
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)  
    return image

# Route d’inférence (POST)
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["file"]
    image = Image.open(file.stream)  
    processed_image = preprocess_image(image) 

    # Faire la prédiction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  
    confidence = float(np.max(predictions)) 

# Page d'accueil
@app.route("/", methods=["GET"])
def home():
    return "🎯 API de classification des fruits est en ligne !", 200

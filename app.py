from flask import Flask, request, jsonify
import pickle
import xgboost
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Charger les modèles
logistic_model = pickle.load(open("models/xgboost_model.pkl", "rb"))
deit_model = torch.load("models/deit_model.pth", map_location=torch.device('cpu'))

data = pd.read_csv("data/donnees_augmentees.csv")  # Charger le fichier CSV contenant les symptômes, pays, environnements

# Création du dossier d'uploads si non existant
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Transformation pour les images
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    symptoms = [col for col in data.columns if "symptome" in col]
    return jsonify({"symptoms": symptoms})

@app.route("/filter", methods=["POST"])
def filter_options():
    user_symptoms = request.json.get("symptoms", [])
    if not user_symptoms:
        return jsonify({"error": "Aucun symptôme sélectionné"}), 400
    
    # Filtrage des données
    filtered_data = data.copy()
    for symptom in user_symptoms:
        filtered_data = filtered_data[filtered_data[symptom] == symptom]
    
    countries = filtered_data["pays"].unique().tolist()
    environments = filtered_data["environement01"].unique().tolist()
    
    return jsonify({"countries": countries, "environments": environments})

@app.route("/predict", methods=["POST"])
def predict():
    user_symptoms = request.json.get("symptoms", [])
    country = request.json.get("country", "")
    environment = request.json.get("environment", "")
    
    if not user_symptoms:
        return jsonify({"error": "Veuillez sélectionner au moins un symptôme"}), 400
    
    # Préparation des features pour la prédiction
    features = [1 if sym in user_symptoms else 0 for sym in data.columns if "symptome" in sym]
    prediction = logistic_model.predict([features])[0]
    
    return jsonify({"predicted_disease": prediction})

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    image_tensor = transform_image(filepath)
    with torch.no_grad():
        prediction = deit_model(image_tensor)
    
    predicted_label = torch.argmax(prediction, dim=1).item()
    return jsonify({"image_prediction": predicted_label})

@app.route("/final-diagnosis", methods=["POST"])
def final_diagnosis():
    user_symptoms = request.json.get("symptoms", [])
    country = request.json.get("country", "")
    environment = request.json.get("environment", "")
    file = request.files.get("file")
    
    if not user_symptoms or not file:
        return jsonify({"error": "Symptômes et image requis"}), 400
    
    # Prédiction par symptômes
    features = [1 if sym in user_symptoms else 0 for sym in data.columns if "symptome" in sym]
    predicted_disease_text = logistic_model.predict([features])[0]
    
    # Prédiction par image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    image_tensor = transform_image(filepath)
    with torch.no_grad():
        prediction = deit_model(image_tensor)
    predicted_label_image = torch.argmax(prediction, dim=1).item()
    
    return jsonify({
        "predicted_disease_text": predicted_disease_text,
        "predicted_disease_image": predicted_label_image,
        "final_diagnosis": "Les résultats doivent être comparés par un professionnel de santé."
    })

if __name__ == "__main__":
    app.run(debug=True)

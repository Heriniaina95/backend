from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pickle
import torch
from torchvision import transforms
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Charger les modèles
logistic_model = pickle.load(open("models/xgboost_model.pkl", "rb"))
deit_model = torch.load("models/deit_model.pth", map_location=torch.device('cpu'))
label_encoders = joblib.load('./models/label_encoders.joblib')

data = pd.read_csv("data/donnees_augmentees.csv")
unique_countries = data['pays'].unique().tolist()

# Création du dossier d'uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Catégorisation des pays par continent
continents = {
    "Afrique": ["Afrique", "Afrique centrale", "Afrique du Sud", "Afrique subsaharienne", "Angola", "Bénin", "Burkina Faso", "Burundi", "Cameroun", "Cap-Vert", "Cap Vert.", "Centrafrique", "Comores", "Congo", "Côte d’Ivoire", "Côte d'Ivoire", "Djibouti", "Egypte", "Érythrée", "Éthiopie", "Gabon", "Gambie", "Ghana", "Guinée", "Guinée Bissau", "Guinée Bissau", "Guinée équatoriale", "Guinée-Bissau", "Kenya", "Liberia", "Libéria", "Madagascar", "Mali", "Mauritanie", "Mozambique", "Namibie", "Niger", "Nigeria", "Nigéria", "Ouganda", "République centrafricaine", "République Démocratique du Congo", "République démocratique du Congo", "Rwanda", "Sao Tome et Principe", "Sénégal", "Seychelles", "Sierra Leone", "Somalie", "Soudan", "Soudan du Sud", "Tanzanie", "Tchad", "Togo", "Togo.", "Zambie", "Zimbabwe"],
    "Asie": ["Afghanistan", "Arabie Saoudite", "Bangladesh", "Bhoutan", "Brunei", "Cambodge", "Chine", "Corse (France)", "Hong-Kong", "Inde", "Indonésie", "Iran frontière Pakistan et Afghanistan", "Japon", "Kiribati", "Laos", "Macao", "Malaisie", "Maldives", "Myanmar (Birmanie)", "Népal", "Oman", "Pakistan", "Philippines", "Singapour", "Sri Lanka", "Taïwan", "Thaïlande", "Timor Leste", "Vietnam", "Yemen"],
    "Europe": ["Allemagne", "Pays-Bas", "Yougoslavie"],
    "Amérique du Nord": ["États-Unis d’Amérique", "Mexique"],
    "Amérique Centrale": ["Amérique Centrale", "Aruba", "Barbade", "Belize", "Costa Rica", "Curaçao", "Dominique", "Guadeloupe", "Guatemala", "Haiti", "Honduras", "Martinique", "Nicaragua", "Panama", "République Dominicaine", "Salvador", "Saint Kitts et Nevis", "Saint Vincent et la Grenadine", "Trinité-et-Tobago (Trinité seulement)"],
    "Amérique du Sud": ["Amérique du Sud", "Argentine", "Bolivie", "Brésil", "Brésil, dans l’État plurinational de Bolivie", "Chili", "Colombie", "Équateur", "Guyana", "Guyane", "Guyane Française", "Guyane française", "Paraguay", "Pérou", "Suriname", "Uruguay", "Venezuela"],
    "Océanie": ["Amériques", "Aruba", "Fidji", "Iles Salomon", "Iles Vierges Britanniques", "Iles Vierges Americaine", "Micronésie", "Nouvelle-Calédonie", "Palau", "Papouasie", "Polynésie Française", "Réunion", "Samoa", "Samoa Americain", "Tonga", "Vanuatu", "Wallis et Futuna"],
    "Moyen-Orient": ["Moyen-Orient", "Yémen"],
    "Autres": ["inconnu", "le bassin méditerranéen", "Amérique latine", "Amériques", "Asie centrale", "Pacifique Sud", "Région de l'Océanie", "Amérique, Moyen-Orient, Caraïbes, Brésil, Venezuela et Suriname", "Chine, Indonésie, Philippines", "Cambodge et de la République démocratique populaire lao", "Afrique, Moyen-Orient, Caraïbes, Brésil, Venezuela et Suriname"]
}

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
    symptoms_data = {}
    for i in range(1, 15):
        col = f"symptome{str(i).zfill(2)}"
        if col in data.columns:
            symptoms_data[col] = data[col].unique().tolist()
    return jsonify({"symptoms": symptoms_data})

@app.route("/countries", methods=["GET"])
def get_countries():
    return jsonify({"countries": unique_countries})

@app.route("/continents", methods=["GET"])
def get_continents():
    return jsonify({"continents": continents})

@app.route("/countries-by-continent", methods=["POST"])
def get_countries_by_continent():
    data = request.get_json()
    continent = data.get("continent", "")
    if continent in continents:
        return jsonify({"countries": continents[continent]})
    else:
        return jsonify({"error": "Continent non valide"}), 400

@app.route("/predict-environment-survey", methods=["POST"])
def predict_environment_survey():
    data = request.get_json()

    # Charger le fichier CSV
    df = pd.read_csv("./data/tableau_enquete_environnements.csv")

    # Filtrer le DataFrame pour trouver la ligne correspondante
    filtered_df = df[
        (df["Type de Logement"] == data.get("Type de Logement")) &
        (df["Proximité de l'Eau"] == data.get("Proximité de l'Eau")) &
        (df["Environnement Naturel"] == data.get("Environnement Naturel")) &
        (df["Conditions Sanitaires"] == data.get("Conditions Sanitaires")) &
        (df["Présence d'Animaux"] == data.get("Présence d'Animaux"))
    ]

    if not filtered_df.empty:
        predicted_environment = filtered_df["Environnement"].values[0]
        return jsonify({'predicted_environment': predicted_environment})
    else:
        return jsonify({'error': 'Aucun environnement trouvé pour ces réponses'}), 400

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    # Récupérer les symptômes depuis l'endpoint /symptoms
    symptoms_data = get_symptoms().json.get("symptoms", {})
    all_symptoms = [item for sublist in symptoms_data.values() for item in sublist]  # Aplatir la liste

    # Obtenir les données envoyées par le client
    received_data = request.json
    symptoms = received_data.get("symptoms", [])
    country = received_data.get("country", "")
    environment_data = received_data.get("environment_data", {})

    # Vérifications
    if not symptoms or not country or not environment_data:
        return jsonify({"error": "Veuillez fournir les symptômes, le pays et l'environnement"}), 400

    if country not in unique_countries:
        return jsonify({"error": "Pays sélectionné non valide"}), 400

    for symptom in symptoms:
        if symptom not in all_symptoms:
            return jsonify({"error": f"Symptôme '{symptom}' non valide"}), 400

    # Récupérer l'environnement prédit à partir de l'endpoint /predict-environment-survey
    predicted_environment = predict_environment_survey().json.get("predicted_environment", "")

    # Construction des features
    features = symptoms + [country, predicted_environment]
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

@app.route("/predict-combined", methods=["POST"])
def predict_combined():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image_tensor = transform_image(filepath)
    with torch.no_grad():
        deit_prediction = deit_model(image_tensor)
    predicted_label = torch.argmax(deit_prediction, dim=1).item()

    symptoms_data = get_symptoms().json.get("symptoms", {})
    unique_countries = get_countries().json.get("countries", [])
    environment_response = predict_environment_survey()

    symptoms = request.json.get("symptoms", [])
    country = request.json.get("country", "")

    if not symptoms or not country or not environment_response:
        return jsonify({"error": "Veuillez fournir les symptômes, le pays et l'environnement"}), 400

    if country not in unique_countries:
        return jsonify({"error": "Pays sélectionné non valide"}), 400

    valid_symptoms = []
    for symptom in symptoms:
        if any(symptom in values for values in symptoms_data.values()):
            valid_symptoms.append(symptom)

    if not valid_symptoms:
        return jsonify({"error": "Aucun symptôme valide fourni"}), 400

    predicted_environment = environment_response.json.get("predicted_environment", "")
    features = valid_symptoms + [country, predicted_environment]
    disease_prediction = logistic_model.predict([features])[0]

    return jsonify({"predicted_disease": disease_prediction, "image_prediction": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)

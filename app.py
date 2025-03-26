from flask import Flask, request, jsonify
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Permet les requêtes CORS

# Charger le modèle correctement
try:
    model = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=15)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))
    model.eval()  # Mode évaluation
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    exit(1)

# Classes (Modifier selon ton dataset)
classes = [
    "data_final_brucellose", "data_final_chikungunya", "data_final_dengue",
    "data_final_fievrehemoragique", "data_final_fievrejaune", "data_final_filariose",
    "data_final_leishmaniose", "data_final_onchocercose", "data_final_pest",
    "data_final_rouge", "data_final_schistosomiase", "data_final_trypanosomiase",
    "data_final_virus", "data_final_zik", "paludisme"
]

# Dictionnaire pour les explications des maladies (Information mise à jour)
disease_explanation = {
    "data_final_brucellose": {
        "name": "Brucellose",
        "description": "La brucellose est une infection bactérienne transmise par les animaux.",
        "symptoms": "Fièvre, douleurs musculaires, sueurs, fatigue, perte d’appétit.",
        "countries": "Présente en Afrique, Asie, Amérique du Sud.",
        "favorableEnvironment": "Zones rurales avec contact avec le bétail et les animaux d'élevage."
    },
    "data_final_chikungunya": {
        "name": "Chikungunya",
        "description": "Le chikungunya est une maladie virale transmise par les moustiques.",
        "symptoms": "Fièvre élevée, douleurs articulaires intenses, éruptions cutanées, maux de tête.",
        "countries": "Afrique, Asie, Caraïbes.",
        "favorableEnvironment": "Zones tropicales humides."
    },
    "data_final_dengue": {
        "name": "Dengue",
        "description": "La dengue est une infection virale transmise par les moustiques.",
        "symptoms": "Fièvre élevée, douleurs musculaires, éruptions cutanées, fatigue.",
        "countries": "Asie, Afrique, Amérique Latine.",
        "favorableEnvironment": "Zones tropicales et subtropicales."
    },
    "data_final_fievrehemoragique": {
        "name": "Fièvre hémorragique",
        "description": "Maladie virale pouvant entraîner des saignements graves et des défaillances organiques.",
        "symptoms": "Fièvre, fatigue, douleurs musculaires, saignements, choc.",
        "countries": "Afrique, Asie.",
        "favorableEnvironment": "Zones tropicales avec transmission animale ou par moustiques."
    },
    "data_final_fievrejaune": {
        "name": "Fièvre jaune",
        "description": "Infection virale transmise par les moustiques.",
        "symptoms": "Fièvre, frissons, douleurs musculaires, nausées, jaunisse.",
        "countries": "Afrique subsaharienne, Amérique latine.",
        "favorableEnvironment": "Zones tropicales avec moustiques."
    },
    "data_final_filariose": {
        "name": "Filariose",
        "description": "Infection parasitaire causée par des vers filaires.",
        "symptoms": "Inflammation, gonflement des membres (éléphantiasis), fièvre.",
        "countries": "Régions tropicales et subtropicales.",
        "favorableEnvironment": "Zones avec moustiques porteurs de filaires."
    },
    "data_final_leishmaniose": {
        "name": "Leishmaniose",
        "description": "Maladie parasitaire transmise par les phlébotomes.",
        "symptoms": "Ulcères cutanés, fièvre, perte de poids, atteinte des organes internes.",
        "countries": "Afrique, Asie, Amérique latine.",
        "favorableEnvironment": "Régions désertiques et semi-désertiques."
    },
    "data_final_onchocercose": {
        "name": "Onchocercose",
        "description": "Maladie parasitaire causée par des vers.",
        "symptoms": "Démangeaisons sévères, lésions cutanées, cécité progressive.",
        "countries": "Afrique subsaharienne.",
        "favorableEnvironment": "Zones riveraines avec mouches noires."
    },
    "data_final_pest": {
        "name": "Peste",
        "description": "Maladie bactérienne transmise par les puces.",
        "symptoms": "Fièvre soudaine, ganglions enflés, frissons, fatigue.",
        "countries": "Asie, Afrique, Amérique.",
        "favorableEnvironment": "Zones où vivent des rongeurs infectés."
    },
    "data_final_rouge": {
        "name": "Rougeole",
        "description": "Infection virale avec éruption cutanée et symptômes respiratoires.",
        "symptoms": "Fièvre, éruptions cutanées, toux, conjonctivite.",
        "countries": "Mondialement, plus fréquente dans les zones à faible couverture vaccinale.",
        "favorableEnvironment": "Zones avec faible couverture vaccinale."
    },
    "data_final_schistosomiase": {
        "name": "Schistosomiase",
        "description": "Maladie parasitaire liée à l'eau douce.",
        "symptoms": "Fièvre, douleurs abdominales, sang dans les urines ou selles.",
        "countries": "Afrique, Asie, Amérique latine.",
        "favorableEnvironment": "Zones avec eaux stagnantes."
    },
    "data_final_trypanosomiase": {
        "name": "Trypanosomiase",
        "description": "Maladie parasitaire transmise par la mouche tsé-tsé.",
        "symptoms": "Fièvre, maux de tête, troubles du sommeil, confusion.",
        "countries": "Afrique subsaharienne.",
        "favorableEnvironment": "Zones rurales et forestières."
    },
    "data_final_virus": {
        "name": "Virus",
        "description": "Maladies virales diverses, souvent transmises par insectes ou animaux.",
        "symptoms": "Varie selon le virus (fièvre, fatigue, douleurs musculaires...).",
        "countries": "Mondialement.",
        "favorableEnvironment": "Environnements où le vecteur du virus est présent."
    },
    "data_final_zik": {
        "name": "Zika",
        "description": "Maladie virale transmise par les moustiques Aedes.",
        "symptoms": "Fièvre modérée, éruptions cutanées, douleurs articulaires, conjonctivite.",
        "countries": "Amérique latine, Asie, Afrique.",
        "favorableEnvironment": "Zones humides où les moustiques Aedes sont présents."
    },
    "paludisme": {
        "name": "Paludisme",
        "description": "Maladie infectieuse causée par le parasite Plasmodium.",
        "symptoms": "Fièvre récurrente, frissons, sueurs, maux de tête, anémie.",
        "countries": "Afrique subsaharienne, Asie, Amérique latine.",
        "favorableEnvironment": "Zones tropicales avec moustiques Anopheles."
    }
}

# Création du dossier d'uploads
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

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Charger l'image et faire la prédiction
    image_tensor = transform_image(filepath)
    with torch.no_grad():
        prediction = model(image_tensor)

    # Obtenir l'indice et le nom de la classe
    predicted_index = torch.argmax(prediction, dim=1).item()
    predicted_label = classes[predicted_index]  # Récupérer le nom de la classe

    # Obtenir l'explication de la maladie
    disease_info = disease_explanation.get(predicted_label, {
        "name": "Inconnu",
        "description": "Aucune information disponible.",
        "symptoms": "Non spécifié",  # Ajoutez les symptômes ici
        "countries": "Non spécifié",
        "favorableEnvironment": "Non spécifié"
    })

    return jsonify({
        "image_prediction": predicted_label,
        "disease_info": disease_info
    })

# Charger les données CSV
data = pd.read_csv("data/donnees_symptomepays.csv")
unique_countries = data['pays'].unique().tolist()

# Catégorisation des pays par continent
continents = {
    "Afrique": ["Afrique", "Afrique centrale", "Afrique du Sud", "Angola", "Bénin", "Burkina Faso", "Burundi", "Cameroun", "Cap-Vert", "Centrafrique", "Comores", "Congo", "Côte d’Ivoire", "Djibouti", "Egypte", "Érythrée", "Éthiopie", "Gabon", "Gambie", "Ghana", "Guinée", "Guinée-Bissau", "Kenya", "Liberia", "Libéria", "Madagascar", "Mali", "Mauritanie", "Mozambique", "Namibie", "Niger", "Nigeria", "Ouganda", "République centrafricaine", "République Démocratique du Congo", "Rwanda", "Sao Tomé et Principe", "Sénégal", "Seychelles", "Sierra Leone", "Somalie", "Soudan", "Tanzanie", "Tchad", "Togo", "Zambie", "Zimbabwe"],
    "Asie": ["Afghanistan", "Arabie Saoudite", "Bangladesh", "Bhoutan", "Brunei", "Cambodge", "Chine", "Hong-Kong", "Inde", "Indonésie", "Iran", "Japon", "Laos", "Malaisie", "Maldives", "Myanmar", "Népal", "Oman", "Pakistan", "Philippines", "Singapour", "Sri Lanka", "Taïwan", "Thaïlande", "Vietnam", "Yemen"],
    "Europe": ["Allemagne", "France", "Italie", "Pays-Bas", "Espagne", "Royaume-Uni", "Yougoslavie"],
    "Amérique du Nord": ["États-Unis", "Canada", "Mexique"],
    "Amérique Centrale": ["Belize", "Costa Rica", "Cuba", "Guadeloupe", "Guatemala", "Honduras", "Nicaragua", "Panama", "République Dominicaine", "Salvador"],
    "Amérique du Sud": ["Argentine", "Bolivie", "Brésil", "Chili", "Colombie", "Équateur", "Guyana", "Paraguay", "Pérou", "Suriname", "Uruguay", "Venezuela"],
    "Océanie": ["Australie", "Fidji", "Nouvelle-Zélande", "Papouasie", "Samoa", "Tonga", "Vanuatu"],
    "Moyen-Orient": ["Arabie Saoudite", "Émirats Arabes Unis", "Iran", "Irak", "Israël", "Jordanie", "Liban", "Oman", "Qatar", "Syrie", "Turquie", "Yémen"],
    "Autres": ["inconnu", "le bassin méditerranéen", "Asie centrale", "Pacifique Sud"]
}

# Préparation des données pour la similarité cosinus
def prepare_data_for_similarity(data):
    # Supposons que les colonnes de symptômes sont nommées 'symptome01', 'symptome02', etc.
    symptom_columns = [f"symptome{str(i).zfill(2)}" for i in range(1, 15)]
    X = data[symptom_columns].fillna(0).values

    # Encodage des étiquettes de maladies
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data['maladie'])

    return X, y_encoded, label_encoder

# Fonction pour prédire la maladie basée sur la similarité des symptômes
def predire_maladie(symptomes_patient, X, y_encoded, label_encoder):
    # Vérifier que symptomes_patient a le même nombre de caractéristiques que X
    if len(symptomes_patient) != X.shape[1]:
        raise ValueError("Le nombre de symptômes du patient doit être égal au nombre de caractéristiques dans X.")

    # Calculer la similarité cosinus entre les symptômes du patient et les symptômes des maladies connues
    similarites = cosine_similarity([symptomes_patient], X)

    # Trouver l'indice de la maladie avec la similarité la plus élevée
    indice_maladie = similarites.argmax()

    # Prédire la maladie
    maladie_predite = label_encoder.inverse_transform([y_encoded[indice_maladie]])

    return maladie_predite[0]

# Préparer les données pour la similarité
X, y_encoded, label_encoder = prepare_data_for_similarity(data)

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    """ Predict a disease based on the provided symptoms and country """
    received_data = request.json
    symptoms = received_data.get("symptoms", [])
    country = received_data.get("country", "")

    print("Received symptoms:", symptoms)  # Add this log
    print("Received country:", country)  # Add this log

    if not symptoms or not country:
        return jsonify({"error": "Please provide both symptoms and country"}), 400

    if country not in unique_countries:
        return jsonify({"error": f"The country '{country}' is not valid"}), 400

    valid_symptoms = get_symptoms_list()
    for symptom in symptoms:
        if symptom not in valid_symptoms:
            return jsonify({"error": f"The symptom '{symptom}' is not valid"}), 400

    # Convert symptoms to binary vector
    symptom_vector = [1 if symptom in symptoms else 0 for symptom in valid_symptoms]

    # Predict the disease
    predicted_disease = predire_maladie(symptom_vector, X, y_encoded, label_encoder)

    return jsonify({"predicted_disease": predicted_disease})

@app.route("/countries", methods=["GET"])
def get_countries():
    """ Renvoie la liste des pays présents dans le dataset """
    continent = request.args.get('continent')
    if continent and continent in continents:
        return jsonify({"countries": continents[continent]})
    return jsonify({"countries": unique_countries})

@app.route("/continents", methods=["GET"])
def get_continents():
    """ Renvoie la liste des continents et leurs pays associés """
    return jsonify({"continents": continents})

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """ Renvoie la liste des symptômes disponibles dans le dataset par colonne """
    return jsonify({"symptoms": get_symptoms_list()})

def get_symptoms_list():
    """ Récupère la liste des symptômes par colonne dans le dataset """
    symptoms_data = {}
    for i in range(1, 15):
        col = f"symptome{str(i).zfill(2)}"
        if col in data.columns:
            symptoms_data[col] = data[col].dropna().unique().tolist()
    return symptoms_data

if __name__ == '__main__':
    app.run(debug=True)

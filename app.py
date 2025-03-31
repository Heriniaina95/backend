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

# Classes (Modifier selon votre dataset)
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
        "symptoms": "Non spécifié",
        "countries": "Non spécifié",
        "favorableEnvironment": "Non spécifié"
    })

    return jsonify({
        "image_prediction": predicted_label,
        "disease_info": disease_info
    })

# Charger les données CSV
data = pd.read_csv("data/donnees_symptomes.csv")

# Préparation des données pour la similarité cosinus
def prepare_data_for_similarity(data):
    # Supposons que les colonnes de symptômes sont nommées 'symptome01', 'symptome02', etc.
    symptom_columns = [f"symptome{str(i).zfill(2)}" for i in range(1, 15)]

    # Vérifier si les colonnes existent
    symptom_columns = [col for col in symptom_columns if col in data.columns]

    # Remplacer 'inconnu' par 0 et les autres valeurs par 1
    X = data[symptom_columns].applymap(lambda x: 0 if x == 'inconnu' else 1).values

    # Encodage des étiquettes de maladies
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data['maladie'])

    return X, y_encoded, label_encoder, symptom_columns

# Fonction pour prédire la maladie basée sur la similarité des symptômes
def predire_maladie(symptomes_patient, X, y_encoded, label_encoder, symptom_columns):
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
X, y_encoded, label_encoder, symptom_columns = prepare_data_for_similarity(data)

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    """ Prédit une maladie en fonction des symptômes fournis """
    received_data = request.json
    symptoms = received_data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "Veuillez fournir les symptômes"}), 400

    valid_symptoms = get_symptoms_list()

    for symptom in symptoms:
        if symptom not in valid_symptoms:
            return jsonify({"error": f"Le symptôme '{symptom}' n'est pas valide"}), 400

    # Convertir les symptômes en vecteur binaire
    symptom_vector = [1 if symptom in symptoms else 0 for symptom in symptom_columns]

    # Prédire la maladie
    predicted_disease = predire_maladie(symptom_vector, X, y_encoded, label_encoder, symptom_columns)

    return jsonify({"predicted_disease": predicted_disease})

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """ Renvoie la liste des symptômes disponibles dans le dataset """
    return jsonify({"symptoms": get_symptoms_list()})

def get_symptoms_list():
    """ Récupère la liste des symptômes dans le dataset sous forme de liste unique """
    symptoms_set = set()
    for i in range(1, 15):
        col = f"symptome{str(i).zfill(2)}"
        if col in data.columns:
            symptoms_set.update(data[col].dropna().unique().tolist())

    return list(symptoms_set)

# Données simulées (à remplacer par une base de données réelle)
mesures = {
    "Peste bubonique": [
        "Consultation médicale immédiate : En cas de contact avec un patient pesteux qui tousse, consulter un médecin pour obtenir des antibiotiques en mesure préventive.",
        "Isolement : Les patients atteints de peste pulmonaire doivent être isolés pour éviter la propagation de la maladie.",
        "Traitement antibiotique : Administrer des antibiotiques appropriés dès que possible.",
        "Précautions standard : Porter des équipements de protection individuelle (EPI).",
        "Surveillance des contacts : Identifier et suivre les proches contacts.",
        "Hygiène et assainissement : Appliquer des mesures de lutte contre les vecteurs."
    ],
    "Peste pulmonaire": [
        "Consultation médicale immédiate : En cas de contact avec un patient pesteux qui tousse, consulter un médecin pour obtenir des antibiotiques en mesure préventive.",
        "Isolement : Les patients atteints de peste pulmonaire doivent être isolés pour éviter la propagation de la maladie.",
        "Traitement antibiotique : Administrer des antibiotiques appropriés dès que possible.",
        "Précautions standard : Porter des équipements de protection individuelle (EPI).",
        "Surveillance des contacts : Identifier et suivre les proches contacts.",
        "Hygiène et assainissement : Appliquer des mesures de lutte contre les vecteurs."
    ],
    "Peste septicémique": [
        "Consultation médicale immédiate : En cas de contact avec un patient pesteux qui tousse, consulter un médecin pour obtenir des antibiotiques en mesure préventive.",
        "Isolement : Les patients atteints de peste pulmonaire doivent être isolés pour éviter la propagation de la maladie.",
        "Traitement antibiotique : Administrer des antibiotiques appropriés dès que possible.",
        "Précautions standard : Porter des équipements de protection individuelle (EPI).",
        "Surveillance des contacts : Identifier et suivre les proches contacts.",
        "Hygiène et assainissement : Appliquer des mesures de lutte contre les vecteurs."
    ],
    "Schistosomiase urogénitale": [
        "Traitement médicamenteux : Le praziquantel est le traitement de choix.",
        "Suivi médical : Examen de suivi recommandé 1 à 2 mois après le traitement.",
        "Prévention : Éviter le contact avec des eaux douces contaminées.",
        "Éducation sanitaire : Sensibiliser à l'importance du dépistage précoce."
    ],
    "Trypanosoma brucei gambiense": [
        "Traitement médicamenteux : La pentamidine est utilisée au premier stade de la maladie. Pour le deuxième stade, des médicaments comme l'éflornithine ou le nifurtimox-éflornithine sont utilisés.",
        "Suivi médical : Un suivi régulier est nécessaire pour surveiller l'évolution de la maladie et ajuster le traitement si nécessaire.",
        "Prévention : Éviter les piqûres de mouches tsé-tsé en portant des vêtements protecteurs et en utilisant des répulsifs. Éliminer les habitats de reproduction des mouches tsé-tsé.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ],
    "Trypanosoma brucei rhodesiense": [
        "Traitement médicamenteux : Le suramin est utilisé au premier stade de la maladie, tandis que la mélarsoprol est utilisée au deuxième stade.",
        "Suivi médical : Un suivi régulier est nécessaire pour surveiller l'évolution de la maladie et ajuster le traitement si nécessaire.",
        "Prévention : Éviter les piqûres de mouches tsé-tsé en portant des vêtements protecteurs et en utilisant des répulsifs. Éliminer les habitats de reproduction des mouches tsé-tsé.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ],
    "Rougeole": [
        "Vaccination : La vaccination est la mesure la plus efficace pour prévenir la rougeole. Deux doses du vaccin ROR (Rougeole-Oreillons-Rubéole) sont recommandées.",
        "Isolement : Les personnes atteintes de rougeole doivent être isolées pendant la période de contagiosité pour éviter la propagation de la maladie.",
        "Hygiène : Se laver fréquemment les mains et éviter de partager des objets personnels avec des personnes infectées.",
        "Vitamine A : Administrer des suppléments de vitamine A aux enfants atteints de rougeole pour réduire le risque de complications."
    ],
    "Zika": [
        "Prévention des piqûres de moustiques : Utiliser des répulsifs, porter des vêtements couvrants et dormir sous des moustiquaires imprégnées d'insecticide.",
        "Élimination des gîtes larvaires : Éliminer les eaux stagnantes autour des habitations pour réduire la reproduction des moustiques.",
        "Protection sexuelle : Utiliser des préservatifs pour prévenir la transmission sexuelle du virus Zika.",
        "Surveillance médicale : Les femmes enceintes doivent être surveillées de près en raison du risque de microcéphalie chez le fœtus."
    ],
    "Chikungunya": [
        "Prévention des piqûres de moustiques : Utiliser des répulsifs, porter des vêtements couvrants et dormir sous des moustiquaires imprégnées d'insecticide.",
        "Élimination des gîtes larvaires : Éliminer les eaux stagnantes autour des habitations pour réduire la reproduction des moustiques.",
        "Traitement symptomatique : Repos, hydratation et utilisation d'antalgiques pour soulager les symptômes.",
        "Surveillance médicale : Suivi médical pour détecter et traiter les complications potentielles."
    ],
    "La filariose lymphatique": [
        "Traitement médicamenteux : L'ivermectine, l'albendazole et la doxycycline sont utilisés pour traiter la filariose lymphatique.",
        "Prévention : Utiliser des répulsifs et dormir sous des moustiquaires imprégnées d'insecticide pour éviter les piqûres de moustiques.",
        "Hygiène personnelle : Maintenir une bonne hygiène corporelle pour prévenir les infections secondaires.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ],
    "La leishmaniose cutanée": [
        "Traitement médicamenteux : Les antimoniés pentavalents sont le traitement de choix. Les alternatives incluent l'amphotéricine B et la miltéfosine.",
        "Prévention : Utiliser des répulsifs et dormir sous des moustiquaires imprégnées d'insecticide pour éviter les piqûres de phlébotomes.",
        "Hygiène personnelle : Maintenir une bonne hygiène corporelle pour prévenir les infections secondaires.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ],
    "Leishmaniose cutanéo-muqueuse": [
        "Traitement médicamenteux : Les antimoniés pentavalents sont le traitement de choix. Les alternatives incluent l'amphotéricine B et la miltéfosine.",
        "Prévention : Utiliser des répulsifs et dormir sous des moustiquaires imprégnées d'insecticide pour éviter les piqûres de phlébotomes.",
        "Hygiène personnelle : Maintenir une bonne hygiène corporelle pour prévenir les infections secondaires.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ],
    "Schistosomiase intestinale": [
        "Traitement médicamenteux : Le praziquantel est le traitement de choix pour la schistosomiase intestinale.",
        "Suivi médical : Un examen de suivi est recommandé 1 à 2 mois après le traitement pour s'assurer de la guérison du patient. Si des œufs sont toujours présents, le traitement peut être réitéré.",
        "Prévention : Éviter le contact avec des eaux douces contaminées par les larves du parasite. Utiliser des installations sanitaires adéquates et avoir accès à de l'eau potable.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur l'importance du dépistage précoce et du traitement rapide pour éviter la progression vers des formes sévères de la maladie."
    ],
    "Virus Marburg": [
        "Isolement : Les patients atteints du virus Marburg doivent être isolés pour éviter la propagation de la maladie.",
        "Traitement symptomatique : Hydratation, maintien de l'équilibre électrolytique et traitement des symptômes spécifiques.",
        "Précautions standard : Porter des équipements de protection individuelle (EPI) et appliquer des mesures de précaution pour le personnel soignant en contact avec des patients atteints du virus Marburg.",
        "Surveillance des contacts : Identifier et suivre les proches contacts des patients atteints du virus Marburg et leur administrer une surveillance médicale."
    ],
    "Paludisme": [
        "Traitement médicamenteux : Les antipaludéens comme l'artémisinine ou la quinine sont utilisés pour traiter le paludisme.",
        "Prévention : Utiliser des moustiquaires imprégnées d'insecticide et des répulsifs pour éviter les piqûres de moustiques. Prendre des médicaments prophylactiques lors de voyages dans des zones endémiques.",
        "Hygiène personnelle : Maintenir une bonne hygiène corporelle pour prévenir les infections secondaires.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ],
    "Fièvre jaune": [
        "Vaccination : La vaccination est la mesure la plus efficace pour prévenir la fièvre jaune. Une dose unique du vaccin confère une immunité à vie.",
        "Prévention des piqûres de moustiques : Utiliser des répulsifs, porter des vêtements couvrants et dormir sous des moustiquaires imprégnées d'insecticide.",
        "Élimination des gîtes larvaires : Éliminer les eaux stagnantes autour des habitations pour réduire la reproduction des moustiques.",
        "Surveillance médicale : Les voyageurs se rendant dans des zones endémiques doivent être vaccinés au moins 10 jours avant leur départ."
    ],
    "La leishmaniose viscérale": [
        "Traitement médicamenteux : Les antimoniés pentavalents sont le traitement de choix. Les alternatives incluent l'amphotéricine B et la miltéfosine.",
        "Hygiène personnelle : Maintenir une bonne hygiène corporelle pour prévenir les infections secondaires.",
        "Prévention : Utiliser des répulsifs et dormir sous des moustiquaires imprégnées d'insecticide pour éviter les piqûres de phlébotomes.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ],
    "Onchocercose": [
        "Traitement médicamenteux : L'ivermectine est le traitement de choix pour l'onchocercose.",
        "Prévention : Utiliser des répulsifs et porter des vêtements protecteurs pour éviter les piqûres de simulies.",
        "Hygiène personnelle : Maintenir une bonne hygiène corporelle pour prévenir les infections secondaires.",
        "Éducation sanitaire : Sensibiliser les populations à risque sur les mesures de prévention et l'importance du dépistage précoce."
    ]
}

@app.route('/get_measurements', methods=['GET'])
def get_measurements():
    disease = request.args.get('disease')  # Récupérer la maladie prédite depuis les paramètres de la requête
    measures = mesures.get(disease, [])  # Récupérer les mesures pour la maladie prédite
    return jsonify({"data": measures}), 200

if __name__ == '__main__':
    app.run(debug=True)

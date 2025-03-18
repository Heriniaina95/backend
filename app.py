from flask import Flask, request, jsonify
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Permet les requêtes CORS

# 🟢 Charger le modèle correctement
try:
    model = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=15)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))
    model.eval()  # Mode évaluation
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    exit(1)

# 🔹 Classes (Modifier selon ton dataset)
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
        "description": "La brucellose est une infection bactérienne transmise par les animaux. Elle peut provoquer de la fièvre, des douleurs musculaires et des sueurs.",
        "countries": "Présente en Afrique, Asie, Amérique du Sud.",
        "favorableEnvironment": "Présente dans les zones rurales avec un contact avec le bétail et les animaux d'élevage."
    },
    "data_final_chikungunya": {
        "name": "Chikungunya",
        "description": "Le chikungunya est une maladie virale transmise par les moustiques, caractérisée par de la fièvre et des douleurs articulaires.",
        "countries": "Présente principalement dans les régions tropicales comme l'Afrique, l'Asie et les Caraïbes.",
        "favorableEnvironment": "Les zones tropicales humides et les environnements où les moustiques sont nombreux."
    },
    "data_final_dengue": {
        "name": "Dengue",
        "description": "La dengue est une infection virale transmise par les moustiques. Elle peut provoquer de la fièvre élevée et des douleurs musculaires.",
        "countries": "Présente en Asie, Afrique, Amérique Latine.",
        "favorableEnvironment": "Zones tropicales et subtropicales, où les moustiques vectoriels sont présents."
    },
    "data_final_fievrehemoragique": {
        "name": "Fièvre hémorragique",
        "description": "Une maladie virale souvent grave, pouvant entraîner des saignements, des défaillances organiques et la mort.",
        "countries": "Présente en Afrique et en Asie.",
        "favorableEnvironment": "Zones tropicales où les virus sont transmis par des hôtes animaux ou des moustiques."
    },
    "data_final_fievrejaune": {
        "name": "Fièvre jaune",
        "description": "La fièvre jaune est une infection virale transmise par les moustiques, pouvant entraîner des symptômes graves comme des douleurs abdominales et des vomissements.",
        "countries": "Présente en Afrique subsaharienne et en Amérique latine.",
        "favorableEnvironment": "Les zones tropicales où les moustiques vectoriels sont présents."
    },
    "data_final_filariose": {
        "name": "Filariose",
        "description": "La filariose est une infection parasitaire provoquée par des vers filaires, souvent transmise par les moustiques.",
        "countries": "Présente dans les régions tropicales et subtropicales.",
        "favorableEnvironment": "Les zones où les moustiques transmettent les vers, généralement en Asie, Afrique et Amérique Latine."
    },
    "data_final_leishmaniose": {
        "name": "Leishmaniose",
        "description": "Maladie parasitaire transmise par les phlébotomes (mouches des sables), qui peut affecter la peau ou les organes internes.",
        "countries": "Présente en Afrique, Asie, Amérique latine.",
        "favorableEnvironment": "Les régions désertiques et semi-désertiques, où les phlébotomes vivent."
    },
    "data_final_onchocercose": {
        "name": "Onchocercose",
        "description": "Maladie parasitaire causée par des vers, transmise par les mouches noires, pouvant entraîner la cécité.",
        "countries": "Présente en Afrique subsaharienne.",
        "favorableEnvironment": "Les zones riveraines où les mouches noires sont présentes."
    },
    "data_final_pest": {
        "name": "Peste",
        "description": "Maladie bactérienne transmise par les puces, pouvant entraîner des symptômes graves comme de la fièvre et des ganglions enflés.",
        "countries": "Présente principalement en Asie, Afrique et Amérique.",
        "favorableEnvironment": "Zones de contact avec des rongeurs et des puces."
    },
    "data_final_rouge": {
        "name": "Rougeole",
        "description": "La rougeole est une infection virale qui provoque une éruption cutanée et des symptômes respiratoires.",
        "countries": "Présente dans de nombreux pays, mais plus fréquente dans les zones à faible couverture vaccinale.",
        "favorableEnvironment": "Particulièrement fréquente dans les régions où la couverture vaccinale est insuffisante."
    },
    "data_final_schistosomiase": {
        "name": "Schistosomiase",
        "description": "Maladie parasitaire transmise par des vers, souvent liée à l'eau douce, qui peut affecter divers organes.",
        "countries": "Présente en Afrique, Asie, Amérique latine.",
        "favorableEnvironment": "Zones avec des étangs ou rivières stagnantes."
    },
    "data_final_trypanosomiase": {
        "name": "Trypanosomiase",
        "description": "Maladie parasitaire transmise par la mouche tsé-tsé, pouvant affecter le système nerveux.",
        "countries": "Présente en Afrique subsaharienne.",
        "favorableEnvironment": "Zones rurales et forestières où vivent les mouches tsé-tsé."
    },
    "data_final_virus": {
        "name": "Virus",
        "description": "Maladies virales diverses, souvent transmises par des insectes ou des animaux.",
        "countries": "Mondialement, selon le virus.",
        "favorableEnvironment": "Environnements où le vecteur du virus est présent."
    },
    "data_final_zik": {
        "name": "Zika",
        "description": "Le virus Zika est transmis par les moustiques et peut entraîner des malformations congénitales chez les nouveau-nés.",
        "countries": "Présente dans les régions tropicales, particulièrement en Amérique latine et Asie.",
        "favorableEnvironment": "Les régions humides où les moustiques Aedes sont abondants."
    },
    "paludisme": {
        "name": "Paludisme",
        "description": "Le paludisme est une maladie infectieuse transmise par les moustiques, causée par des parasites du genre Plasmodium.",
        "countries": "Présente en Afrique subsaharienne, Asie et certaines parties de l'Amérique latine.",
        "favorableEnvironment": "Les zones tropicales où les moustiques du genre Anopheles prolifèrent."
    }
}

# Création du dossier d'uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔹 Transformation pour les images
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

    # 🔹 Obtenir l'indice et le nom de la classe
    predicted_index = torch.argmax(prediction, dim=1).item()
    predicted_label = classes[predicted_index]  # Récupérer le nom de la classe

    # Obtenir l'explication de la maladie
    disease_info = disease_explanation.get(predicted_label, {
        "name": "Inconnu",
        "description": "Aucune information disponible.",
        "countries": "Non spécifié",
        "favorableEnvironment": "Non spécifié"
    })

    return jsonify({
        "image_prediction": predicted_label,
        "disease_info": disease_info
    })

if __name__ == '__main__':
    app.run(debug=True)

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
    # Créer à nouveau le modèle, avec les mêmes paramètres que lors de l'entraînement
    model = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=15)  # Modifiez le nombre de classes ici

    # Charger les poids sauvegardés
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))  # S'assurer que le modèle est bien sur CPU
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
    return transform(image).unsqueeze(0)  # Ajouter une dimension batch

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

    return jsonify({"image_prediction": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)

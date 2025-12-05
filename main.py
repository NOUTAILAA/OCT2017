from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# -----------------------------
# 1) CONFIG API
# -----------------------------
app = FastAPI(
    title="OCT Retinal Disease Classification API",
    description="API qui classe une image OCT en : CNV, DME, DRUSEN ou Normal Retina.",
    version="1.0.0",
)

# -----------------------------
# 2) CLASSES & TRANSFORMS
# -----------------------------
class_names = [
    "Choroidal Neovascularization (CNV)",
    "Diabetic Macular Edema (DME)",
    "Drusen - Age-related Macular Degeneration (DRUSEN)",
    "Normal Retina",
]

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 3) CHARGER LE MODELE
# -----------------------------
def load_model(model_path: str = "resnet18_oct_best.pth"):
    # Même architecture que dans ton notebook
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()  # chargé une fois au démarrage


# -----------------------------
# 4) ENDPOINTS
# -----------------------------
@app.get("/", tags=["Healthcheck"])
def root():
    return {"message": "OCT API is running. Go to /docs for Swagger UI."}


@app.post("/predict-oct", tags=["OCT Classification"])
async def predict_oct(file: UploadFile = File(...)):
    """
    Upload une image OCT (JPG/PNG) et retourne la classe prédite.
    """

    # Lire les bytes du fichier
    image_bytes = await file.read()

    # Ouvrir l'image avec PIL
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Le fichier uploadé n'est pas une image valide."},
        )

    # Appliquer les transforms
    img_tensor = transform(img).unsqueeze(0)  # shape [1, 3, H, W]

    # Prédiction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

    # Construire réponse
    return {
        "filename": file.filename,
        "prediction_index": pred_idx,
        "prediction_label": class_names[pred_idx],
        "probabilities": [
            {"class": class_names[i], "probability": float(probs[i])}
            for i in range(len(class_names))
        ],
    }

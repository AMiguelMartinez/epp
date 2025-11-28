# main.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.local_model import predict_local
from services.roboflow_model import predict_roboflow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Requeridos por contexto/modelo
REQUIRED_MEDICAL = ["lab_coat", "stethoscope"]
REQUIRED_INDUSTRIAL = ["helmet", "vest", "goggles", "gloves", "mask"]  # ajusta a tus clases reales


@app.post("/predict")
async def predict(
    model: str = Query(..., description='"local" o "roboflow"'),
    file: UploadFile = File(...),
):
    """
    model = "local"    -> modelo YOLO local (best.pt, contexto médico)
    model = "roboflow" -> workflow Roboflow (contexto industrial)
    """

    image_bytes = await file.read()

    if model == "local":
        detections = predict_local(image_bytes)
        required = REQUIRED_MEDICAL
    elif model == "roboflow":
        detections = predict_roboflow(image_bytes)
        required = REQUIRED_INDUSTRIAL
    else:
        raise HTTPException(status_code=400, detail="Modelo inválido")

    # Extraer clases detectadas (sin duplicados)
    detected_classes = sorted(
        {d.get("class") for d in detections if d.get("class")}
    )

    # Calcular faltantes
    missing = [c for c in required if c not in detected_classes]

    return {
        "model": model,
        "detections": detections,          # lista completa con bboxes
        "detected": detected_classes,      # clases presentes
        "missing": missing,                # clases requeridas que faltan
        "is_complete": len(missing) == 0,  # bool
    }

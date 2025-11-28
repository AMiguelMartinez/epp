from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Cargar el modelo local una sola vez al iniciar el microservicio
model = YOLO("best.pt")

@app.post("/predict/local")
async def predict_local(file: UploadFile = File(...)):

    # Convertir archivo en imagen
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Inference
    results = model(image)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy.tolist()
        })

    return JSONResponse(content={"detections": detections})

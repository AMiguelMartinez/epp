# services/local_model.py
from ultralytics import YOLO
import cv2
import numpy as np

model_local = YOLO("best.pt")

def predict_local(image_bytes: bytes):
    """
    Recibe bytes de imagen y devuelve detecciones normalizadas:
    [{ class: str, confidence: float, bbox: [x1,y1,x2,y2] }, ...]
    """
    # Decodificar bytes a imagen OpenCV
    npimg = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        # No se pudo decodificar la imagen
        return []

    # Puedes ajustar conf / imgsz según tu modelo
    results = model_local(image, conf=0.25)[0]

    detections = []

    # Si no hay boxes, devolvemos lista vacía
    if results.boxes is None or len(results.boxes) == 0:
        return []

    for box in results.boxes:
        # ID de clase
        cls_id = int(box.cls[0])
        # Nombre de clase (lab_coat, stethoscope, etc.)
        label = str(model_local.names.get(cls_id, cls_id))

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

        detections.append({
            "class": label,            # 👈 nombre, no índice
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],  # formato xyxy
        })

    return detections

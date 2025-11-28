# services/roboflow_model.py
from inference_sdk import InferenceHTTPClient
import base64
import concurrent.futures
import os

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)


def _call_roboflow_internal(image_bytes: bytes):
    """Lógica real de llamada + parseo al workflow de Roboflow."""

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    result = client.run_workflow(
        workspace_name="sst-sidp0",
        workflow_id="find-vests-goggles-helmets-gloves-and-masks",
        images={"image": image_b64},
        use_cache=True,
    )

    print("DEBUG Roboflow raw result:", result)

    # ---- Aquí parseas la respuesta según ya la tengas definida ----
    # Para no liarnos, asumo que result ya es lista de dicts con x,y,width,height,confidence,class.
    # Ajusta estas claves a lo que realmente devuelva tu workflow.

    predictions = result.get("predictions", []) if isinstance(result, dict) else []

    detections = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue

        label = str(pred.get("class", "unknown"))
        conf = float(pred.get("confidence", 0.0))

        x = pred.get("x")
        y = pred.get("y")
        w = pred.get("width")
        h = pred.get("height")

        if None not in (x, y, w, h):
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            bbox = [float(x1), float(y1), float(x2), float(y2)]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]

        detections.append(
            {
                "class": label,
                "confidence": conf,
                "bbox": bbox,
            }
        )

    return detections


def predict_roboflow(image_bytes: bytes, timeout_seconds: float = 8.0):
    """
    Envuelve la llamada a Roboflow con un timeout.
    Si la llamada tarda más de `timeout_seconds`, devolvemos [] (o fallback).
    Compatible con main.py porque sigue retornando una lista de detecciones.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_roboflow_internal, image_bytes)
            return future.result(timeout=timeout_seconds)

    except concurrent.futures.TimeoutError:
        print(f"⚠️ Roboflow tardó más de {timeout_seconds}s. Devolviendo lista vacía.")
        # Si quieres fallback al modelo local:
        # from services.local_model import predict_local
        # return predict_local(image_bytes)
        return []

    except Exception as e:
        print(f"ERROR en predict_roboflow: {e}")
        import traceback
        traceback.print_exc()
        return []

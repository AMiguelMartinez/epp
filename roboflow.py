from inference_sdk import InferenceHTTPClient
import base64

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="aRauZwUW9e3JnTzlXSkc"
)

def predict_roboflow(image_bytes: bytes):
    """Recibe bytes de imagen y devuelve las predicciones limpias del workflow."""

    # Convertir a base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Ejecutar workflow
    result = client.run_workflow(
        workspace_name="sst-sidp0",
        workflow_id="find-vests-goggles-helmets-gloves-and-masks",
        images={"image": image_b64},
        use_cache=True
    )

    # ==========================
    # Procesar estructura del workflow
    # ==========================
    predictions = None

    for item in result:
        # Formato 1
        if "name" in item and "value" in item:
            if item["name"] == "predictions":
                predictions = item["value"]

        # Formato 2 nuevo
        elif "predictions" in item:
            predictions = item["predictions"]

    return predictions

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


def norm(label: str) -> str:
    """
    Normaliza labels para comparación:
    - quita espacios al inicio/fin
    - pasa a minúsculas
    """
    return (label or "").strip().lower()


# Variantes aceptadas para cada requerido (todo ya en minúsculas)
ALIASES = {
    # MEDICAL
    "white coat": {
        "white coat",
        "medical uniform",
        "doctor",
        "nurse",
        "surgeon",
        "surgical gown",
    },
    "stethoscope": {
        "stethoscope",
        "nurse",
    },
    "face mask": {
        "face mask",
        "surgical mask",
        "safety_mask",
        "nurse",
    },
    "gloves": {
        "gloves",
        "safety_gloves",
        "latex gloves",
        "nurse",
        "afety_gloves",  # por si el modelo devuelve este typo
    },

    # CONSTRUCTION
    "helmet": {
        "helmet",
        "hard hat",
        "hard_hat",
    },
    "safety vest": {
        "safety vest",
        "vest",
        "safety_vest",
    },
    "safety boots": {
        "safety boots",
        "boots",
    },
    "protective glasses": {
        "protective glasses",
        "goggles",
        "safety_goggles",
    },
    "ear protection": {
        "ear protection",
    },

    # SECURITY GUARD
    "security outfit": {
        "security outfit",
        "uniform",
    },
    "security guard": {
        "security guard",
        "safety_worker",
        "worker",
        "construction_worker",
    },
    "belt": {
        "belt",
        "tool belt",
    },
    "boots": {
        "boots",
        "safety boots",
    },
    "cap": {
        "cap",
    },
    "handcuffs": {
        "handcuffs",
    },
    "police baton": {
        "police baton",
        "baton",
    },

    # WELDER
    "welding gear": {
        "welding gear",
    },
    "welding mask": {
        "welding mask",
        "welding gear",  # si detecta gear, también damos por cumplida la máscara
    },
    "safety mask": {
        "safety mask",
        "face mask",
        "surgical mask",
        "safety_mask",
    },
}

REQUIRED_BY_CONTEXT = {
    "medical": ["white coat", "stethoscope", "face mask", "gloves"],
    "construction": [
        "helmet",
        "safety vest",
        "safety boots",
        "protective glasses",
        "ear protection",
    ],
    "security_guard": [
        "security outfit",
        "security guard",
        "belt",
        "boots",
        "cap",
        "handcuffs",
        "police baton",
    ],
    "welder": ["welding gear", "welding mask", "gloves", "safety mask"],
}

CORE_REQUIRED_BY_CONTEXT = {
    # Para médicos, pedimos mínimo bata/uniforme + mascarilla
    "medical": ["white coat", "face mask"],

    # Para construcción, mínimo casco
    "construction": ["helmet"],

    # Para guardia, con que el modelo reconozca Security guard ya lo damos por bueno
    "security_guard": ["security guard"],

    # Para soldador, con que detecte Welding gear
    "welder": ["welding gear"],
}


def get_detected_set(detections):
    """Conjunto de clases detectadas, normalizadas."""
    return {norm(d.get("class")) for d in detections if d.get("class")}


def is_required_satisfied(required_label: str, detected: set[str]) -> bool:
    """
    Un requerido se cumple si al menos uno de sus alias aparece en detected.
    Todo se compara normalizado (minúsculas + strip).
    """
    key = norm(required_label)
    variants = ALIASES.get(key, {key})
    return any(v in detected for v in variants)


@app.post("/predict")
async def predict(
    model: str = Query(..., description='"local" o "roboflow"'),
    context: str = Query(..., description='"medical", "construction", "security_guard", "welder"'),
    file: UploadFile = File(...),
):
    """
    model   = "local"    -> YOLO best.pt
    model   = "roboflow" -> workflow en la nube (si lo sigues usando)
    context = "medical" | "construction" | "security_guard" | "welder"
    """

    image_bytes = await file.read()

    model = norm(model)
    context = norm(context)

    # 1. Ejecutar modelo
    if model == "local":
        detections = predict_local(image_bytes)
    elif model == "roboflow":
        detections = predict_roboflow(image_bytes)
    else:
        raise HTTPException(status_code=400, detail="Modelo inválido")

    # 2. Validar contexto
    if context not in REQUIRED_BY_CONTEXT:
        raise HTTPException(status_code=400, detail="Contexto inválido")

    required = REQUIRED_BY_CONTEXT[context]
    core_required = CORE_REQUIRED_BY_CONTEXT.get(context, required)

    # 3. Conjunto normalizado de clases detectadas
    detected_set = get_detected_set(detections)

    # Para mostrar al cliente, mantenemos también las clases originales
    detected_classes = sorted(
        {d.get("class") for d in detections if d.get("class")}
    )

    # 4. Faltantes "ideales" (todas las piezas)
    missing = [
        req for req in required
        if not is_required_satisfied(req, detected_set)
    ]

    # 5. Faltantes "core" (para decidir is_complete)
    missing_core = [
        req for req in core_required
        if not is_required_satisfied(req, detected_set)
    ]

    is_complete = len(missing_core) == 0

    return {
        "model": model,
        "context": context,
        "detections": detections,        # caja cruda del modelo
        "detected": detected_classes,    # nombres de clases para UI
        "required": required,
        "core_required": core_required,
        "missing": missing,              # todo lo ideal que falta
        "missing_core": missing_core,    # lo mínimo crítico que falta
        "is_complete": is_complete,
    }

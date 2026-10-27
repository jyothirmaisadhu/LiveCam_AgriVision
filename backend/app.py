
import io
import os
import time
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

DEVICE = torch.device("cpu")
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

CROP_CONFIG = {
    "tomato": {
        "checkpoint": "tomato_v2_best.pth",
        "display_name": "Tomato",
    },
    "corn": {
        "checkpoint": "corn_best.pth",
        "display_name": "Corn",
    },
    "paddy": {
        "checkpoint": "paddy_v2_best.pth",
        "display_name": "Paddy",
    },
    "chilli": {
        "checkpoint": "chilli_best.pth",
        "display_name": "Chilli",
    },
    "cotton": {
        "checkpoint": "cotton_best.pth",
        "display_name": "Cotton",
    },
}

loaded_models = {}
model_errors = {}


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
def load_crop_model(crop_name, config):
    checkpoint_path = os.path.join(
        MODELS_DIR, config["checkpoint"]
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 160))

    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        input_features, len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return {
        "model": model,
        "classes": classes,
        "transform": image_transform,
        "image_size": image_size,
        "checkpoint": config["checkpoint"],
        "display_name": config["display_name"],
    }


def load_all_models():
    loaded_models.clear()
    model_errors.clear()

    print("\nLoading crop classification models...")

    for crop_name, config in CROP_CONFIG.items():
        try:
            loaded_models[crop_name] = load_crop_model(
                crop_name, config
            )
            print(
                f"[OK] {config['display_name']} model loaded "
                f"from {config['checkpoint']}"
            )
        except Exception as exc:
            model_errors[crop_name] = str(exc)
            print(
                f"[ERROR] Could not load {crop_name}: {exc}"
            )

    print(
        f"Loaded {len(loaded_models)}/{len(CROP_CONFIG)} models.\n"
    )


# --------------------------------------------------
# FASTAPI LIFESPAN
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    loaded_models.clear()


app = FastAPI(
    title="LiveCam AgriVision API",
    description=(
        "Plant disease classification API for Tomato, Corn, "
        "Paddy, Chilli and Cotton."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Development frontend origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to LiveCam AgriVision API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": (
            "healthy"
            if len(loaded_models) == len(CROP_CONFIG)
            else "degraded"
        ),
        "device": str(DEVICE),
        "loaded_models": list(loaded_models.keys()),
        "model_count": len(loaded_models),
        "expected_model_count": len(CROP_CONFIG),
        "model_errors": model_errors,
    }


@app.get("/api/crops")
def get_crops():
    return {
        "crops": [
            {
                "id": crop_name,
                "name": config["display_name"],
                "available": crop_name in loaded_models,
                "classes": (
                    loaded_models[crop_name]["classes"]
                    if crop_name in loaded_models
                    else []
                ),
            }
            for crop_name, config in CROP_CONFIG.items()
        ]
    }


# --------------------------------------------------
# IMAGE PREDICTION
# --------------------------------------------------
@app.post("/api/predict/{crop_name}")
async def predict(
    crop_name: str,
    file: UploadFile = File(...),
):
    crop_name = crop_name.strip().lower()

    if crop_name not in CROP_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported crop '{crop_name}'. "
                f"Choose one of: {', '.join(CROP_CONFIG)}"
            ),
        )

    if crop_name not in loaded_models:
        raise HTTPException(
            status_code=503,
            detail=(
                f"The {crop_name} model is unavailable. "
                "Check the backend terminal and model checkpoint."
            ),
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        image_bytes = await file.read(
            MAX_IMAGE_SIZE_BYTES + 1
        )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 10 MB.",
            )

        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Unable to read this image. Try a valid JPG or PNG.",
        )
    finally:
        await file.close()

    crop_model = loaded_models[crop_name]
    input_tensor = crop_model["transform"](image).unsqueeze(0)

    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            logits = crop_model["model"](input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_probability, top_index = torch.max(
                probabilities, dim=0
            )
    except Exception as exc:
        print(f"Inference error for {crop_name}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend terminal.",
        )

    inference_ms = (time.perf_counter() - start_time) * 1000

    predicted_class = crop_model["classes"][top_index.item()]
    confidence = float(top_probability.item()) * 100

    top_k = min(3, len(crop_model["classes"]))
    top_values, top_indices = torch.topk(
        probabilities, k=top_k
    )

    alternatives = [
        {
            "class_name": crop_model["classes"][index.item()],
            "confidence": round(float(value.item()) * 100, 2),
        }
        for value, index in zip(top_values, top_indices)
    ]

    return {
        "success": True,
        "crop": crop_name,
        "crop_name": crop_model["display_name"],
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "top_predictions": alternatives,
        "inference_time_ms": round(inference_ms, 2),
        "model": crop_model["checkpoint"],
        "notice": (
            "This is an image classification prediction, not a "
            "confirmed diagnosis. Results can be affected by image "
            "quality and real-world conditions."
        ),
    }

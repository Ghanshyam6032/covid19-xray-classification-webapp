import io
import os
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import modal
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("neuroscan-covid-xray")

# ---------------------------------------------------------------------------
# Paths & Constants (Pure Backend - Only Model & Labels)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONTAINER_DIR = Path("/app")

MODEL_PATH = (
    CONTAINER_DIR / "keras_model.h5"
    if (CONTAINER_DIR / "keras_model.h5").exists()
    else BASE_DIR / "keras_model.h5"
)
LABELS_PATH = (
    CONTAINER_DIR / "labels.txt"
    if (CONTAINER_DIR / "labels.txt").exists()
    else BASE_DIR / "labels.txt"
)

REPORTS_DIR = Path("/tmp/neuroscan_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

APP_VERSION = "2.1.0"
MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
}

# ---------------------------------------------------------------------------
# Modal Configuration (Strictly Backend Only - Python 3.11 + TF 2.15 Keras 2)
# ---------------------------------------------------------------------------
modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgomp1",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("keras_model.h5", remote_path="/app/keras_model.h5", copy=True)
    .add_local_file("labels.txt", remote_path="/app/labels.txt", copy=True)
)

app = modal.App("neuroscan-covid-xray")

# ---------------------------------------------------------------------------
# In-Memory Application State
# ---------------------------------------------------------------------------
ml_state: Dict[str, Any] = {
    "model": None,
    "labels": [],
    "model_loaded": False,
    "load_error": None,
    "loaded_at": None,
    "input_shape": (224, 224, 3),
    "tf_version": None,
}

prediction_history: List[Dict[str, Any]] = []
reports_registry: Dict[str, str] = {}
startup_timestamp: float = time.time()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def parse_labels_file(labels_path: Path) -> List[str]:
    if not labels_path.exists():
        logger.warning(f"Labels file not found at {labels_path}. Using defaults.")
        return ["COVID-19", "Normal"]

    cleaned_labels: List[str] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                cleaned_labels.append(parts[1].strip())
            else:
                cleaned_labels.append(raw)
    return cleaned_labels if cleaned_labels else ["COVID-19", "Normal"]


def preprocess_xray_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {str(exc)}",
        )

    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0

    data = np.ndarray(shape=(1, target_size[0], target_size[1], 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data


# ---------------------------------------------------------------------------
# FastAPI Lifespan (Lazy TensorFlow Import inside Modal Container)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    if ml_state["model_loaded"] and ml_state["model"] is not None:
        yield
        return

    logger.info("Initializing FastAPI lifespan: importing TensorFlow and loading model...")
    try:
        import tensorflow as tf

        ml_state["tf_version"] = tf.__version__
        logger.info(f"TensorFlow {tf.__version__} imported successfully.")

        class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        logger.info(f"Loading Keras model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
            custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D},
        )

        try:
            in_shape = model.input_shape
            if isinstance(in_shape, tuple) and len(in_shape) == 4:
                ml_state["input_shape"] = (
                    in_shape[1] or 224,
                    in_shape[2] or 224,
                    in_shape[3] or 3,
                )
        except Exception:
            ml_state["input_shape"] = (224, 224, 3)

        labels = parse_labels_file(LABELS_PATH)

        dummy_h, dummy_w, dummy_c = ml_state["input_shape"]
        dummy_input = np.zeros((1, dummy_h, dummy_w, dummy_c), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)

        ml_state["model"] = model
        ml_state["labels"] = labels
        ml_state["model_loaded"] = True
        ml_state["load_error"] = None
        ml_state["loaded_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Model loaded and warmed up successfully. Classes: {labels}")

    except Exception as exc:
        logger.exception(f"Failed to load model during lifespan startup: {exc}")
        ml_state["model_loaded"] = False
        ml_state["load_error"] = str(exc)

    yield

    logger.info("Shutting down container...")
    ml_state["model"] = None
    ml_state["model_loaded"] = False


# ---------------------------------------------------------------------------
# FastAPI Application (`web_app`)
# ---------------------------------------------------------------------------
web_app = FastAPI(
    title="NeuroScan COVID-19 Chest X-Ray Classification API",
    description="Pure Backend API hosted on Modal for GitHub Pages Frontend.",
    version=APP_VERSION,
    lifespan=lifespan,
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


class ReportRequest(BaseModel):
    prediction_id: Optional[str] = Field(default=None)
    patient_name: Optional[str] = Field(default="Anonymous Patient")
    patient_id: Optional[str] = Field(default="N/A")
    patient_age: Optional[Any] = Field(default="N/A")
    patient_gender: Optional[str] = Field(default="N/A")
    doctor_name: Optional[str] = Field(default="Assigned Physician")
    filename: Optional[str] = Field(default="xray_scan.png")
    prediction: Optional[str] = Field(default=None)
    predicted_class: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=0.0)
    probabilities: Optional[Dict[str, float]] = Field(default_factory=dict)
    notes: Optional[str] = Field(
        default="AI-assisted screening result. Correlate clinically with RT-PCR and radiologist review."
    )


def compute_statistics() -> Dict[str, Any]:
    total = len(prediction_history)
    if total == 0:
        class_counts = {label: 0 for label in ml_state["labels"]}
        return {
            "total_predictions": 0,
            "covid_detected": 0,
            "normal_detected": 0,
            "other_detected": 0,
            "class_distribution": class_counts,
            "average_confidence": 0.0,
            "average_latency_ms": 0.0,
            "average_inference_time_ms": 0.0,
            "last_prediction_at": None,
            "persistence_note": "In-memory container storage (resets on container restart).",
        }

    class_counts: Dict[str, int] = {label: 0 for label in ml_state["labels"]}
    covid_count = 0
    normal_count = 0
    other_count = 0
    confidences: List[float] = []
    inference_times: List[float] = []

    for item in prediction_history:
        pred_label = item.get("prediction", "Unknown")
        class_counts[pred_label] = class_counts.get(pred_label, 0) + 1

        lower_label = pred_label.lower()
        if "covid" in lower_label:
            covid_count += 1
        elif "normal" in lower_label:
            normal_count += 1
        else:
            other_count += 1

        conf = float(item.get("confidence", 0.0))
        conf_pct = conf * 100.0 if conf <= 1.0 else conf
        confidences.append(conf_pct)
        inference_times.append(float(item.get("latency_ms", item.get("inference_time_ms", 0.0))))

    avg_lat = round(sum(inference_times) / total, 2)
    return {
        "total_predictions": total,
        "covid_detected": covid_count,
        "normal_detected": normal_count,
        "other_detected": other_count,
        "class_distribution": class_counts,
        "average_confidence": round(sum(confidences) / total, 2),
        "average_latency_ms": avg_lat,
        "average_inference_time_ms": avg_lat,
        "last_prediction_at": prediction_history[0].get("timestamp"),
        "persistence_note": "In-memory container storage (resets on container scale-down/restart).",
    }


# ---------------------------------------------------------------------------
# Pure JSON API Endpoints
# ---------------------------------------------------------------------------
@web_app.get("/")
async def root():
    return {
        "status": "online",
        "service": "NeuroScan COVID-19 Chest X-Ray Classification API (Backend Only)",
        "version": APP_VERSION,
        "model_loaded": ml_state["model_loaded"],
        "classes": ml_state["labels"],
        "endpoints": [
            "GET /",
            "GET /health",
            "GET /version",
            "GET /model-info",
            "GET /statistics",
            "GET /history",
            "DELETE /history",
            "POST /predict",
            "POST /generate-report",
            "GET /download-report/{report_id}",
        ],
    }


@web_app.get("/health")
async def health_check():
    return {
        "status": "healthy" if ml_state["model_loaded"] else "degraded",
        "model_loaded": ml_state["model_loaded"],
        "load_error": ml_state["load_error"],
        "uptime_seconds": round(time.time() - startup_timestamp, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@web_app.get("/version")
async def get_version():
    import platform

    return {
        "api_version": APP_VERSION,
        "python_version": platform.python_version(),
        "tensorflow_version": ml_state["tf_version"] or "Not loaded yet",
        "framework": "FastAPI + TensorFlow/Keras on Modal",
    }


@web_app.get("/model-info")
async def get_model_info():
    model = ml_state["model"]
    return {
        "model_name": "COVID-19 Chest X-Ray Classifier (Keras .h5)",
        "model_file": MODEL_PATH.name,
        "model_loaded": ml_state["model_loaded"],
        "loaded_at": ml_state["loaded_at"],
        "input_shape": list(ml_state["input_shape"]),
        "classes": ml_state["labels"],
        "num_classes": len(ml_state["labels"]),
        "total_parameters": int(model.count_params()) if model is not None else None,
        "total_layers": len(model.layers) if model is not None else None,
        "tensorflow_version": ml_state["tf_version"],
    }


@web_app.get("/statistics")
async def get_statistics():
    return compute_statistics()


@web_app.get("/history")
async def get_history(limit: int = 50):
    sliced = prediction_history[: max(1, min(limit, 200))]
    return {
        "count": len(sliced),
        "total": len(prediction_history),
        "history": sliced,
    }


@web_app.delete("/history")
async def clear_history():
    cleared_count = len(prediction_history)
    prediction_history.clear()
    return {
        "status": "success",
        "message": f"Cleared {cleared_count} prediction record(s) from history.",
        "cleared_count": cleared_count,
    }


@web_app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    if not ml_state["model_loaded"] or ml_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {ml_state['load_error']}",
        )

    filename = file.filename or "uploaded_xray.png"
    ext = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    if ext and ext not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or content_type}'. Allowed: JPG, JPEG, PNG, BMP, WEBP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB.",
        )

    target_h, target_w, _ = ml_state["input_shape"]
    input_tensor = preprocess_xray_image(image_bytes, target_size=(target_h, target_w))

    start_t = time.perf_counter()
    raw_preds = ml_state["model"].predict(input_tensor, verbose=0)
    inference_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    probs = raw_preds[0].tolist()
    labels = ml_state["labels"]

    probabilities_raw_dict: Dict[str, float] = {}
    probabilities_percent_dict: Dict[str, float] = {}
    all_predictions_list: List[Dict[str, Any]] = []

    for idx, score in enumerate(probs):
        label_name = labels[idx] if idx < len(labels) else f"Class_{idx}"
        score_float = float(score)
        probabilities_raw_dict[label_name] = round(score_float, 6)
        probabilities_percent_dict[label_name] = round(score_float * 100.0, 2)
        all_predictions_list.append(
            {
                "class": label_name,
                "label": label_name,
                "confidence": round(score_float, 6),
                "percentage": round(score_float * 100.0, 2),
            }
        )

    best_idx = int(np.argmax(probs))
    best_label = labels[best_idx] if best_idx < len(labels) else f"Class_{best_idx}"
    best_conf = float(probs[best_idx])
    best_conf_pct = round(best_conf * 100.0, 2)

    prediction_id = str(uuid.uuid4())
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    is_covid = "covid" in best_label.lower()
    risk_level = (
        "High"
        if (is_covid and best_conf >= 0.7)
        else ("Moderate" if is_covid else "Low")
    )

    record = {
        "id": prediction_id,
        "prediction_id": prediction_id,
        "filename": filename,
        "prediction": best_label,
        "predicted_class": best_label,
        "class_index": best_idx,
        "confidence": best_conf_pct,
        "confidence_raw": round(best_conf, 6),
        "confidence_percentage": best_conf_pct,
        "probabilities": probabilities_percent_dict,
        "probabilities_raw": probabilities_raw_dict,
        "predictions": all_predictions_list,
        "risk_level": risk_level,
        "latency_ms": inference_ms,
        "inference_time_ms": inference_ms,
        "file_size_bytes": len(image_bytes),
        "timestamp": timestamp_iso,
    }

    prediction_history.insert(0, record)
    if len(prediction_history) > 500:
        prediction_history.pop()

    return record


@web_app.post("/generate-report")
async def generate_pdf_report(payload: ReportRequest):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    matched_record: Optional[Dict[str, Any]] = None
    if payload.prediction_id:
        for item in prediction_history:
            if item.get("id") == payload.prediction_id or item.get("prediction_id") == payload.prediction_id:
                matched_record = item
                break

    pred_label = (
        payload.prediction
        or payload.predicted_class
        or (matched_record.get("prediction") if matched_record else None)
        or (prediction_history[0].get("prediction") if prediction_history else "Unknown")
    )

    raw_conf = (
        payload.confidence
        if payload.confidence and payload.confidence > 0
        else (matched_record.get("confidence", 0.0) if matched_record else 0.0)
    )
    conf_pct = round(raw_conf * 100.0, 2) if raw_conf <= 1.0 else round(raw_conf, 2)

    probs_map = (
        payload.probabilities
        if payload.probabilities
        else (matched_record.get("probabilities", {}) if matched_record else {})
    )
    scan_filename = (
        payload.filename
        if payload.filename != "xray_scan.png"
        else (matched_record.get("filename", "xray_scan.png") if matched_record else "xray_scan.png")
    )

    report_id = str(uuid.uuid4())
    pdf_filename = f"NeuroScan_Report_{report_id[:8]}.pdf"
    pdf_path = REPORTS_DIR / pdf_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=8,
    )
    normal_style = styles["BodyText"]

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elements = [
        Paragraph("NeuroScan AI - Chest X-Ray Diagnostic Report", title_style),
        Paragraph(f"<b>Report ID:</b> {report_id}", normal_style),
        Paragraph(f"<b>Generated At:</b> {now_str}", normal_style),
        Spacer(1, 14),
        Paragraph("1. Patient & Scan Information", section_style),
    ]

    patient_data = [
        ["Patient Name", str(payload.patient_name or "Anonymous Patient")],
        ["Age / Gender", f"{payload.patient_age or 'N/A'} / {payload.patient_gender or 'N/A'}"],
        ["Attending Physician", str(payload.doctor_name or "Assigned Physician")],
        ["Source Image File", str(scan_filename)],
    ]
    patient_table = Table(patient_data, colWidths=[2.2 * inch, 4.3 * inch])
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    elements.append(patient_table)

    elements.append(Paragraph("2. AI Classification Summary", section_style))
    result_data = [
        ["Predicted Classification", str(pred_label)],
        ["Confidence Score", f"{conf_pct:.2f}%"],
        ["Model Architecture", "Keras Deep CNN (224x224x3)"],
    ]
    for cls_name, cls_prob in probs_map.items():
        val = cls_prob * 100.0 if cls_prob <= 1.0 else cls_prob
        result_data.append([f"Probability ({cls_name})", f"{val:.2f}%"])

    result_table = Table(result_data, colWidths=[2.2 * inch, 4.3 * inch])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eff6ff")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#93c5fd")),
            ]
        )
    )
    elements.append(result_table)

    elements.append(Paragraph("3. Clinical Notes & Medical Disclaimer", section_style))
    elements.append(Paragraph(payload.notes or "", normal_style))
    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by an automated deep learning model for "
            "research and decision-support purposes only. Always consult a qualified radiologist.",
            normal_style,
        )
    )

    doc.build(elements)
    reports_registry[report_id] = str(pdf_path)

    return {
        "status": "success",
        "report_id": report_id,
        "filename": pdf_filename,
        "download_url": f"/download-report/{report_id}",
        "generated_at": now_str,
    }


@web_app.get("/download-report/{report_id}")
async def download_report(report_id: str):
    pdf_path_str = reports_registry.get(report_id)
    if not pdf_path_str:
        candidate = REPORTS_DIR / f"NeuroScan_Report_{report_id[:8]}.pdf"
        if candidate.exists():
            pdf_path_str = str(candidate)

    if not pdf_path_str or not Path(pdf_path_str).exists():
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    return FileResponse(
        path=pdf_path_str,
        media_type="application/pdf",
        filename=Path(pdf_path_str).name,
    )


# ---------------------------------------------------------------------------
# Modal ASGI Entrypoint
# ---------------------------------------------------------------------------
@app.function(
    image=modal_image,
    cpu=2.0,
    memory=4096,
    timeout=300,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=15)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:web_app", host="0.0.0.0", port=8000, reload=True)
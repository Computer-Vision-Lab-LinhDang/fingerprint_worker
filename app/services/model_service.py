"""Model management service — download and track loaded models."""

import json
import logging
import os
import threading

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Default model directory (relative to worker root)
MODEL_DIR = os.path.join(os.getcwd(), "models")
STATE_FILE = os.path.join(MODEL_DIR, "loaded_models.json")


class ModelService:
    """Manages model downloads and tracks which models are loaded."""

    def __init__(self):
        self._loaded_models = {}  # {"embedding": "embedding_v1", ...}
        self._lock = threading.Lock()
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._load_state()

    # ── State persistence ────────────────────────────────────
    def _load_state(self):
        """Load saved model state from disk."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    self._loaded_models = json.load(f)
                logger.info("Loaded model state: %s", self._loaded_models)
        except Exception as exc:
            logger.error("Failed to load model state: %s", exc)

    def _save_state(self):
        """Save current model state to disk."""
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self._loaded_models, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save model state: %s", exc)

    # ── Properties ───────────────────────────────────────────
    @property
    def loaded_models(self):
        """Return dict of loaded models: {type: name}."""
        with self._lock:
            return dict(self._loaded_models)

    # ── Download ─────────────────────────────────────────────
    def download_model(self, model_type, model_name, version, download_url):
        """
        Download a model file from presigned URL.
        Saves to: models/{model_type}/{model_name}  (e.g. models/embedding/embedding_v1.onnx)
        Returns (success, error_message).
        """
        save_dir = os.path.join(MODEL_DIR, model_type)
        save_path = os.path.join(save_dir, model_name)

        try:
            os.makedirs(save_dir, exist_ok=True)

            logger.info(
                "Downloading model: %s/%s → %s",
                model_type, model_name, save_path,
            )

            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

            file_size = os.path.getsize(save_path)
            logger.info(
                "✅ Model downloaded: %s/%s (%.1f MB)",
                model_type, model_name, file_size / (1024 * 1024),
            )

            # Update loaded models
            with self._lock:
                self._loaded_models[model_type] = model_name
                self._save_state()

            return True, None

        except requests.RequestException as exc:
            error = "Download failed: {}".format(exc)
            logger.error("❌ %s", error)
            return False, error

        except Exception as exc:
            error = "Model save failed: {}".format(exc)
            logger.error("❌ %s", error)
            return False, error

    # ── List local models ────────────────────────────────────
    def list_local_models(self):
        """List all .onnx model files on disk."""
        results = []
        if not os.path.exists(MODEL_DIR):
            return results

        for model_type in os.listdir(MODEL_DIR):
            type_dir = os.path.join(MODEL_DIR, model_type)
            if not os.path.isdir(type_dir):
                continue
            for filename in os.listdir(type_dir):
                filepath = os.path.join(type_dir, filename)
                if os.path.isfile(filepath) and filename.endswith(".onnx"):
                    size = os.path.getsize(filepath)
                    results.append({
                        "model_type": model_type,
                        "model_name": filename,
                        "path": filepath,
                        "size_bytes": size,
                    })
        return results


# ── Singleton ────────────────────────────────────────────────
_service = None


def get_model_service():
    global _service
    if _service is None:
        _service = ModelService()
    return _service

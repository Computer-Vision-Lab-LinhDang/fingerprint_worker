"""
Task processing service — handles embed/match tasks from orchestrator.
Downloads image, runs inference, returns result via MQTT.
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# ── Cached inference engine (singleton) ──────────────────────
_cached_engine = None
_cached_model_path = None


def _get_cached_engine(onnx_path):
    """Get or create cached inference engine to avoid PyCUDA context crash."""
    global _cached_engine, _cached_model_path
    if _cached_engine is None or _cached_model_path != onnx_path:
        from app.services.inference_service import create_inference_engine
        _cached_engine = create_inference_engine(onnx_path)
        _cached_engine.load()
        _cached_model_path = onnx_path
        logger.info("Loaded inference engine: %s", onnx_path)
    return _cached_engine


class TaskService(object):
    """Handles task processing logic with real inference."""

    def __init__(self, mqtt_client):
        self._mqtt_client = mqtt_client

    def process_embed(self, task_data):
        """
        Process an embedding task:
        1. Download image from presigned URL
        2. Run inference to get embedding vector
        3. Publish result back to orchestrator

        Args:
            task_data: dict with task_id, image_url, model_name, extra
        """
        task_id = task_data.get("task_id", "")
        image_url = task_data.get("image_url", "")
        model_name = task_data.get("model_name", "default")
        extra = task_data.get("extra", {})

        logger.info("Processing EMBED task %s", task_id)
        t0 = time.time()

        try:
            # 1. Download image from presigned URL
            logger.info("Downloading image from MinIO...")
            image_bytes = self._download_image(image_url)
            logger.info("Downloaded %d bytes", len(image_bytes))

            # 2. Run inference (use cached engine)
            from app.services.inference_service import (
                create_inference_engine,
                preprocess_from_bytes,
                normalize_embedding,
            )

            engine = _get_cached_engine(self._find_model(model_name))

            input_data = preprocess_from_bytes(image_bytes)
            raw_output = engine.infer(input_data)
            embedding = normalize_embedding(raw_output)

            elapsed_ms = (time.time() - t0) * 1000

            # 3. Publish result
            result = {
                "task_id": task_id,
                "worker_id": self._mqtt_client.worker_id,
                "status": "completed",
                "result": {
                    "vector": embedding.tolist(),
                    "vector_dim": len(embedding),
                    "model_name": model_name,
                    "processing_time_ms": round(elapsed_ms, 2),
                },
                "processing_time_ms": round(elapsed_ms, 2),
            }

            self._publish_result(task_id, result)
            logger.info(
                "EMBED task %s completed: %dD vector in %.1fms",
                task_id, len(embedding), elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000
            logger.error("EMBED task %s failed: %s", task_id, exc)
            result = {
                "task_id": task_id,
                "worker_id": self._mqtt_client.worker_id,
                "status": "failed",
                "error": str(exc),
                "processing_time_ms": round(elapsed_ms, 2),
            }
            self._publish_result(task_id, result)

    def process_match(self, task_data):
        """Process a matching task (TODO)."""
        logger.info("MATCH task received — not implemented yet")
        return None

    def _download_image(self, url):
        """Download image from presigned URL, return bytes."""
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content

    def _find_model(self, model_name):
        """Find the ONNX model file on disk."""
        model_dir = os.path.join(os.getcwd(), "models")

        # Try exact name first
        for model_type in ("embedding", "matching", "pad"):
            type_dir = os.path.join(model_dir, model_type)
            if not os.path.isdir(type_dir):
                continue
            for f in os.listdir(type_dir):
                if f.endswith(".onnx"):
                    return os.path.join(type_dir, f)

        raise FileNotFoundError(
            "No .onnx model found in {}".format(model_dir)
        )

    def _publish_result(self, task_id, result):
        """Publish task result back to orchestrator via MQTT."""
        topic = "result/{}".format(task_id)
        payload = json.dumps(result)
        self._mqtt_client.publish(topic, payload, qos=1)
        logger.info("Published result to %s", topic)

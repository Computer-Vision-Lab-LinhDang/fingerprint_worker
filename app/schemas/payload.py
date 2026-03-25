
from enum import Enum

# ── Enums ────────────────────────────────────────────────────
class TaskType(str, Enum):
    EMBED = "embed"
    MATCH = "match"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerStatus(str, Enum):
    ONLINE = "online"
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class ModelStatus(str, Enum):
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


# ── Orchestrator → Worker ────────────────────────────────────
class TaskPayload(object):
    def __init__(self, **kwargs):
        self.task_id = kwargs.get("task_id", "")
        self.task_type = kwargs.get("task_type", "")
        self.image_url = kwargs.get("image_url", "")
        self.model_name = kwargs.get("model_name", "default")


class MatchPayload(object):
    def __init__(self, **kwargs):
        self.task_id = kwargs.get("task_id", "")
        self.task_type = kwargs.get("task_type", "match")
        self.query_vector = kwargs.get("query_vector", [])
        self.candidate_vectors = kwargs.get("candidate_vectors", [])
        self.top_k = kwargs.get("top_k", 5)
        self.threshold = kwargs.get("threshold", 0.7)


class ModelUpdatePayload(object):
    """Orchestrator → Worker: command to download a new model."""
    def __init__(self, **kwargs):
        self.model_type = kwargs.get("model_type", "")       # "embedding", "matching", "pad"
        self.model_name = kwargs.get("model_name", "")       # "embedding_v1"
        self.version = kwargs.get("version", "")             # "v1"
        self.download_url = kwargs.get("download_url", "")   # presigned URL
        self.s3_path = kwargs.get("s3_path", "")             # "embedding/embedding_v1/model.onnx"


class ModelStatusPayload(object):
    """Worker → Orchestrator: report model download status."""
    def __init__(self, **kwargs):
        self.worker_id = kwargs.get("worker_id", "")
        self.model_type = kwargs.get("model_type", "")
        self.model_name = kwargs.get("model_name", "")
        self.version = kwargs.get("version", "")
        self.status = kwargs.get("status", "")               # "downloading", "ready", "failed"
        self.error = kwargs.get("error", None)


# ── Heartbeat ────────────────────────────────────────────────
class HeartbeatPayload(object):
    def __init__(self, **kwargs):
        self.worker_id = kwargs.get("worker_id", "")
        self.status = kwargs.get("status", "")
        self.gpu_memory_used_mb = kwargs.get("gpu_memory_used_mb", None)
        self.gpu_memory_total_mb = kwargs.get("gpu_memory_total_mb", None)
        self.current_task_id = kwargs.get("current_task_id", None)
        self.uptime_seconds = kwargs.get("uptime_seconds", None)
        self.loaded_models = kwargs.get("loaded_models", {})  # {"embedding": "embedding_v1", ...}

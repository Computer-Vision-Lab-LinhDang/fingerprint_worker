
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


from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


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
@dataclass
class TaskPayload:
    task_id: str = ""
    task_type: str = ""
    image_url: str = ""
    model_name: str = "default"


@dataclass
class MatchPayload:
    task_id: str = ""
    task_type: str = "match"
    query_vector: list = field(default_factory=list)
    candidate_vectors: list = field(default_factory=list)
    top_k: int = 5
    threshold: float = 0.7

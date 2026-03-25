"""
Centralized constants for the worker.
"""

# ── MQTT Topic Patterns ─────────────────────────────────────
TOPIC_HEARTBEAT = "worker/{worker_id}/heartbeat"
TOPIC_STATUS = "worker/{worker_id}/status"
TOPIC_MESSAGE = "worker/{worker_id}/message"
TOPIC_TASK_EMBED = "task/{worker_id}/embed"
TOPIC_TASK_MATCH = "task/{worker_id}/match"
TOPIC_TASK_MESSAGE = "task/{worker_id}/message"
TOPIC_MODEL_UPDATE = "task/{worker_id}/model/update"
TOPIC_MODEL_STATUS = "worker/{worker_id}/model/status"
TOPIC_RESULT = "result/{task_id}"

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
TOPIC_RESULT = "result/{task_id}"

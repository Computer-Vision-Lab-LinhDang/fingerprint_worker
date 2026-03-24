"""Heartbeat management service — extracted from mqtt/client.py."""

import json
import logging
import threading
import time

from app.schemas.payload import HeartbeatPayload, WorkerStatus

logger = logging.getLogger(__name__)


class HeartbeatService:
    """Manages periodic heartbeat publishing to the orchestrator."""

    def __init__(self, mqtt_client, worker_id, interval=10):
        self._mqtt_client = mqtt_client
        self._worker_id = worker_id
        self._interval = interval
        self._thread = None
        self._stop_event = threading.Event()
        self._start_time = time.time()

    @property
    def uptime(self):
        return time.time() - self._start_time

    def start(self):
        """Start the heartbeat background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="heartbeat",
        )
        self._thread.start()

    def stop(self):
        """Stop the heartbeat background thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def send(self, status=WorkerStatus.IDLE, current_task_id=None):
        """Send a single heartbeat."""
        heartbeat = HeartbeatPayload(
            worker_id=self._worker_id,
            status=status.value if hasattr(status, "value") else status,
            current_task_id=current_task_id,
            uptime_seconds=round(self.uptime, 1),
        )
        topic = "worker/{}/heartbeat".format(self._worker_id)
        return self._mqtt_client.publish(topic, json.dumps(heartbeat.__dict__), qos=1)

    def _loop(self):
        """Background heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                current_task = getattr(self._mqtt_client, "current_task_id", None)
                status = WorkerStatus.BUSY if current_task else WorkerStatus.IDLE
                self.send(status=status, current_task_id=current_task)
            except Exception as exc:
                logger.error("Heartbeat error: %s", exc)
            self._stop_event.wait(timeout=self._interval)

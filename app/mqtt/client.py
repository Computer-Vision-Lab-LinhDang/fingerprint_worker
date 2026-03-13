from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional, Callable, List, Tuple

import paho.mqtt.client as mqtt

from app.core.config import get_settings
from app.schemas.payload import HeartbeatPayload, WorkerStatus

logger = logging.getLogger(__name__)

MessageHandler = Callable[[mqtt.Client, mqtt.MQTTMessage], None]


class MQTTWorkerClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._worker_id = self._settings.WORKER_ID
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._message_handler: Optional[MessageHandler] = None
        self._current_task_id: Optional[str] = None
        self._start_time = time.time()

        # Recent message log (for CLI display)
        self._message_log: List[Tuple[float, str, str]] = []  # (timestamp, topic, payload_preview)
        self._max_log = 50

        # Stats
        self.stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "heartbeats_sent": 0,
            "connect_count": 0,
            "last_connected_at": None,
            "last_disconnected_at": None,
        }

    # ── Properties ───────────────────────────────────────────
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def current_task_id(self) -> Optional[str]:
        return self._current_task_id

    @current_task_id.setter
    def current_task_id(self, value: Optional[str]) -> None:
        self._current_task_id = value

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    @property
    def message_log(self) -> List[Tuple[float, str, str]]:
        return list(self._message_log)

    # ── Connect ──────────────────────────────────────────────
    def connect(self) -> None:
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self._settings.mqtt_client_id,
            protocol=mqtt.MQTTv311,
        )

        if self._settings.MQTT_USERNAME:
            self._client.username_pw_set(
                self._settings.MQTT_USERNAME,
                self._settings.MQTT_PASSWORD,
            )

        # LWT — auto-sent when worker disconnects unexpectedly
        lwt_topic = f"worker/{self._worker_id}/status"
        lwt_payload = json.dumps({"status": "offline", "worker_id": self._worker_id})
        self._client.will_set(lwt_topic, payload=lwt_payload, qos=1, retain=True)

        # Callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        logger.info(
            "Connecting to MQTT broker: %s:%d ...",
            self._settings.MQTT_BROKER_HOST,
            self._settings.MQTT_BROKER_PORT,
        )
        self._client.connect(
            host=self._settings.MQTT_BROKER_HOST,
            port=self._settings.MQTT_BROKER_PORT,
            keepalive=self._settings.MQTT_KEEPALIVE,
        )
        self._client.loop_start()

    def disconnect(self) -> None:
        self._send_heartbeat(status=WorkerStatus.OFFLINE)
        self._stop_event.set()

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)

        if self._client:
            self._client.disconnect()
            self._client.loop_stop()

        self._connected = False
        self.stats["last_disconnected_at"] = time.time()

    # ── Callbacks ────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            self._connected = True
            self.stats["connect_count"] += 1
            self.stats["last_connected_at"] = time.time()

            topics = [
                (f"task/{self._worker_id}/embed", 1),
                (f"task/{self._worker_id}/match", 1),
                (f"task/{self._worker_id}/message", 1),
            ]
            for topic, qos in topics:
                client.subscribe(topic, qos=qos)

            self._send_heartbeat(status=WorkerStatus.ONLINE)
            self._start_heartbeat()
        else:
            logger.error("MQTT connection failed, rc=%s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._connected = False
        self.stats["last_disconnected_at"] = time.time()
        if reason_code != 0:
            logger.warning("MQTT connection lost (rc=%s)", reason_code)

    def _on_message(self, client, userdata, message: mqtt.MQTTMessage) -> None:
        self.stats["messages_received"] += 1

        # Save to log buffer
        try:
            payload_str = message.payload.decode()[:200]
        except Exception:
            payload_str = "<binary>"

        self._message_log.append((time.time(), message.topic, payload_str))
        if len(self._message_log) > self._max_log:
            self._message_log.pop(0)

        if self._message_handler:
            try:
                self._message_handler(client, message)
            except Exception as exc:
                logger.error("Error processing message '%s': %s", message.topic, exc)

    # ── Handler ──────────────────────────────────────────────
    def set_message_handler(self, handler: MessageHandler) -> None:
        self._message_handler = handler

    # ── Publish ──────────────────────────────────────────────
    def publish(self, topic: str, payload: str, qos: int = 1) -> bool:
        if self._client and self._connected:
            result = self._client.publish(topic, payload=payload, qos=qos)
            self.stats["messages_sent"] += 1
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        return False

    def publish_result(self, task_id: str, payload: str) -> bool:
        return self.publish(f"result/{task_id}", payload, qos=1)

    # ── Heartbeat ────────────────────────────────────────────
    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat",
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                status = WorkerStatus.BUSY if self._current_task_id else WorkerStatus.IDLE
                self._send_heartbeat(status=status)
            except Exception as exc:
                logger.error("Heartbeat error: %s", exc)
            self._stop_event.wait(timeout=self._settings.HEARTBEAT_INTERVAL)

    def _send_heartbeat(self, status: WorkerStatus = WorkerStatus.IDLE) -> None:
        heartbeat = HeartbeatPayload(
            worker_id=self._worker_id,
            status=status,
            current_task_id=self._current_task_id,
            uptime_seconds=round(self.uptime, 1),
        )
        topic = f"worker/{self._worker_id}/heartbeat"
        if self.publish(topic, heartbeat.model_dump_json(), qos=1):
            self.stats["heartbeats_sent"] += 1

    # ── Manual heartbeat (for CLI) ───────────────────────────
    def send_manual_heartbeat(self, status: WorkerStatus = WorkerStatus.IDLE) -> bool:
        self._send_heartbeat(status)
        return self._connected


# ── Singleton ────────────────────────────────────────────────
_mqtt_client: Optional[MQTTWorkerClient] = None


def get_mqtt_client() -> MQTTWorkerClient:
    global _mqtt_client
    if _mqtt_client is None:
        _mqtt_client = MQTTWorkerClient()
    return _mqtt_client

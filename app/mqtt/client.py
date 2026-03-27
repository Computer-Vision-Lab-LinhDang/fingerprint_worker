import json
import logging
import threading
import time
from typing import Optional, Callable

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
        self.stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "heartbeats_sent": 0,
            "connect_count": 0,
            "last_connected_at": None,
            "last_disconnected_at": None,
        }

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

    def connect(self) -> None:
        try:
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self._settings.mqtt_client_id,
                protocol=mqtt.MQTTv311,
            )
        except (AttributeError, TypeError):
            self._client = mqtt.Client(
                client_id=self._settings.mqtt_client_id,
                protocol=mqtt.MQTTv311,
            )

        if self._settings.MQTT_USERNAME:
            self._client.username_pw_set(
                self._settings.MQTT_USERNAME,
                self._settings.MQTT_PASSWORD,
            )

        lwt_topic = "worker/{}/status".format(self._worker_id)
        lwt_payload = json.dumps({"status": "offline", "worker_id": self._worker_id})
        self._client.will_set(lwt_topic, payload=lwt_payload, qos=1, retain=True)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        logger.info("Connecting to MQTT %s:%d ...", self._settings.MQTT_BROKER_HOST, self._settings.MQTT_BROKER_PORT)
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

    def _on_connect(self, client, userdata, flags, *args) -> None:
        rc = args[0]
        if rc == 0:
            self._connected = True
            self.stats["connect_count"] += 1
            self.stats["last_connected_at"] = time.time()

            topics = [
                ("task/{}/embed".format(self._worker_id), 1),
                ("task/{}/match".format(self._worker_id), 1),
                ("task/{}/message".format(self._worker_id), 1),
                ("task/{}/model/update".format(self._worker_id), 1),
            ]
            for topic, qos in topics:
                client.subscribe(topic, qos=qos)

            self._send_heartbeat(status=WorkerStatus.ONLINE)
            self._start_heartbeat()
        else:
            logger.error("MQTT connection failed, rc=%s", rc)

    def _on_disconnect(self, client, userdata, *args) -> None:
        self._connected = False
        self.stats["last_disconnected_at"] = time.time()
        rc = args[-2] if len(args) >= 2 else args[0]
        if rc != 0:
            logger.warning("MQTT connection lost (rc=%s)", rc)

    def _on_message(self, client, userdata, message: mqtt.MQTTMessage) -> None:
        self.stats["messages_received"] += 1
        if self._message_handler:
            try:
                self._message_handler(client, message)
            except Exception as exc:
                logger.error("Error processing message '%s': %s", message.topic, exc)

    def set_message_handler(self, handler: MessageHandler) -> None:
        self._message_handler = handler

    def publish(self, topic: str, payload: str, qos: int = 1) -> bool:
        if self._client and self._connected:
            result = self._client.publish(topic, payload=payload, qos=qos)
            self.stats["messages_sent"] += 1
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        return False

    def publish_result(self, task_id: str, payload: str) -> bool:
        return self.publish("result/{}".format(task_id), payload, qos=1)

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

    def _send_heartbeat(self, status=WorkerStatus.IDLE):
        try:
            from app.services.model_service import get_model_service
            loaded_models = get_model_service().loaded_models
        except Exception:
            loaded_models = {}

        heartbeat = HeartbeatPayload(
            worker_id=self._worker_id,
            status=status.value if hasattr(status, 'value') else status,
            current_task_id=self._current_task_id,
            uptime_seconds=round(self.uptime, 1),
            loaded_models=loaded_models,
        )
        topic = "worker/{}/heartbeat".format(self._worker_id)
        if self.publish(topic, json.dumps(heartbeat.__dict__), qos=1):
            self.stats["heartbeats_sent"] += 1

    def send_manual_heartbeat(self, status: WorkerStatus = WorkerStatus.IDLE) -> bool:
        self._send_heartbeat(status)
        return self._connected


_mqtt_client: Optional[MQTTWorkerClient] = None


def get_mqtt_client() -> MQTTWorkerClient:
    global _mqtt_client
    if _mqtt_client is None:
        _mqtt_client = MQTTWorkerClient()
    return _mqtt_client

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        # ── Worker Identity ─────────────────────────────────────
        self.WORKER_ID = os.getenv("WORKER_ID", "jetson-nano-01")

        # ── MQTT / Mosquitto ────────────────────────────────────
        self.MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
        self.MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
        self.MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
        self.MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
        self.MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "")
        self.MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "60"))
        self.MQTT_RECONNECT_DELAY = int(os.getenv("MQTT_RECONNECT_DELAY", "5"))

        # ── Heartbeat ───────────────────────────────────────────
        self.HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))

    @property
    def mqtt_client_id(self):
        return self.MQTT_CLIENT_ID or f"worker-{self.WORKER_ID}"


_settings = None


def get_settings():
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

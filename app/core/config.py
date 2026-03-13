from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Worker Identity ─────────────────────────────────────
    WORKER_ID: str = "jetson-nano-01"

    # ── MQTT / Mosquitto ────────────────────────────────────
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""
    MQTT_CLIENT_ID: str = ""
    MQTT_KEEPALIVE: int = 60
    MQTT_RECONNECT_DELAY: int = 5

    # ── Heartbeat ───────────────────────────────────────────
    HEARTBEAT_INTERVAL: int = 10

    @property
    def mqtt_client_id(self) -> str:
        return self.MQTT_CLIENT_ID or f"worker-{self.WORKER_ID}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

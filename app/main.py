
import logging
import time

from app.core.config import get_settings
from app.mqtt.client import get_mqtt_client
from app.mqtt.handlers import create_message_handler
from app.cli.menu import run_cli
from app.cli.colors import C, clear_screen

# ── Logging config ───────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    settings = get_settings()
    mqtt_client = get_mqtt_client()

    clear_screen()
    print("\n  {}{}🚀 Fingerprint Worker starting...{}".format(C.CYAN, C.BOLD, C.RESET))
    print("  {}Worker ID : {}{}".format(C.DIM, settings.WORKER_ID, C.RESET))
    print("  {}Broker    : {}:{}{}".format(C.DIM, settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, C.RESET))
    print()

    # 1. Set message handler
    handler = create_message_handler(mqtt_client)
    mqtt_client.set_message_handler(handler)

    # 2. Connect to MQTT broker
    try:
        print("  {}Connecting to MQTT broker...{}".format(C.YELLOW, C.RESET))
        mqtt_client.connect()
        time.sleep(2)  # wait for on_connect callback

        if mqtt_client.is_connected:
            print("  {}✓ MQTT connected successfully!{}".format(C.GREEN, C.RESET))
        else:
            print("  {}✗ Could not connect to MQTT{}".format(C.RED, C.RESET))
            print("  {}Check MQTT_BROKER_HOST in .env{}".format(C.DIM, C.RESET))
            print("  {}You can still enter CLI to try reconnect{}".format(C.DIM, C.RESET))

    except Exception as exc:
        print("  {}✗ Connection error: {}{}".format(C.RED, exc, C.RESET))
        print("  {}You can still enter CLI to try reconnect{}".format(C.DIM, C.RESET))

    print()
    input("  {}Press Enter to open main menu...{}".format(C.DIM, C.RESET))

    # 3. Run interactive CLI
    try:
        run_cli()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  {}Shutting down worker...{}".format(C.YELLOW, C.RESET))
        try:
            mqtt_client.disconnect()
        except Exception:
            pass
        print("  {}👋 Worker stopped.{}\n".format(C.GREEN, C.RESET))


if __name__ == "__main__":
    main()

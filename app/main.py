
import logging
import time

from app.core.config import get_settings
from app.mqtt.client import get_mqtt_client
from app.mqtt.handlers import create_message_handler
from app.cli import run_cli, clear_screen, C

# ── Logging config ───────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    mqtt_client = get_mqtt_client()

    clear_screen()
    print(f"\n  {C.CYAN}{C.BOLD}🚀 Fingerprint Worker starting...{C.RESET}")
    print(f"  {C.DIM}Worker ID : {settings.WORKER_ID}{C.RESET}")
    print(f"  {C.DIM}Broker    : {settings.MQTT_BROKER_HOST}:{settings.MQTT_BROKER_PORT}{C.RESET}")
    print()

    # 1. Set message handler
    handler = create_message_handler(mqtt_client)
    mqtt_client.set_message_handler(handler)

    # 2. Connect to MQTT broker
    try:
        print(f"  {C.YELLOW}Connecting to MQTT broker...{C.RESET}")
        mqtt_client.connect()
        time.sleep(2)  # wait for on_connect callback

        if mqtt_client.is_connected:
            print(f"  {C.GREEN}✓ MQTT connected successfully!{C.RESET}")
        else:
            print(f"  {C.RED}✗ Could not connect to MQTT{C.RESET}")
            print(f"  {C.DIM}Check MQTT_BROKER_HOST in .env{C.RESET}")
            print(f"  {C.DIM}You can still enter CLI to try reconnect{C.RESET}")

    except Exception as exc:
        print(f"  {C.RED}✗ Connection error: {exc}{C.RESET}")
        print(f"  {C.DIM}You can still enter CLI to try reconnect{C.RESET}")

    print()
    input(f"  {C.DIM}Press Enter to open main menu...{C.RESET}")

    # 3. Run interactive CLI
    try:
        run_cli()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n  {C.YELLOW}Shutting down worker...{C.RESET}")
        try:
            mqtt_client.disconnect()
        except Exception:
            pass
        print(f"  {C.GREEN}👋 Worker stopped.{C.RESET}\n")


if __name__ == "__main__":
    main()

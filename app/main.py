
import logging
import sys
import time

from app.core.config import get_settings
from app.mqtt.client import get_mqtt_client
from app.mqtt.handlers import create_message_handler

# ── Logging config ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    cli_mode = "--cli" in sys.argv

    settings = get_settings()
    mqtt_client = get_mqtt_client()

    logger.info("Worker ID : %s", settings.WORKER_ID)
    logger.info("Broker    : %s:%s", settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT)

    # 1. Set message handler
    handler = create_message_handler(mqtt_client)
    mqtt_client.set_message_handler(handler)

    # 2. Connect to MQTT broker
    try:
        logger.info("Connecting to MQTT broker...")
        mqtt_client.connect()
        time.sleep(2)

        if mqtt_client.is_connected:
            logger.info("MQTT connected successfully!")
        else:
            logger.error("Could not connect to MQTT. Check MQTT_BROKER_HOST in .env")

    except Exception as exc:
        logger.error("Connection error: %s", exc)

    # 3. Run in selected mode
    if cli_mode:
        _run_cli(mqtt_client)
    else:
        _run_daemon(mqtt_client)


def _run_cli(mqtt_client):
    """Interactive CLI mode."""
    from app.cli.colors import C, clear_screen
    from app.cli.menu import run_cli

    clear_screen()
    print("\n  {}{}FINGERPRINT WORKER (CLI mode){}".format(C.CYAN, C.BOLD, C.RESET))
    print()
    run_cli()

    logger.info("Shutting down worker...")
    try:
        mqtt_client.disconnect()
    except Exception:
        pass
    logger.info("Worker stopped.")


def _run_daemon(mqtt_client):
    """Daemon mode — log output only, no interactive CLI."""
    logger.info("Running in daemon mode (use --cli for interactive menu)")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down worker...")
        try:
            mqtt_client.disconnect()
        except Exception:
            pass
        logger.info("Worker stopped.")


if __name__ == "__main__":
    main()

"""CLI menu display and main loop."""

from app.core.config import get_settings
from app.mqtt.client import get_mqtt_client
from app.cli.colors import C, clear_screen, fmt_uptime
from app.cli.commands import (
    show_connection_status,
    send_heartbeat,
    show_stats,
    show_config,
    reconnect,
    show_loaded_models,
    test_model_inference,
)


# ── Banner ───────────────────────────────────────────────────
def print_banner():
    print("\n{}{}".format(C.CYAN, C.BOLD))
    print("╔══════════════════════════════════════════════════╗")
    print("║        🖐  FINGERPRINT WORKER — CLI              ║")
    print("╚══════════════════════════════════════════════════╝{}".format(C.RESET))
    print()


# ── Main menu ────────────────────────────────────────────────
def print_menu():
    client = get_mqtt_client()
    settings = get_settings()

    if client.is_connected:
        status = "{}● CONNECTED{}".format(C.GREEN, C.RESET)
    else:
        status = "{}● DISCONNECTED{}".format(C.RED, C.RESET)

    print("  {}Worker:{} {}{}{}  │  {}  │  {}Uptime:{} {}".format(
        C.DIM, C.RESET, C.BOLD, settings.WORKER_ID, C.RESET,
        status, C.DIM, C.RESET, fmt_uptime(client.uptime),
    ))
    print("  {}Broker:{} {}:{}".format(C.DIM, C.RESET, settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT))
    print()
    print("  {}{}{}".format(C.YELLOW, "-" * 48, C.RESET))
    print("  {}[1]{}  📊  Connection Status".format(C.BOLD, C.RESET))
    print("  {}[2]{}  💓  Send Heartbeat".format(C.BOLD, C.RESET))
    print("  {}[3]{}  📈  Statistics".format(C.BOLD, C.RESET))
    print("  {}[4]{}  ⚙️   Current Configuration".format(C.BOLD, C.RESET))
    print("  {}[5]{}  🔄  Reconnect MQTT".format(C.BOLD, C.RESET))
    print("  {}[6]{}  🧹  Clear Screen".format(C.BOLD, C.RESET))
    print("  {}[7]{}  🧠  Loaded Models".format(C.BOLD, C.RESET))
    print("  {}[8]{}  🚀  Test Model Inference".format(C.BOLD, C.RESET))
    print("  {}[0]{}  🚪  Exit".format(C.BOLD, C.RESET))
    print("  {}{}{}".format(C.YELLOW, "-" * 48, C.RESET))


# ── Main loop ────────────────────────────────────────────────
def run_cli():
    clear_screen()
    print_banner()

    actions = {
        "1": show_connection_status,
        "2": send_heartbeat,
        "3": show_stats,
        "4": show_config,
        "5": reconnect,
        "6": lambda: (clear_screen(), print_banner()),
        "7": show_loaded_models,
        "8": test_model_inference,
    }

    while True:
        print_menu()
        try:
            choice = input("\n  {}{}▸ Select [0-8]: {}".format(C.YELLOW, C.BOLD, C.RESET)).strip()
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "0":
            print("\n  {}Exiting...{}".format(C.DIM, C.RESET))
            break

        action = actions.get(choice)
        if action:
            action()
            input("\n  {}Press Enter to continue...{}".format(C.DIM, C.RESET))
            clear_screen()
            print_banner()
        else:
            print("  {}Invalid choice!{}".format(C.RED, C.RESET))

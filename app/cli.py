
import json
import os
import time
import uuid
from datetime import datetime

from app.core.config import get_settings
from app.mqtt.client import get_mqtt_client
from app.schemas.payload import WorkerStatus


# ── ANSI Colors ──────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;236m"


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def fmt_time(ts: float) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_uptime(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ── Banner ───────────────────────────────────────────────────
def print_banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════╗
║        🖐  FINGERPRINT WORKER — CLI              ║
╚══════════════════════════════════════════════════╝{C.RESET}
""")


# ── Main menu ────────────────────────────────────────────────
def print_menu():
    client = get_mqtt_client()
    settings = get_settings()

    if client.is_connected:
        status = f"{C.GREEN}● CONNECTED{C.RESET}"
    else:
        status = f"{C.RED}● DISCONNECTED{C.RESET}"

    print(f"  {C.DIM}Worker:{C.RESET} {C.BOLD}{settings.WORKER_ID}{C.RESET}  │  {status}  │  {C.DIM}Uptime:{C.RESET} {fmt_uptime(client.uptime)}")
    print(f"  {C.DIM}Broker:{C.RESET} {settings.MQTT_BROKER_HOST}:{settings.MQTT_BROKER_PORT}")
    print()
    print(f"  {C.YELLOW}{'─' * 48}{C.RESET}")
    print(f"  {C.BOLD}[1]{C.RESET}  📊  Connection Status")
    print(f"  {C.BOLD}[2]{C.RESET}  💓  Send Heartbeat")
    print(f"  {C.BOLD}[3]{C.RESET}  📨  Send Test Message")
    print(f"  {C.BOLD}[4]{C.RESET}  ✉️   Send Message to Orchestrator")
    print(f"  {C.BOLD}[5]{C.RESET}  📋  Recent Message Log")
    print(f"  {C.BOLD}[6]{C.RESET}  📈  Statistics")
    print(f"  {C.BOLD}[7]{C.RESET}  ⚙️   Current Configuration")
    print(f"  {C.BOLD}[8]{C.RESET}  🔄  Reconnect MQTT")
    print(f"  {C.BOLD}[9]{C.RESET}  🧹  Clear Screen")
    print(f"  {C.BOLD}[0]{C.RESET}  🚪  Exit")
    print(f"  {C.YELLOW}{'─' * 48}{C.RESET}")


# ── [1] Connection Status ───────────────────────────────────
def show_connection_status():
    client = get_mqtt_client()
    settings = get_settings()

    print(f"\n  {C.CYAN}{C.BOLD}═══ CONNECTION STATUS ═══{C.RESET}\n")

    if client.is_connected:
        print(f"  MQTT       : {C.GREEN}● Connected{C.RESET}")
    else:
        print(f"  MQTT       : {C.RED}● Disconnected{C.RESET}")

    print(f"  Broker     : {settings.MQTT_BROKER_HOST}:{settings.MQTT_BROKER_PORT}")
    print(f"  Client ID  : {settings.mqtt_client_id}")
    print(f"  Worker ID  : {settings.WORKER_ID}")
    print(f"  Keepalive  : {settings.MQTT_KEEPALIVE}s")
    print(f"  Heartbeat  : every {settings.HEARTBEAT_INTERVAL}s")
    print()
    print(f"  {C.DIM}Subscribe topics:{C.RESET}")
    print(f"    → task/{settings.WORKER_ID}/embed")
    print(f"    → task/{settings.WORKER_ID}/match")
    print()
    print(f"  {C.DIM}Publish topics:{C.RESET}")
    print(f"    → worker/{settings.WORKER_ID}/heartbeat")
    print(f"    → worker/{settings.WORKER_ID}/status  (LWT)")
    print(f"    → result/{{task_id}}")
    print()


# ── [2] Send Heartbeat ──────────────────────────────────────
def send_heartbeat():
    client = get_mqtt_client()

    print(f"\n  {C.CYAN}{C.BOLD}═══ SEND HEARTBEAT ═══{C.RESET}\n")

    if not client.is_connected:
        print(f"  {C.RED}✗ Not connected to MQTT!{C.RESET}")
        return

    print(f"  Select status:")
    print(f"    [1] IDLE   (available)")
    print(f"    [2] BUSY   (processing)")
    print(f"    [3] ONLINE (just online)")

    choice = input(f"\n  {C.YELLOW}▸ Choose (1-3): {C.RESET}").strip()
    status_map = {"1": WorkerStatus.IDLE, "2": WorkerStatus.BUSY, "3": WorkerStatus.ONLINE}
    status = status_map.get(choice, WorkerStatus.IDLE)

    ok = client.send_manual_heartbeat(status)
    if ok:
        print(f"  {C.GREEN}✓ Heartbeat sent: {status.value}{C.RESET}")
    else:
        print(f"  {C.RED}✗ Failed to send heartbeat{C.RESET}")


# ── [3] Send Test Message ───────────────────────────────────
def send_test_message():
    client = get_mqtt_client()
    settings = get_settings()

    print(f"\n  {C.CYAN}{C.BOLD}═══ SEND TEST MESSAGE ═══{C.RESET}\n")

    if not client.is_connected:
        print(f"  {C.RED}✗ Not connected to MQTT!{C.RESET}")
        return

    print(f"  Select message type:")
    print(f"    [1] Publish to custom topic")
    print(f"    [2] Send fake result (test orchestrator)")
    print()

    choice = input(f"  {C.YELLOW}▸ Choose (1-2): {C.RESET}").strip()

    if choice == "1":
        topic = input(f"  {C.YELLOW}▸ Topic: {C.RESET}").strip()
        message = input(f"  {C.YELLOW}▸ Message: {C.RESET}").strip()
        if topic and message:
            ok = client.publish(topic, message)
            if ok:
                print(f"  {C.GREEN}✓ Sent → {topic}{C.RESET}")
            else:
                print(f"  {C.RED}✗ Send failed{C.RESET}")
        else:
            print(f"  {C.RED}✗ Topic and message cannot be empty{C.RESET}")

    elif choice == "2":
        task_id = input(f"  {C.YELLOW}▸ Task ID (or Enter for random): {C.RESET}").strip()
        if not task_id:
            import uuid
            task_id = str(uuid.uuid4())

        fake_result = {
            "task_id": task_id,
            "worker_id": settings.WORKER_ID,
            "status": "completed",
            "result": {"vector": [0.1, 0.2, 0.3], "model_name": "test"},
            "processing_time_ms": 42.0,
        }
        payload = json.dumps(fake_result)
        ok = client.publish_result(task_id, payload)
        if ok:
            print(f"  {C.GREEN}✓ Fake result sent → result/{task_id}{C.RESET}")
        else:
            print(f"  {C.RED}✗ Send failed{C.RESET}")

    else:
        print(f"  {C.DIM}Skipped.{C.RESET}")


# ── [4] Message Log ─────────────────────────────────────────
def show_message_log():
    client = get_mqtt_client()

    print(f"\n  {C.CYAN}{C.BOLD}═══ RECENT MESSAGE LOG ═══{C.RESET}\n")

    logs = client.message_log
    if not logs:
        print(f"  {C.DIM}No messages yet.{C.RESET}")
        return

    print(f"  {C.DIM}{'Time':<12} {'Topic':<35} {'Payload'}{C.RESET}")
    print(f"  {C.DIM}{'─' * 70}{C.RESET}")

    for ts, topic, payload_preview in logs[-15:]:
        t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        topic_short = topic if len(topic) <= 33 else "..." + topic[-30:]
        payload_short = payload_preview[:40] + "..." if len(payload_preview) > 40 else payload_preview
        print(f"  {C.WHITE}{t:<12}{C.RESET} {C.BLUE}{topic_short:<35}{C.RESET} {payload_short}")

    print(f"\n  {C.DIM}Showing {min(15, len(logs))}/{len(logs)} most recent messages{C.RESET}")


# ── [5] Statistics ───────────────────────────────────────────
def show_stats():
    client = get_mqtt_client()
    stats = client.stats

    print(f"\n  {C.CYAN}{C.BOLD}═══ STATISTICS ═══{C.RESET}\n")
    print(f"  Uptime              : {fmt_uptime(client.uptime)}")
    print(f"  Connection count    : {stats['connect_count']}")
    print(f"  Last connected at   : {fmt_time(stats['last_connected_at'])}")
    print(f"  Last disconnected at: {fmt_time(stats['last_disconnected_at'])}")
    print()
    print(f"  Messages received   : {C.GREEN}{stats['messages_received']}{C.RESET}")
    print(f"  Messages sent       : {C.BLUE}{stats['messages_sent']}{C.RESET}")
    print(f"  Heartbeats sent     : {C.MAGENTA}{stats['heartbeats_sent']}{C.RESET}")
    print()


# ── [6] Configuration ───────────────────────────────────────
def show_config():
    settings = get_settings()

    print(f"\n  {C.CYAN}{C.BOLD}═══ CURRENT CONFIGURATION (.env) ═══{C.RESET}\n")
    print(f"  WORKER_ID          = {C.BOLD}{settings.WORKER_ID}{C.RESET}")
    print(f"  MQTT_BROKER_HOST   = {settings.MQTT_BROKER_HOST}")
    print(f"  MQTT_BROKER_PORT   = {settings.MQTT_BROKER_PORT}")
    print(f"  MQTT_USERNAME      = {settings.MQTT_USERNAME or '(empty)'}")
    print(f"  MQTT_PASSWORD      = {'***' if settings.MQTT_PASSWORD else '(empty)'}")
    print(f"  MQTT_CLIENT_ID     = {settings.mqtt_client_id}")
    print(f"  MQTT_KEEPALIVE     = {settings.MQTT_KEEPALIVE}s")
    print(f"  MQTT_RECONNECT     = {settings.MQTT_RECONNECT_DELAY}s")
    print(f"  HEARTBEAT_INTERVAL = {settings.HEARTBEAT_INTERVAL}s")
    print()


# ── [4] Send Message to Orchestrator ────────────────────────
def send_message_to_orchestrator():
    client = get_mqtt_client()
    settings = get_settings()

    print(f"\n  {C.CYAN}{C.BOLD}═══ SEND MESSAGE TO ORCHESTRATOR ═══{C.RESET}\n")

    if not client.is_connected:
        print(f"  {C.RED}✗ Not connected to MQTT!{C.RESET}")
        return

    message_text = input(f"  {C.YELLOW}▸ Enter message: {C.RESET}").strip()
    if not message_text:
        print(f"  {C.RED}✗ Message cannot be empty{C.RESET}")
        return

    payload = json.dumps({
        "worker_id": settings.WORKER_ID,
        "message_id": str(uuid.uuid4()),
        "content": message_text,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    topic = f"worker/{settings.WORKER_ID}/message"
    ok = client.publish(topic, payload, qos=1)
    if ok:
        print(f"  {C.GREEN}✓ Message sent → {topic}{C.RESET}")
        print(f"  {C.DIM}Content: {message_text}{C.RESET}")
    else:
        print(f"  {C.RED}✗ Send failed{C.RESET}")


# ── [8] Reconnect ───────────────────────────────────────────
def reconnect():
    client = get_mqtt_client()

    print(f"\n  {C.CYAN}{C.BOLD}═══ RECONNECT ═══{C.RESET}\n")

    if client.is_connected:
        print(f"  {C.YELLOW}Disconnecting...{C.RESET}")
        client.disconnect()
        time.sleep(1)

    print(f"  {C.YELLOW}Reconnecting...{C.RESET}")
    try:
        client.connect()
        time.sleep(2)  # wait for on_connect callback
        if client.is_connected:
            print(f"  {C.GREEN}✓ Reconnected successfully!{C.RESET}")
        else:
            print(f"  {C.RED}✗ Reconnection failed{C.RESET}")
    except Exception as exc:
        print(f"  {C.RED}✗ Error: {exc}{C.RESET}")


# ── Main loop ────────────────────────────────────────────────
def run_cli():
    clear_screen()
    print_banner()

    actions = {
        "1": show_connection_status,
        "2": send_heartbeat,
        "3": send_test_message,
        "4": send_message_to_orchestrator,
        "5": show_message_log,
        "6": show_stats,
        "7": show_config,
        "8": reconnect,
        "9": lambda: (clear_screen(), print_banner()),
    }

    while True:
        print_menu()
        try:
            choice = input(f"\n  {C.YELLOW}{C.BOLD}▸ Select [0-9]: {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "0":
            print(f"\n  {C.DIM}Exiting...{C.RESET}")
            break

        action = actions.get(choice)
        if action:
            action()
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
            clear_screen()
            print_banner()
        else:
            print(f"  {C.RED}Invalid choice!{C.RESET}")

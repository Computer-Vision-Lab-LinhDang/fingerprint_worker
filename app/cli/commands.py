"""Individual CLI commands for the worker."""

import json
import time
import uuid
from datetime import datetime

from app.core.config import get_settings
from app.mqtt.client import get_mqtt_client
from app.schemas.payload import WorkerStatus
from app.cli.colors import C, fmt_time, fmt_uptime


# ── [1] Connection Status ───────────────────────────────────
def show_connection_status():
    client = get_mqtt_client()
    settings = get_settings()

    print("\n  {}{}=== CONNECTION STATUS ==={}\n".format(C.CYAN, C.BOLD, C.RESET))

    if client.is_connected:
        print("  MQTT       : {}● Connected{}".format(C.GREEN, C.RESET))
    else:
        print("  MQTT       : {}● Disconnected{}".format(C.RED, C.RESET))

    print("  Broker     : {}:{}".format(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT))
    print("  Client ID  : {}".format(settings.mqtt_client_id))
    print("  Worker ID  : {}".format(settings.WORKER_ID))
    print("  Keepalive  : {}s".format(settings.MQTT_KEEPALIVE))
    print("  Heartbeat  : every {}s".format(settings.HEARTBEAT_INTERVAL))
    print()
    print("  {}Subscribe topics:{}".format(C.DIM, C.RESET))
    print("    -> task/{}/embed".format(settings.WORKER_ID))
    print("    -> task/{}/match".format(settings.WORKER_ID))
    print()
    print("  {}Publish topics:{}".format(C.DIM, C.RESET))
    print("    -> worker/{}/heartbeat".format(settings.WORKER_ID))
    print("    -> worker/{}/status  (LWT)".format(settings.WORKER_ID))
    print("    -> result/{{task_id}}")
    print()


# ── [2] Send Heartbeat ──────────────────────────────────────
def send_heartbeat():
    client = get_mqtt_client()

    print("\n  {}{}=== SEND HEARTBEAT ==={}\n".format(C.CYAN, C.BOLD, C.RESET))

    if not client.is_connected:
        print("  {}✗ Not connected to MQTT!{}".format(C.RED, C.RESET))
        return

    print("  Select status:")
    print("    [1] IDLE   (available)")
    print("    [2] BUSY   (processing)")
    print("    [3] ONLINE (just online)")

    choice = input("\n  {}▸ Choose (1-3): {}".format(C.YELLOW, C.RESET)).strip()
    status_map = {"1": WorkerStatus.IDLE, "2": WorkerStatus.BUSY, "3": WorkerStatus.ONLINE}
    status = status_map.get(choice, WorkerStatus.IDLE)

    ok = client.send_manual_heartbeat(status)
    if ok:
        print("  {}✓ Heartbeat sent: {}{}".format(C.GREEN, status.value, C.RESET))
    else:
        print("  {}✗ Failed to send heartbeat{}".format(C.RED, C.RESET))


# ── [3] Send Test Message ───────────────────────────────────
def send_test_message():
    client = get_mqtt_client()
    settings = get_settings()

    print("\n  {}{}=== SEND TEST MESSAGE ==={}\n".format(C.CYAN, C.BOLD, C.RESET))

    if not client.is_connected:
        print("  {}✗ Not connected to MQTT!{}".format(C.RED, C.RESET))
        return

    print("  Select message type:")
    print("    [1] Publish to custom topic")
    print("    [2] Send fake result (test orchestrator)")
    print()

    choice = input("  {}▸ Choose (1-2): {}".format(C.YELLOW, C.RESET)).strip()

    if choice == "1":
        topic = input("  {}▸ Topic: {}".format(C.YELLOW, C.RESET)).strip()
        message = input("  {}▸ Message: {}".format(C.YELLOW, C.RESET)).strip()
        if topic and message:
            ok = client.publish(topic, message)
            if ok:
                print("  {}✓ Sent → {}{}".format(C.GREEN, topic, C.RESET))
            else:
                print("  {}✗ Send failed{}".format(C.RED, C.RESET))
        else:
            print("  {}✗ Topic and message cannot be empty{}".format(C.RED, C.RESET))

    elif choice == "2":
        task_id = input("  {}▸ Task ID (or Enter for random): {}".format(C.YELLOW, C.RESET)).strip()
        if not task_id:
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
            print("  {}✓ Fake result sent → result/{}{}".format(C.GREEN, task_id, C.RESET))
        else:
            print("  {}✗ Send failed{}".format(C.RED, C.RESET))

    else:
        print("  {}Skipped.{}".format(C.DIM, C.RESET))


# ── [4] Send Message to Orchestrator ────────────────────────
def send_message_to_orchestrator():
    client = get_mqtt_client()
    settings = get_settings()

    print("\n  {}{}=== SEND MESSAGE TO ORCHESTRATOR ==={}\n".format(C.CYAN, C.BOLD, C.RESET))

    if not client.is_connected:
        print("  {}✗ Not connected to MQTT!{}".format(C.RED, C.RESET))
        return

    message_text = input("  {}▸ Enter message: {}".format(C.YELLOW, C.RESET)).strip()
    if not message_text:
        print("  {}✗ Message cannot be empty{}".format(C.RED, C.RESET))
        return

    payload = json.dumps({
        "worker_id": settings.WORKER_ID,
        "message_id": str(uuid.uuid4()),
        "content": message_text,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    topic = "worker/{}/message".format(settings.WORKER_ID)
    ok = client.publish(topic, payload, qos=1)
    if ok:
        print("  {}✓ Message sent → {}{}".format(C.GREEN, topic, C.RESET))
        print("  {}Content: {}{}".format(C.DIM, message_text, C.RESET))
    else:
        print("  {}✗ Send failed{}".format(C.RED, C.RESET))


# ── [5] Message Log ─────────────────────────────────────────
def show_message_log():
    client = get_mqtt_client()

    print("\n  {}{}=== RECENT MESSAGE LOG ==={}\n".format(C.CYAN, C.BOLD, C.RESET))

    logs = client.message_log
    if not logs:
        print("  {}No messages yet.{}".format(C.DIM, C.RESET))
        return

    print("  {}{:<12} {:<35} {}{}".format(C.DIM, "Time", "Topic", "Payload", C.RESET))
    print("  {}{}{}".format(C.DIM, "-" * 70, C.RESET))

    for ts, topic, payload_preview in logs[-15:]:
        t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        topic_short = topic if len(topic) <= 33 else "..." + topic[-30:]
        payload_short = payload_preview[:40] + "..." if len(payload_preview) > 40 else payload_preview
        print("  {}{:<12}{} {}{:<35}{} {}".format(C.WHITE, t, C.RESET, C.BLUE, topic_short, C.RESET, payload_short))

    print("\n  {}Showing {}/{} most recent messages{}".format(C.DIM, min(15, len(logs)), len(logs), C.RESET))


# ── [6] Statistics ───────────────────────────────────────────
def show_stats():
    client = get_mqtt_client()
    stats = client.stats

    print("\n  {}{}=== STATISTICS ==={}\n".format(C.CYAN, C.BOLD, C.RESET))
    print("  Uptime              : {}".format(fmt_uptime(client.uptime)))
    print("  Connection count    : {}".format(stats["connect_count"]))
    print("  Last connected at   : {}".format(fmt_time(stats["last_connected_at"])))
    print("  Last disconnected at: {}".format(fmt_time(stats["last_disconnected_at"])))
    print()
    print("  Messages received   : {}{}{}".format(C.GREEN, stats["messages_received"], C.RESET))
    print("  Messages sent       : {}{}{}".format(C.BLUE, stats["messages_sent"], C.RESET))
    print("  Heartbeats sent     : {}{}{}".format(C.MAGENTA, stats["heartbeats_sent"], C.RESET))
    print()


# ── [7] Configuration ───────────────────────────────────────
def show_config():
    settings = get_settings()

    print("\n  {}{}=== CURRENT CONFIGURATION (.env) ==={}\n".format(C.CYAN, C.BOLD, C.RESET))
    print("  WORKER_ID          = {}{}{}".format(C.BOLD, settings.WORKER_ID, C.RESET))
    print("  MQTT_BROKER_HOST   = {}".format(settings.MQTT_BROKER_HOST))
    print("  MQTT_BROKER_PORT   = {}".format(settings.MQTT_BROKER_PORT))
    print("  MQTT_USERNAME      = {}".format(settings.MQTT_USERNAME or "(empty)"))
    print("  MQTT_PASSWORD      = {}".format("***" if settings.MQTT_PASSWORD else "(empty)"))
    print("  MQTT_CLIENT_ID     = {}".format(settings.mqtt_client_id))
    print("  MQTT_KEEPALIVE     = {}s".format(settings.MQTT_KEEPALIVE))
    print("  MQTT_RECONNECT     = {}s".format(settings.MQTT_RECONNECT_DELAY))
    print("  HEARTBEAT_INTERVAL = {}s".format(settings.HEARTBEAT_INTERVAL))
    print()


# ── [8] Reconnect ───────────────────────────────────────────
def reconnect():
    client = get_mqtt_client()

    print("\n  {}{}=== RECONNECT ==={}\n".format(C.CYAN, C.BOLD, C.RESET))

    if client.is_connected:
        print("  {}Disconnecting...{}".format(C.YELLOW, C.RESET))
        client.disconnect()
        time.sleep(1)

    print("  {}Reconnecting...{}".format(C.YELLOW, C.RESET))
    try:
        client.connect()
        time.sleep(2)  # wait for on_connect callback
        if client.is_connected:
            print("  {}✓ Reconnected successfully!{}".format(C.GREEN, C.RESET))
        else:
            print("  {}✗ Reconnection failed{}".format(C.RED, C.RESET))
    except Exception as exc:
        print("  {}✗ Error: {}{}".format(C.RED, exc, C.RESET))


# ── [10] Loaded Models ──────────────────────────────────────
def show_loaded_models():
    from app.services.model_service import get_model_service

    print("\n  {}{}=== LOADED MODELS ==={}".format(C.CYAN, C.BOLD, C.RESET))

    svc = get_model_service()

    # 1. Show live model state (what heartbeat reports)
    loaded = svc.loaded_models
    print("\n  {}Active models (reported in heartbeat):{}".format(C.DIM, C.RESET))
    if loaded:
        for mtype, mname in loaded.items():
            print("    {}●{} {}: {}{}{}".format(C.GREEN, C.RESET, mtype, C.MAGENTA, mname, C.RESET))
    else:
        print("    {}(none — no models loaded yet){}".format(C.DIM, C.RESET))

    # 2. Show all model files on disk
    local = svc.list_local_models()
    print("\n  {}Model files on disk:{}".format(C.DIM, C.RESET))
    if local:
        for m in local:
            size_mb = m["size_bytes"] / (1024 * 1024)
            print("    📦 {}/{} {}({:.1f} MB){} → {}".format(
                m["model_type"], m["model_name"],
                C.DIM, size_mb, C.RESET,
                m["path"],
            ))
    else:
        print("    {}(no .onnx files found in models/ directory){}".format(C.DIM, C.RESET))

    print()


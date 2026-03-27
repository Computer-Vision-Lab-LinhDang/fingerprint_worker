"""Individual CLI commands for the worker."""

import time



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


# ── [3] Statistics ───────────────────────────────────────────
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


# ── [11] Test Model Inference ───────────────────────────────
def test_model_inference():
    import os
    import glob
    from app.services.model_service import get_model_service

    print("\n  {}{}=== TEST MODEL INFERENCE ==={}".format(C.CYAN, C.BOLD, C.RESET))

    svc = get_model_service()
    loaded = svc.loaded_models
    local = svc.list_local_models()

    if not local:
        print("  {}✗ No models found on disk. Deploy a model first.{}".format(C.RED, C.RESET))
        return

    # 1. Select model
    print("\n  {}Available models:{}".format(C.DIM, C.RESET))
    for i, m in enumerate(local, 1):
        size_mb = m["size_bytes"] / (1024 * 1024)
        active = " {}● active{}".format(C.GREEN, C.RESET) if loaded.get(m["model_type"]) == m["model_name"] else ""
        print("    {}[{}]{} {}/{} ({:.1f} MB){}".format(
            C.BOLD, i, C.RESET,
            m["model_type"], m["model_name"],
            size_mb, active,
        ))

    try:
        idx = int(input("\n  {}▸ Select model [1-{}]: {}".format(
            C.YELLOW, len(local), C.RESET,
        )).strip()) - 1
        if idx < 0 or idx >= len(local):
            print("  {}✗ Invalid selection{}".format(C.RED, C.RESET))
            return
    except (ValueError, EOFError):
        print("  {}✗ Invalid input{}".format(C.RED, C.RESET))
        return

    selected = local[idx]
    model_type = selected["model_type"]
    model_name = selected["model_name"]

    # 2. Check sample data
    sample_dir = os.path.join(os.getcwd(), "data", "sample")
    output_dir = os.path.join(os.getcwd(), "data", "sample_output")

    if not os.path.exists(sample_dir):
        print("  {}✗ Sample directory not found: {}{}".format(C.RED, sample_dir, C.RESET))
        return

    images = sorted(
        glob.glob(os.path.join(sample_dir, "*.tif"))
        + glob.glob(os.path.join(sample_dir, "*.png"))
        + glob.glob(os.path.join(sample_dir, "*.jpg"))
        + glob.glob(os.path.join(sample_dir, "*.bmp"))
    )
    if not images:
        print("  {}✗ No images found in {}{}".format(C.RED, sample_dir, C.RESET))
        return

    print("\n  {}Model :{}  {}/{}".format(C.DIM, C.RESET, model_type, model_name))
    print("  {}Input :{}  {} ({} images)".format(C.DIM, C.RESET, sample_dir, len(images)))
    print("  {}Output:{}  {}".format(C.DIM, C.RESET, output_dir))
    print()

    # 3. Progress callback
    def on_progress(msg):
        if "ERROR" in msg or "error" in msg:
            color = C.RED
            icon = "✗"
        elif "saved" in msg.lower() or "done" in msg.lower():
            color = C.GREEN
            icon = "✓"
        elif "/" in msg and "[" in msg:
            color = C.WHITE
            icon = "▸"
        else:
            color = C.CYAN
            icon = "•"
        print("  {} {}{}{}".format(icon, color, msg, C.RESET))

    # 4. Run inference
    try:
        from app.services.inference_service import run_sample_test

        on_progress("Starting: {}/{} on {} images".format(model_type, model_name, len(images)))

        results = run_sample_test(
            model_type=model_type,
            model_name=model_name,
            sample_dir=sample_dir,
            output_dir=output_dir,
            progress_callback=on_progress,
        )

        # 5. Summary
        success = sum(1 for r in results if "vector" in r)
        failed = sum(1 for r in results if "error" in r)
        avg_time = 0.0
        if success > 0:
            avg_time = sum(r.get("inference_time_ms", 0) for r in results if "vector" in r) / success

        print("\n  {}{}=== RESULTS ==={}".format(C.GREEN, C.BOLD, C.RESET))
        print("  ✓ {}{}/{}{} images processed".format(C.GREEN, success, len(results), C.RESET))
        if failed:
            print("  ✗ {}{}{} failed".format(C.RED, failed, C.RESET))
        if success > 0:
            dim = results[0].get("vector_dim", "?")
            print("  {}Embedding dim:{} {}".format(C.DIM, C.RESET, dim))
            print("  {}Avg inference :{} {:.1f} ms".format(C.DIM, C.RESET, avg_time))
        print("  {}Output saved  :{} {}".format(C.DIM, C.RESET, os.path.join(output_dir, "results.json")))

    except ImportError as exc:
        print("  {}✗ Missing dependency: {}{}".format(C.RED, exc, C.RESET))
        print("  {}On Jetson Nano, TensorRT comes with JetPack.{}".format(C.DIM, C.RESET))
        print("  {}On PC, install: pip install onnxruntime{}".format(C.DIM, C.RESET))
    except FileNotFoundError as exc:
        print("  {}✗ {}{}".format(C.RED, exc, C.RESET))
    except Exception as exc:
        print("  {}✗ Inference failed: {}{}".format(C.RED, exc, C.RESET))
        import traceback
        traceback.print_exc()

    print()

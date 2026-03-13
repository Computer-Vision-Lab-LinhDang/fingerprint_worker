# Fingerprint Worker

MQTT client for **Jetson Nano** — connects to the orchestrator via Mosquitto broker.
Includes an **interactive CLI** for monitoring and testing MQTT communication.

---

## Prerequisites

- Python 3.6
- Mosquitto broker running on the orchestrator machine
- Network access to the broker (LAN or Tailscale VPN)

## Installation

```bash
git clone <repo_url> fingerprint_worker
cd fingerprint_worker

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
WORKER_ID=jetson-nano-01

MQTT_BROKER_HOST=100.106.35.45   # Orchestrator IP (Tailscale or LAN)
MQTT_BROKER_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=
MQTT_KEEPALIVE=60
MQTT_RECONNECT_DELAY=5

HEARTBEAT_INTERVAL=10
```

| Variable             | Description                                                          |
| -------------------- | -------------------------------------------------------------------- |
| `WORKER_ID`          | Unique ID for this worker (e.g. `jetson-nano-01`)                    |
| `MQTT_BROKER_HOST`   | IP of the machine running Mosquitto. Use `localhost` if same machine |
| `MQTT_BROKER_PORT`   | Default: `1883`                                                      |
| `MQTT_USERNAME`      | Leave empty if broker has `allow_anonymous true`                     |
| `MQTT_PASSWORD`      | Leave empty if broker has `allow_anonymous true`                     |
| `HEARTBEAT_INTERVAL` | Seconds between heartbeats (default: `10`)                           |

## Run

```bash
source venv/bin/activate
python -m app.main
```

## CLI Menu

```
╔══════════════════════════════════════════════════╗
║        🖐  FINGERPRINT WORKER — CLI              ║
╚══════════════════════════════════════════════════╝

  Worker: jetson-nano-01  │  ● CONNECTED  │  Uptime: 2m 30s

  [1]  📊  Connection Status
  [2]  💓  Send Heartbeat
  [3]  📨  Send Test Message
  [4]  ✉️   Send Message to Orchestrator
  [5]  📋  Recent Message Log
  [6]  📈  Statistics
  [7]  ⚙️   Current Configuration
  [8]  🔄  Reconnect MQTT
  [9]  🧹  Clear Screen
  [0]  🚪  Exit
```

## MQTT Topics

| Direction | Topic                          | Description                              |
| --------- | ------------------------------ | ---------------------------------------- |
| Subscribe | `task/{worker_id}/embed`       | Receive embed tasks                      |
| Subscribe | `task/{worker_id}/match`       | Receive match tasks                      |
| Subscribe | `task/{worker_id}/message`     | Receive messages from orchestrator       |
| Publish   | `worker/{worker_id}/heartbeat` | Send heartbeat every N seconds           |
| Publish   | `worker/{worker_id}/status`    | LWT — auto-sent on unexpected disconnect |
| Publish   | `worker/{worker_id}/message`   | Send messages to orchestrator            |
| Publish   | `result/{task_id}`             | Send task results                        |

## Project Structure

```
fingerprint_worker/
├── app/
│   ├── main.py              # Entry point
│   ├── cli.py               # Interactive CLI
│   ├── core/config.py       # Settings from .env
│   ├── schemas/payload.py   # MQTT payload models
│   └── mqtt/
│       ├── client.py        # MQTT client + heartbeat + LWT
│       └── handlers.py      # Message handlers
├── .env
├── requirements.txt
└── README.md
```

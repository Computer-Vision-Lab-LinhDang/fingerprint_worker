
import json
import logging
import threading

import paho.mqtt.client as mqtt

from app.schemas.payload import TaskPayload, MatchPayload, ModelUpdatePayload, ModelStatusPayload

logger = logging.getLogger(__name__)


def create_message_handler(mqtt_client_ref):
    def on_message(client: mqtt.Client, message: mqtt.MQTTMessage) -> None:
        topic = message.topic
        parts = topic.split("/")

        try:
            if len(parts) >= 3 and parts[0] == "task":
                data = json.loads(message.payload.decode())

                # task/{worker_id}/model/update
                if len(parts) >= 4 and parts[2] == "model" and parts[3] == "update":
                    payload = ModelUpdatePayload(**data)
                    logger.info(
                        "📥 MODEL UPDATE: type=%s, name=%s, ver=%s",
                        payload.model_type, payload.model_name, payload.version,
                    )
                    # Run download in background thread to not block MQTT
                    thread = threading.Thread(
                        target=_handle_model_update,
                        args=(mqtt_client_ref, payload),
                        daemon=True,
                    )
                    thread.start()
                    return

                task_type = parts[2]

                if task_type == "embed":
                    payload = TaskPayload(**data)
                    logger.info(
                        "📥 EMBED task: id=%s, image_url=%s",
                        payload.task_id, payload.image_url[:60],
                    )
                    # Mark busy
                    mqtt_client_ref.current_task_id = payload.task_id

                    # Run inference in background thread
                    from app.services.task_service import TaskService
                    task_svc = TaskService(mqtt_client_ref)
                    thread = threading.Thread(
                        target=_handle_embed_task,
                        args=(task_svc, mqtt_client_ref, data),
                        daemon=True,
                    )
                    thread.start()
                    return

                elif task_type == "match":
                    payload = MatchPayload(**data)
                    logger.info(
                        "📥 MATCH task: id=%s, candidates=%d",
                        payload.task_id, len(payload.candidate_vectors),
                    )

                elif task_type == "message":
                    content = data.get("content", "")
                    sender = data.get("sender", "orchestrator")
                    logger.info(
                        "📩 MESSAGE from %s: %s", sender, content,
                    )
                    print("\n  📩 Message from {}: {}".format(sender, content))

                else:
                    logger.warning("Unknown task type: %s", task_type)

            else:
                logger.warning("Unknown topic: %s", topic)

        except Exception as exc:
            logger.error("Error processing message '%s': %s", topic, exc)

    return on_message


def _handle_embed_task(task_svc, mqtt_client_ref, task_data):
    """Handle embed task in background thread — runs inference."""
    try:
        task_svc.process_embed(task_data)
    except Exception as exc:
        logger.error("Embed task failed: %s", exc)
    finally:
        # Reset to idle
        mqtt_client_ref.current_task_id = None


def _handle_model_update(mqtt_client_ref, payload):
    """Handle model download in background thread."""
    from app.services.model_service import get_model_service

    worker_id = mqtt_client_ref.worker_id
    model_service = get_model_service()

    # Publish: downloading
    _publish_model_status(
        mqtt_client_ref, worker_id, payload, "downloading",
    )

    # Download
    success, error = model_service.download_model(
        model_type=payload.model_type,
        model_name=payload.model_name,
        version=payload.version,
        download_url=payload.download_url,
    )

    # Publish: ready or failed
    status = "ready" if success else "failed"
    _publish_model_status(
        mqtt_client_ref, worker_id, payload, status, error,
    )


def _publish_model_status(mqtt_client_ref, worker_id, payload, status, error=None):
    """Publish model status to orchestrator."""
    status_payload = ModelStatusPayload(
        worker_id=worker_id,
        model_type=payload.model_type,
        model_name=payload.model_name,
        version=payload.version,
        status=status,
        error=error,
    )
    topic = "worker/{}/model/status".format(worker_id)
    mqtt_client_ref.publish(
        topic, json.dumps(status_payload.__dict__), qos=1,
    )
    logger.info(
        "📤 Model status: %s/%s → %s",
        payload.model_type, payload.model_name, status,
    )

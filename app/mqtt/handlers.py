from __future__ import annotations

import json
import logging

import paho.mqtt.client as mqtt

from app.schemas.payload import TaskPayload, MatchPayload

logger = logging.getLogger(__name__)


def create_message_handler(mqtt_client_ref):
    def on_message(client: mqtt.Client, message: mqtt.MQTTMessage) -> None:
        topic = message.topic
        parts = topic.split("/")

        try:
            if len(parts) >= 3 and parts[0] == "task":
                task_type = parts[2]
                data = json.loads(message.payload.decode())

                if task_type == "embed":
                    payload = TaskPayload(**data)
                    logger.info(
                        "📥 EMBED task: id=%s, image_url=%s",
                        payload.task_id, payload.image_url[:60],
                    )

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
                    print(f"\n  📩 Message from {sender}: {content}")

                else:
                    logger.warning("Unknown task type: %s", task_type)

                # Mark busy
                mqtt_client_ref.current_task_id = data.get("task_id")

                # TODO: Integrate model inference here when ready
                # Currently just logs and resets state
                mqtt_client_ref.current_task_id = None

            else:
                logger.warning("Unknown topic: %s", topic)

        except Exception as exc:
            logger.error("Error processing message '%s': %s", topic, exc)

    return on_message

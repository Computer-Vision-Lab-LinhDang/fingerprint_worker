"""Task processing service — placeholder for AI inference integration."""

import logging

logger = logging.getLogger(__name__)


class TaskService:
    """Handles task processing logic.

    TODO: Integrate with inference engine when ready.
    """

    def __init__(self, mqtt_client):
        self._mqtt_client = mqtt_client

    def process_embed(self, task_data):
        """Process an embedding task.

        TODO: Call inference engine, return embedding vector.
        """
        logger.info("EMBED task received — inference not implemented yet")
        return None

    def process_match(self, task_data):
        """Process a matching task.

        TODO: Call inference engine, compare vectors.
        """
        logger.info("MATCH task received — inference not implemented yet")
        return None

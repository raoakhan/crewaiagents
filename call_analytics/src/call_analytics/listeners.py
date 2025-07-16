"""Basic OpenTelemetry trace listener example."""

from crewai.listeners.base_listener import BaseListener
import logging
import time

logger = logging.getLogger(__name__)

class TraceListener(BaseListener):
    """Logs task timings and token usage (placeholder)."""

    def before_task(self, task, *args, **kwargs):  # noqa: D401
        task._start_time = time.perf_counter()
        logger.info("Starting task %s", task.name)

    def after_task(self, task, result, *args, **kwargs):  # noqa: D401
        duration = time.perf_counter() - getattr(task, "_start_time", 0)
        logger.info("Finished task %s in %.2fs", task.name, duration)
        # Here you could push to OTLP exporter.

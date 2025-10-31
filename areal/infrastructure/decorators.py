"""Decorators for event-driven infrastructure.

This module provides decorators that simplify event firing for common patterns
like before/after method execution.
"""

import asyncio
import functools
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def fire_events(
    before: str | None = None,
    after: str | None = None,
    on_error: str | Callable | None = None,
    extract_context: Callable | None = None,
):
    """
    Decorator that fires events before and after method execution.

    This decorator simplifies event firing by automatically sending events
    at method boundaries. It handles event bus initialization, error cases,
    and context extraction.

    Parameters
    ----------
    before : str, optional
        Event name to fire before method execution.
        Event payload includes: sender=self, method_name, args, kwargs, and extracted context.
    after : str, optional
        Event name to fire after successful method execution.
        Event payload includes: sender=self, method_name, result, and extracted context.
    on_error : str or callable, optional
        - If str: Event name to fire if method raises an exception.
          Event payload includes: sender=self, method_name, error, and extracted context.
        - If callable: Custom error handler with signature:
          on_error(bus, self, exception, method_name, context) -> None
          The handler can decide whether to fire events, log, or handle the error differently.
    extract_context : callable, optional
        Function to extract additional context from method arguments.
        Signature: extract_context(self, *args, **kwargs) -> dict
        The returned dict is merged into event payload.

    Examples
    --------
    >>> from areal.infrastructure import WorkflowEvents, fire_events
    >>>
    >>> class InferenceEngine:
    ...     def __init__(self):
    ...         self._version = 0
    ...
    ...     @fire_events(
    ...         before=WorkflowEvents.BEFORE_WEIGHT_UPDATE,
    ...         after=WorkflowEvents.AFTER_WEIGHT_UPDATE,
    ...         extract_context=lambda self, meta: {
    ...             'current_version': self.get_version(),
    ...             'next_version': meta.model_version,
    ...         }
    ...     )
    ...     def update_weights(self, meta):
    ...         # Just business logic - events fire automatically!
    ...         self._apply_update(meta)
    ...         self._version = meta.model_version
    ...
    ...     def get_version(self):
    ...         return self._version

    Advanced Usage with Async Methods
    ----------------------------------
    >>> @fire_events(before='task-started', after='task-completed')
    ... async def process_task(self, task_id):
    ...     result = await self._process(task_id)
    ...     return result

    Context Extraction
    ------------------
    >>> def extract_update_context(self, meta):
    ...     return {
    ...         'current_version': self.get_version(),
    ...         'next_version': meta.model_version,
    ...         'update_type': meta.type,
    ...         'model_path': meta.path,
    ...     }
    >>>
    >>> @fire_events(
    ...     before=WorkflowEvents.BEFORE_WEIGHT_UPDATE,
    ...     after=WorkflowEvents.AFTER_WEIGHT_UPDATE,
    ...     extract_context=extract_update_context
    ... )
    ... def update_weights(self, meta):
    ...     # Context automatically added to events
    ...     self._apply_update(meta)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Get event bus
            try:
                from .events import get_event_bus

                bus = get_event_bus()
            except Exception as e:
                # Event bus not initialized - just execute method
                logger.debug(f"Event bus not available: {e}")
                return func(self, *args, **kwargs)

            # Extract context
            context = {}
            if extract_context is not None:
                try:
                    context = extract_context(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to extract context: {e}")

            # Build base payload
            method_name = func.__name__
            base_payload = {
                "method_name": method_name,
                **context,
            }

            # Fire BEFORE event
            if before:
                try:
                    bus.send(
                        before,
                        sender=self,
                        **base_payload,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fire BEFORE event '{before}' for {method_name}: {e}"
                    )

            # Execute method
            result = None
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                # Handle error: fire event or call custom handler
                if on_error:
                    try:
                        if callable(on_error):
                            # Custom error handler
                            on_error(bus, self, e, method_name, context)
                        else:
                            # Fire error event
                            bus.send(
                                on_error,
                                sender=self,
                                error=e,
                                **base_payload,
                            )
                    except Exception as event_error:
                        logger.warning(
                            f"Failed to handle error for {method_name}: {event_error}"
                        )
                raise

            # Fire AFTER event (only if no error)
            if after:
                try:
                    # Update context with result info
                    after_payload = {
                        **base_payload,
                    }
                    # Add result info if available
                    if "new_version" in context:
                        after_payload["new_version"] = context["new_version"]

                    bus.send(
                        after,
                        sender=self,
                        result=result,
                        **after_payload,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fire AFTER event '{after}' for {method_name}: {e}"
                    )

            return result

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Get event bus
            try:
                from .events import get_event_bus

                bus = get_event_bus()
            except Exception as e:
                # Event bus not initialized - just execute method
                logger.debug(f"Event bus not available: {e}")
                return await func(self, *args, **kwargs)

            # Extract context
            context = {}
            if extract_context is not None:
                try:
                    context = extract_context(self, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to extract context: {e}")

            # Build base payload
            method_name = func.__name__
            base_payload = {
                "method_name": method_name,
                **context,
            }

            # Fire BEFORE event
            if before:
                try:
                    bus.send(
                        before,
                        sender=self,
                        **base_payload,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fire BEFORE event '{before}' for {method_name}: {e}"
                    )

            # Execute async method
            result = None
            try:
                result = await func(self, *args, **kwargs)
            except Exception as e:
                # Handle error: fire event or call custom handler
                if on_error:
                    try:
                        if callable(on_error):
                            # Custom error handler
                            on_error(bus, self, e, method_name, context)
                        else:
                            # Fire error event
                            bus.send(
                                on_error,
                                sender=self,
                                error=e,
                                **base_payload,
                            )
                    except Exception as event_error:
                        logger.warning(
                            f"Failed to handle error for {method_name}: {event_error}"
                        )
                raise

            # Fire AFTER event (only if no error)
            if after:
                try:
                    after_payload = {
                        **base_payload,
                    }
                    if "new_version" in context:
                        after_payload["new_version"] = context["new_version"]

                    bus.send(
                        after,
                        sender=self,
                        result=result,
                        **after_payload,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fire AFTER event '{after}' for {method_name}: {e}"
                    )

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

"""Event bus implementation using blinker for local in-process events.

This module provides a unified event system with support for:
- Local events (blinker) - Phase 1 implementation
- Future distributed events (ZMQ/Redis) - Phase 2 (placeholder)

Key Features:
- Synchronous event dispatch (blocks until all handlers complete)
- Type-safe event namespaces
- Easy handler registration and management
- Thread-safe signal dispatching
"""

import logging
from collections.abc import Callable
from typing import Any, Literal

try:
    from blinker import signal
except ImportError:
    raise ImportError(
        "blinker is required for event system. Install with: pip install blinker"
    )

logger = logging.getLogger(__name__)


class EventBus:
    """
    Event bus with local (blinker) and future distributed support.

    The event bus uses blinker for in-process event dispatching. Events are
    dispatched synchronously, meaning send() blocks until all handlers complete.

    Future Enhancement (Phase 2):
    - Add distributed mode for cross-node event propagation
    - Support for ZMQ or Redis Pub/Sub backends

    Parameters
    ----------
    mode : {'local', 'distributed'}, default 'local'
        Event bus mode. Currently only 'local' is implemented.
        'distributed' mode is reserved for future implementation.

    Examples
    --------
    >>> bus = EventBus(mode='local')
    >>> def my_handler(sender, **kwargs):
    ...     print(f"Event received: {kwargs}")
    >>> bus.connect('my-event', my_handler)
    >>> bus.send('my-event', sender=None, data='hello')
    Event received: {'data': 'hello'}
    """

    def __init__(self, mode: Literal["local", "distributed"] = "local"):
        """
        Initialize event bus.

        Parameters
        ----------
        mode : str
            Event bus mode ('local' or 'distributed')
        """
        if mode not in ("local", "distributed"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'distributed'")

        self.mode = mode
        self._local_signals = {}
        # Use instance ID to create unique signal namespace
        import uuid

        self._signal_namespace = str(uuid.uuid4())

        if mode == "distributed":
            logger.warning(
                "Distributed mode is not yet implemented. "
                "Falling back to local mode. Distributed events will only "
                "be dispatched locally."
            )

    def signal(self, name: str):
        """
        Get or create a signal by name.

        Parameters
        ----------
        name : str
            Name of the signal/event

        Returns
        -------
        blinker.Signal
            The signal object for this event name
        """
        if name not in self._local_signals:
            # Use namespaced signal name to avoid cross-instance contamination
            signal_name = f"{self._signal_namespace}:{name}"
            self._local_signals[name] = signal(signal_name)
        return self._local_signals[name]

    def connect(
        self,
        event_name: str,
        handler: Callable,
        sender: Any = None,
        distributed: bool = False,
    ):
        """
        Connect a handler to an event.

        The handler will be called when the event is sent. Handlers are called
        synchronously in the order they were registered.

        Parameters
        ----------
        event_name : str
            Name of the event to listen for
        handler : Callable
            Callback function with signature: handler(sender, **kwargs)
        sender : Any, optional
            Only call handler for events from this specific sender.
            If None, handler is called for all senders.
        distributed : bool, default False
            If True, also listen for this event from remote nodes.
            Currently not implemented (Phase 2 feature).

        Examples
        --------
        >>> def on_batch_ready(sender, batch_size, **kwargs):
        ...     print(f"Batch ready with {batch_size} items")
        >>> bus.connect('batch-ready', on_batch_ready)
        >>> bus.send('batch-ready', sender=None, batch_size=128)
        Batch ready with 128 items
        """
        # Connect to local signal
        # Use weak=False to prevent garbage collection of lambda handlers
        # Only pass sender if not None (blinker interprets sender=None as filtering for None)
        if sender is None:
            self.signal(event_name).connect(handler, weak=False)
        else:
            self.signal(event_name).connect(handler, sender=sender, weak=False)

        # Distributed mode (Phase 2 - not implemented yet)
        if distributed and self.mode == "distributed":
            logger.debug(
                f"Distributed event '{event_name}' registered, but distributed "
                "mode is not yet implemented. Events will only be dispatched locally."
            )

    def send(
        self, event_name: str, sender: Any = None, distributed: bool = False, **kwargs
    ):
        """
        Send an event and block until all handlers complete.

        This method is synchronous - it will not return until all registered
        handlers have finished processing the event. Handlers are called
        sequentially in registration order.

        Parameters
        ----------
        event_name : str
            Name of the event to send
        sender : Any, optional
            The object sending the event. Handlers can filter by sender.
        distributed : bool, default False
            If True, also propagate event to remote nodes.
            Currently not implemented (Phase 2 feature).
        **kwargs
            Event data passed to handlers

        Examples
        --------
        >>> bus.send('model-version-updated', sender=trainer, version=42)
        # Blocks until all handlers for 'model-version-updated' complete

        Notes
        -----
        If a handler raises an exception, subsequent handlers will not run.
        Consider wrapping handler logic in try-except blocks for robustness.
        """
        # Send to local handlers (always)
        self.signal(event_name).send(sender, **kwargs)

        # Send to distributed nodes (Phase 2 - not implemented yet)
        if distributed and self.mode == "distributed":
            logger.debug(
                f"Distributed send for '{event_name}' requested, but distributed "
                "mode is not yet implemented. Event only dispatched locally."
            )

    async def send_async(
        self, event_name: str, sender: Any = None, distributed: bool = False, **kwargs
    ):
        """
        Send event to async handlers (future support).

        This method is reserved for future async handler support. Use send()
        for synchronous handlers.

        Parameters
        ----------
        event_name : str
            Name of the event
        sender : Any, optional
            Event sender
        distributed : bool, default False
            Propagate to remote nodes
        **kwargs
            Event data

        Raises
        ------
        NotImplementedError
            Async handlers are not yet implemented
        """
        raise NotImplementedError(
            "Async event handlers not yet implemented. Use send() for synchronous handlers."
        )

    def disconnect(self, event_name: str, handler: Callable, sender: Any = None):
        """
        Disconnect a handler from an event.

        Parameters
        ----------
        event_name : str
            Name of the event
        handler : Callable
            The handler to disconnect
        sender : Any, optional
            Only disconnect for this specific sender
        """
        if event_name in self._local_signals:
            if sender is None:
                self.signal(event_name).disconnect(handler)
            else:
                self.signal(event_name).disconnect(handler, sender=sender)

    def has_receivers(self, event_name: str) -> bool:
        """
        Check if an event has any registered handlers.

        Parameters
        ----------
        event_name : str
            Name of the event to check

        Returns
        -------
        bool
            True if event has at least one handler, False otherwise
        """
        if event_name not in self._local_signals:
            return False
        return bool(self.signal(event_name).receivers)


# ==============================================================================
# Event Name Constants
# ==============================================================================


class QueueEvents:
    """Event names for queue operations (local only)."""

    ITEM_ADDED = "queue-item-added"
    """Fired when item is added to queue."""

    ITEM_REMOVED = "queue-item-removed"
    """Fired when item is removed from queue."""

    ITEM_FILTERED = "queue-item-filtered"
    """Fired when item is rejected by filter."""


class CacheEvents:
    """Event names for cache operations (local only)."""

    ITEM_ADDED = "cache-item-added"
    """Fired when item is added to cache."""

    ITEM_REMOVED = "cache-item-removed"
    """Fired when item is removed from cache."""

    CACHE_CLEARED = "cache-cleared"
    """Fired when cache is cleared."""


class WorkflowEvents:
    """Event names for workflow lifecycle (local and distributed)."""

    # Local events
    ROLLOUT_STARTED = "rollout-started"
    """Fired when rollout starts (local)."""

    ROLLOUT_COMPLETED = "rollout-completed"
    """Fired when rollout completes (local)."""

    BATCH_READY = "batch-ready"
    """Fired when training batch is ready (local)."""

    # Distributed events (Phase 2)
    PRE_WEIGHT_UPDATE = "pre-weight-update"
    """Fired before weight update (should be distributed in Phase 2)."""

    POST_WEIGHT_UPDATE = "post-weight-update"
    """Fired after weight update (should be distributed in Phase 2)."""

    MODEL_VERSION_UPDATED = "model-version-updated"
    """Fired when model version changes (should be distributed in Phase 2)."""

    TRAINING_STEP_COMPLETED = "training-step-completed"
    """Fired when training step completes (should be distributed in Phase 2)."""

    CAPACITY_CHANGED = "capacity-changed"
    """Fired when staleness capacity changes (should be distributed in Phase 2)."""


# ==============================================================================
# Global Event Bus Instance
# ==============================================================================

_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.

    Returns
    -------
    EventBus
        The global event bus

    Raises
    ------
    RuntimeError
        If event bus has not been initialized via initialize_event_bus()
    """
    global _event_bus
    if _event_bus is None:
        raise RuntimeError(
            "Event bus not initialized. Call initialize_event_bus() first."
        )
    return _event_bus


def initialize_event_bus(mode: Literal["local", "distributed"] = "local") -> EventBus:
    """
    Initialize the global event bus.

    This should be called once at application startup.

    Parameters
    ----------
    mode : {'local', 'distributed'}, default 'local'
        Event bus mode. Currently only 'local' is fully implemented.

    Returns
    -------
    EventBus
        The initialized event bus

    Examples
    --------
    >>> # In application startup
    >>> bus = initialize_event_bus(mode='local')
    >>> # Later in code
    >>> bus = get_event_bus()
    """
    global _event_bus
    _event_bus = EventBus(mode=mode)
    logger.info(f"Initialized event bus in {mode} mode")
    return _event_bus

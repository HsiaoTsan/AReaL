"""AReaL Infrastructure: Queues, Caches, Events, and Dependency Injection.

This package provides the foundational infrastructure for AReaL:

1. **Event System** (events.py)
   - EventBus for local in-process events using blinker
   - Event name constants (QueueEvents, CacheEvents, WorkflowEvents)
   - Future distributed event support (Phase 2)

2. **Queues** (queue.py)
   - FilterableQueue with filter support
   - Compatible with queue.Queue API
   - Pluggable backends (memory, Redis, RabbitMQ)

3. **Caches** (cache.py)
   - ListCache with thread-safe list operations
   - Protocol-based interface (Cache)
   - Compatible with Python list[] operations

4. **Dependency Injection** (container.py, providers.py)
   - InfrastructureContainer for component wiring
   - InitializableProvider for two-phase initialization
   - Easy configuration and testing

Quick Start
-----------
>>> from areal.infrastructure import initialize_infrastructure, container, get_event_bus
>>>
>>> # Initialize infrastructure at application startup
>>> initialize_infrastructure(config={
...     'event_bus_mode': 'local',
...     'max_queue_size': 10240,
... })
>>>
>>> # Get components
>>> bus = get_event_bus()
>>> queue = container.task_input_queue()
>>> cache = container.result_cache()
>>>
>>> # Use components
>>> def my_handler(sender, **kwargs):
...     print(f"Event received: {kwargs}")
>>> bus.connect('my-event', my_handler)
>>> bus.send('my-event', sender=None, data='hello')
"""

import logging
from typing import Any

from .cache import Cache, ListCache
from .container import InfrastructureContainer, container

# Core components
from .events import (
    CacheEvents,
    EventBus,
    QueueEvents,
    WorkflowEvents,
    get_event_bus,
    initialize_event_bus,
)
from .providers import InitializableProvider, ThreadSafeInitializableProvider
from .queue import FilterableQueue

logger = logging.getLogger(__name__)


# ==============================================================================
# Initialization Helper
# ==============================================================================


def initialize_infrastructure(
    config: dict[str, Any] | None = None,
) -> InfrastructureContainer:
    """
    Initialize AReaL infrastructure with configuration.

    This is the main entry point for setting up the infrastructure layer.
    Call this once at application startup before using any infrastructure
    components.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with keys:
        - event_bus_mode : {'local', 'distributed'}, default 'local'
        - queue_backend : str, default 'memory'
        - max_queue_size : int, default 10240
        - fire_queue_events : bool, default False
        - fire_cache_events : bool, default False
        - ... other configuration options

    Returns
    -------
    InfrastructureContainer
        The configured container

    Examples
    --------
    >>> # Minimal initialization (all defaults)
    >>> from areal.infrastructure import initialize_infrastructure
    >>> container = initialize_infrastructure()

    >>> # With custom configuration
    >>> container = initialize_infrastructure({
    ...     'event_bus_mode': 'local',
    ...     'max_queue_size': 5000,
    ...     'fire_queue_events': True,
    ... })

    >>> # Later, get components
    >>> from areal.infrastructure import get_event_bus, container
    >>> bus = get_event_bus()
    >>> queue = container.task_input_queue()

    Notes
    -----
    This function:
    1. Initializes the global event bus
    2. Configures the DI container
    3. Registers event handlers (if any)
    """
    config = config or {}

    # Set default configuration values
    default_config = {
        "event_bus_mode": "local",
        "queue_backend": "memory",
        "max_queue_size": 10240,
        "fire_queue_events": False,
        "fire_cache_events": False,
    }
    default_config.update(config)

    # Initialize event bus
    event_bus_mode = default_config.get("event_bus_mode", "local")
    initialize_event_bus(mode=event_bus_mode)

    # Configure container
    container.config.from_dict(default_config)

    logger.info("AReaL infrastructure initialized successfully")
    logger.info(f"  Event bus mode: {event_bus_mode}")
    logger.info(f"  Queue backend: {default_config.get('queue_backend')}")
    logger.info(f"  Max queue size: {default_config.get('max_queue_size')}")

    return container


# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Initialization
    "initialize_infrastructure",
    # Event System
    "EventBus",
    "get_event_bus",
    "initialize_event_bus",
    "QueueEvents",
    "CacheEvents",
    "WorkflowEvents",
    # Queues
    "FilterableQueue",
    # Caches
    "Cache",
    "ListCache",
    # Dependency Injection
    "InitializableProvider",
    "ThreadSafeInitializableProvider",
    "InfrastructureContainer",
    "container",
]


# ==============================================================================
# Version Info
# ==============================================================================

__version__ = "0.1.0"
__author__ = "AReaL Infrastructure Team"
__doc_url__ = "https://github.com/your-org/areal"

"""Dependency injection container for AReaL infrastructure.

This module provides the main DI container that wires together all infrastructure
components (queues, caches, event bus, handlers, engines, etc.).

The container follows a declarative configuration style where providers are
defined as class attributes. Dependencies are automatically resolved when
instances are requested.
"""

import logging

try:
    from dependency_injector import containers, providers
except ImportError:
    raise ImportError(
        "dependency-injector is required. Install with: pip install dependency-injector"
    )

from .cache import ListCache
from .events import EventBus
from .queue import FilterableQueue

logger = logging.getLogger(__name__)


class InfrastructureContainer(containers.DeclarativeContainer):
    """
    Main DI container for AReaL infrastructure components.

    This container provides factories for creating infrastructure components
    with automatic dependency injection. Configuration can be provided via
    the config provider.

    Configuration Structure
    ----------------------
    {
        'event_bus_mode': 'local',  # or 'distributed'
        'queue_backend': 'memory',  # or 'redis://...'
        'max_queue_size': 10240,
        'fire_queue_events': False,
        'fire_cache_events': False,
    }

    Examples
    --------
    >>> # Create and configure container
    >>> container = InfrastructureContainer()
    >>> container.config.from_dict({
    ...     'event_bus_mode': 'local',
    ...     'max_queue_size': 1000,
    ... })
    >>>
    >>> # Get components with auto-injected dependencies
    >>> bus = container.event_bus()
    >>> queue = container.task_input_queue()
    >>> cache = container.result_cache()

    >>> # Override for testing
    >>> from unittest.mock import Mock
    >>> with container.event_bus.override(Mock()):
    ...     bus = container.event_bus()  # Returns mock
    """

    # ==============================================================================
    # Configuration
    # ==============================================================================

    config = providers.Configuration()
    """
    Configuration provider for infrastructure settings.

    Can be populated from dict, YAML, JSON, or environment variables.

    Examples
    --------
    >>> # From dict
    >>> container.config.from_dict({'max_queue_size': 5000})
    >>>
    >>> # From YAML file
    >>> container.config.from_yaml('config.yaml')
    >>>
    >>> # From environment variables
    >>> container.config.event_bus_mode.from_env('AREAL_EVENT_BUS_MODE', default='local')
    """

    # ==============================================================================
    # Event System
    # ==============================================================================

    event_bus = providers.Singleton(
        EventBus,
        mode=providers.Callable(
            lambda mode=None: mode or "local", mode=config.event_bus_mode
        ),
    )
    """
    Global event bus (singleton).

    Provides in-process event dispatching using blinker. Future support for
    distributed events (ZMQ/Redis) in Phase 2.

    Examples
    --------
    >>> bus = container.event_bus()
    >>> bus.connect('my-event', my_handler)
    >>> bus.send('my-event', sender=self, data='hello')
    """

    # ==============================================================================
    # Queues
    # ==============================================================================

    task_input_queue = providers.Factory(
        FilterableQueue,
        name="task_input",
        maxsize=providers.Callable(lambda x=None: x or 10240, x=config.max_queue_size),
        backend=providers.Callable(
            lambda x=None: x or "memory", x=config.queue_backend
        ),
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_queue_events
        ),
    )
    """
    Factory for task input queues.

    Creates new queue instances on each call. Use for submitting tasks to
    AsyncTaskRunner or similar components.

    Examples
    --------
    >>> queue = container.task_input_queue()
    >>> queue.put(task_data)
    """

    task_output_queue = providers.Factory(
        FilterableQueue,
        name="task_output",
        maxsize=providers.Callable(lambda x=None: x or 10240, x=config.max_queue_size),
        backend=providers.Callable(
            lambda x=None: x or "memory", x=config.queue_backend
        ),
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_queue_events
        ),
    )
    """Factory for task output queues."""

    # ==============================================================================
    # Caches
    # ==============================================================================

    result_cache = providers.Factory(
        ListCache,
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_cache_events
        ),
    )
    """
    Factory for result caches.

    Creates new cache instances on each call. Caches are thread-safe and
    compatible with Python list[] operations.

    Examples
    --------
    >>> cache = container.result_cache()
    >>> cache.append(result)
    >>> items = cache[:10]  # Get first 10 items
    """

    pending_results = providers.Factory(
        ListCache,
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_cache_events
        ),
    )
    """Factory for pending results cache."""

    pending_inputs = providers.Factory(
        ListCache,
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_cache_events
        ),
    )
    """Factory for pending inputs cache."""

    # ==============================================================================
    # Example: Inference Engine (Commented - for reference)
    # ==============================================================================

    # Uncomment and configure when integrating with actual inference engines
    #
    # inference_engine = InitializableProvider(
    #     # Dynamically select engine type based on config
    #     lambda cfg: (
    #         __import__('areal.experimental.sglang_engine', fromlist=['SGLangEngine'])
    #         .SGLangEngine(cfg['inference_config'], cfg.get('engine_args'))
    #         if cfg.get('engine_type') == 'sglang'
    #         else __import__('areal.core.remote_inf_engine', fromlist=['RemoteInfEngine'])
    #         .RemoteInfEngine(cfg['inference_config'], ...)
    #     ),
    #     cfg=config,
    #     init_kwargs={'train_data_parallel_size': config.dp_size}
    # )

    # ==============================================================================
    # Example: Event Handlers (Commented - for reference)
    # ==============================================================================

    # Uncomment and configure when implementing actual handlers
    #
    # pre_weight_update_handler = providers.Factory(
    #     PreWeightUpdateHandler,
    #     queue=task_input_queue,
    #     cache=pending_results,
    #     inference_engine=inference_engine,
    # )


# ==============================================================================
# Global Container Instance
# ==============================================================================

container = InfrastructureContainer()
"""
Global container instance for convenient access.

This is initialized automatically and can be used throughout the application.
Configuration should be set via container.config before requesting components.

Examples
--------
>>> from areal.infrastructure import container
>>> container.config.from_dict({'max_queue_size': 5000})
>>> bus = container.event_bus()
"""

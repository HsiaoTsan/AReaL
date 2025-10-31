"""Application-level dependency injection container for AReaL core components.

This module extends the infrastructure container to provide application-specific
components like AsyncTaskRunner, WorkflowExecutor, and InferenceEngines.
"""

import logging

try:
    from dependency_injector import containers, providers
except ImportError:
    raise ImportError(
        "dependency-injector is required. Install with: pip install dependency-injector"
    )

from areal.core.async_task_runner import AsyncTaskRunner
from areal.core.workflow_executor import WorkflowExecutor
from areal.infrastructure import FilterableQueue, InfrastructureContainer, ListCache

logger = logging.getLogger(__name__)


class ApplicationContainer(containers.DeclarativeContainer):
    """
    Application-level DI container for AReaL core components.

    This container extends the infrastructure layer by adding providers for
    application-specific components that use queues, caches, and events.

    Configuration Structure
    ----------------------
    Inherits from InfrastructureContainer and adds:
    {
        # Infrastructure config (inherited)
        'event_bus_mode': 'local',
        'queue_backend': 'memory',
        'max_queue_size': 10240,
        'fire_queue_events': False,
        'fire_cache_events': False,

        # Application-specific config
        'poll_wait_time': 0.05,
        'poll_sleep_time': 0.5,
        'enable_rollout_tracing': False,
    }

    Examples
    --------
    >>> # Create and configure container
    >>> container = ApplicationContainer()
    >>> container.config.from_dict({
    ...     'max_queue_size': 1000,
    ...     'enable_rollout_tracing': True,
    ... })
    >>
    >>> # Get components with auto-injected dependencies
    >>> runner = container.async_task_runner()  # AsyncTaskRunner with queues injected
    >>> # Note: runner is already initialized via InitializableProvider
    """

    # ==============================================================================
    # Configuration
    # ==============================================================================

    config = providers.Configuration()
    """Configuration provider for application settings."""

    # ==============================================================================
    # Infrastructure Components (from parent)
    # ==============================================================================

    # Import infrastructure providers
    infrastructure = providers.Container(
        InfrastructureContainer,
        config=config,
    )
    """Infrastructure container with event bus, queues, and caches."""

    # ==============================================================================
    # AsyncTaskRunner Components
    # ==============================================================================

    async_task_input_queue = providers.Singleton(
        FilterableQueue,
        name="async_task_input",
        maxsize=providers.Callable(lambda x=None: x or 10240, x=config.max_queue_size),
        backend=providers.Callable(
            lambda x=None: x or "memory", x=config.queue_backend
        ),
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_queue_events
        ),
    )
    """
    Singleton AsyncTaskRunner input queue.

    All AsyncTaskRunner instances share this queue (application-wide).
    Event handlers can reliably get this queue reference.
    """

    async_task_output_queue = providers.Singleton(
        FilterableQueue,
        name="async_task_output",
        maxsize=providers.Callable(lambda x=None: x or 10240, x=config.max_queue_size),
        backend=providers.Callable(
            lambda x=None: x or "memory", x=config.queue_backend
        ),
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_queue_events
        ),
    )
    """Singleton AsyncTaskRunner output queue."""

    async_task_result_cache = providers.Singleton(
        ListCache,
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_cache_events
        ),
    )
    """Singleton AsyncTaskRunner result cache."""

    async_task_runner = providers.Singleton(
        AsyncTaskRunner,
        input_queue=async_task_input_queue,
        output_queue=async_task_output_queue,
        result_cache=async_task_result_cache,
        max_queue_size=providers.Callable(
            lambda x=None: x or 10240, x=config.max_queue_size
        ),
        poll_wait_time=providers.Callable(
            lambda x=None: x if x is not None else 0.05, x=config.poll_wait_time
        ),
        poll_sleep_time=providers.Callable(
            lambda x=None: x if x is not None else 0.5, x=config.poll_sleep_time
        ),
        enable_tracing=providers.Callable(
            lambda x=None: x if x is not None else False,
            x=config.enable_rollout_tracing,
        ),
    )
    """
    Singleton AsyncTaskRunner with shared queues.

    Returns the same instance on every call, with singleton queues and caches injected.
    Note: You must call initialize(logger) manually before first use.

    Examples
    --------
    >>> runner = container.async_task_runner()
    >>> runner.initialize(logger=my_logger)  # Must initialize manually
    >>> runner.submit(my_async_fn, *args)
    >>> results = runner.wait(count=10)

    >>> # Event handlers can get the same queue
    >>> queue = container.async_task_input_queue()  # Same queue as runner uses
    """

    # ==============================================================================
    # WorkflowExecutor Components
    # ==============================================================================

    workflow_pending_results = providers.Factory(
        ListCache,
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_cache_events
        ),
    )
    """Factory for WorkflowExecutor pending results cache."""

    workflow_pending_inputs = providers.Factory(
        ListCache,
        fire_events=providers.Callable(
            lambda x=None: x if x is not None else False, x=config.fire_cache_events
        ),
    )
    """Factory for WorkflowExecutor pending inputs cache."""

    # ==============================================================================
    # Event Handlers
    # ==============================================================================

    prox_t_handler = providers.Singleton(
        lambda: __import__(
            "areal.handlers", fromlist=["ProxTLogprobHandler"]
        ).ProxTLogprobHandler(
            output_queue=app_container.async_task_output_queue(),
            result_cache=app_container.async_task_result_cache(),
        )
    )
    """
    Singleton handler for computing log_prob_prox_t before weight updates.

    This handler implements segment-wise decoupled PPO by recomputing proximal
    policy logprobs at prox_t version (behavior + 1) before weight updates.

    To enable, call handler.register() after initializing the container:

    Examples
    --------
    >>> from areal.core.app_container import app_container
    >>> handler = app_container.prox_t_handler()
    >>> handler.register()  # Register to BEFORE_WEIGHT_UPDATE events
    """

    # ==============================================================================
    # Inference Engine Factory Methods
    # ==============================================================================

    @staticmethod
    def create_workflow_executor_for_engine(
        config,
        inference_engine,
    ):
        """
        Factory method for creating WorkflowExecutor for an engine.

        This handles the complex wiring of WorkflowExecutor with shared
        AsyncTaskRunner and engine-specific caches.

        Parameters
        ----------
        config : InferenceEngineConfig
            Configuration for the workflow executor
        inference_engine : InferenceEngine
            The engine that will use this executor

        Returns
        -------
        WorkflowExecutor
            Fully configured executor with dependencies injected
        """

        # Get singleton runner
        runner = app_container.async_task_runner()

        # Initialize runner if needed
        if not hasattr(runner, "thread") or runner.thread is None:
            # Get logger from engine if available
            logger = getattr(inference_engine, "logger", None)
            runner.initialize(logger=logger)

        # Create new caches for this executor
        pending_results = app_container.workflow_pending_results()
        pending_inputs = app_container.workflow_pending_inputs()

        # Create workflow executor
        return WorkflowExecutor(
            config=config,
            inference_engine=inference_engine,
            runner=runner,
            pending_results=pending_results,
            pending_inputs=pending_inputs,
        )

    @staticmethod
    def create_remote_inf_engine(config, backend, workflow_executor=None):
        """
        Factory method for creating fully-wired RemoteInfEngine.

        This handles the circular dependency between engine and workflow_executor
        by creating them in two phases and wiring them together.

        Parameters
        ----------
        config : InferenceEngineConfig
            Configuration for the engine
        backend : RemoteInfBackendProtocol
            Backend implementation for remote server communication
        workflow_executor : WorkflowExecutor, optional
            Pre-created workflow executor. If None, creates one automatically.

        Returns
        -------
        RemoteInfEngine
            Fully configured engine with workflow_executor injected

        Examples
        --------
        >>> from areal.core.app_container import app_container
        >>> from areal.experimental.sglang_backend import SGLangBackend
        >>>
        >>> backend = SGLangBackend()
        >>> engine = app_container.create_remote_inf_engine(config, backend)
        >>> engine.initialize()
        """
        from areal.core.remote_inf_engine import RemoteInfEngine

        # Phase 1: Create engine without workflow_executor
        engine = RemoteInfEngine(
            config=config,
            backend=backend,
            workflow_executor=workflow_executor,
        )

        # Phase 2: Create workflow_executor if not provided
        if workflow_executor is None:
            engine.workflow_executor = (
                app_container.create_workflow_executor_for_engine(
                    config=config,
                    inference_engine=engine,
                )
            )

        return engine

    @staticmethod
    def create_sglang_engine(config, engine_args=None, workflow_executor=None):
        """
        Factory method for creating fully-wired SGLangEngine.

        Parameters
        ----------
        config : InferenceEngineConfig
            Configuration for the engine
        engine_args : dict, optional
            Arguments to pass to SGLang engine
        workflow_executor : WorkflowExecutor, optional
            Pre-created workflow executor. If None, creates one automatically.

        Returns
        -------
        SGLangEngine
            Fully configured engine with workflow_executor injected

        Examples
        --------
        >>> from areal.core.app_container import app_container
        >>>
        >>> engine = app_container.create_sglang_engine(
        ...     config=config,
        ...     engine_args={'model_path': '/path/to/model'}
        ... )
        >>> engine.initialize()
        """
        from areal.experimental.sglang_engine import SGLangEngine

        # Phase 1: Create engine without workflow_executor
        engine = SGLangEngine(
            config=config,
            engine_args=engine_args,
            workflow_executor=workflow_executor,
        )

        # Phase 2: Create workflow_executor if not provided
        if workflow_executor is None:
            engine.workflow_executor = (
                app_container.create_workflow_executor_for_engine(
                    config=config,
                    inference_engine=engine,
                )
            )

        return engine


# ==============================================================================
# Global Container Instance
# ==============================================================================

app_container = ApplicationContainer()
"""
Global application container instance.

Examples
--------
>>> from areal.core.app_container import app_container
>>> app_container.config.from_dict({'max_queue_size': 5000})
>>> runner = app_container.async_task_runner()
>>> runner.initialize()
"""

"""Custom dependency injection providers for AReaL infrastructure.

This module provides specialized DI providers that handle AReaL's specific
initialization patterns, particularly the common two-phase initialization
pattern (__init__ + initialize()).
"""

import logging
from typing import Any

try:
    from dependency_injector import providers
except ImportError:
    raise ImportError(
        "dependency-injector is required. Install with: pip install dependency-injector"
    )

logger = logging.getLogger(__name__)


class InitializableProvider(providers.Singleton):
    """
    Singleton provider that automatically calls initialize() after construction.

    AReaL components (InferenceEngine, WorkflowExecutor, etc.) follow a two-phase
    initialization pattern:
    - Phase 1: __init__() - Lightweight construction, store config
    - Phase 2: initialize() - Heavy operations (GPU, network, threads)

    This provider handles both phases automatically on first access:
    1. Calls __init__() via standard Singleton behavior
    2. Calls initialize() with provided arguments
    3. Caches the fully initialized instance

    Parameters
    ----------
    provides : type
        The class to instantiate (must have an initialize() method)
    *args, **kwargs
        Arguments passed to __init__()
    init_method : str, default 'initialize'
        Name of the initialization method to call after construction
    init_args : tuple, optional
        Positional arguments for the init method
    init_kwargs : dict, optional
        Keyword arguments for the init method

    Examples
    --------
    >>> # Without InitializableProvider (manual two-phase init)
    >>> engine = RemoteSGLangEngine(config=config)
    >>> engine.initialize(train_data_parallel_size=4)

    >>> # With InitializableProvider (automatic)
    >>> from dependency_injector import containers, providers
    >>> class Container(containers.DeclarativeContainer):
    ...     config = providers.Configuration()
    ...
    ...     engine = InitializableProvider(
    ...         RemoteSGLangEngine,
    ...         config=config.inference_config,
    ...         init_kwargs={'train_data_parallel_size': config.dp_size}
    ...     )
    >>>
    >>> engine = container.engine()  # Fully initialized!
    >>> engine.submit(...)  # Ready to use immediately

    Notes
    -----
    - Thread-safe via Singleton base class
    - initialize() is called exactly once, even with concurrent access
    - Works with any class that has the specified init_method
    - Polymorphic - works with multiple implementation types
    """

    def __init__(
        self,
        provides=None,
        *args,
        init_method: str = "initialize",
        init_args: tuple | None = None,
        init_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the provider.

        Parameters
        ----------
        provides : type, optional
            Class to instantiate
        *args
            Positional arguments for __init__()
        init_method : str, default 'initialize'
            Name of initialization method
        init_args : tuple, optional
            Positional arguments for init method
        init_kwargs : dict, optional
            Keyword arguments for init method
        **kwargs
            Keyword arguments for __init__()
        """
        if provides is not None:
            super().__init__(provides, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self._init_method = init_method
        self._init_args = init_args or ()
        self._init_kwargs = init_kwargs or {}
        self._initialized = False

    def _provide(self, args, kwargs):
        """
        Override to add two-phase initialization.

        This method is called by the Singleton provider when the instance
        is first requested. It performs both construction and initialization.
        """
        # Phase 1: Call __init__() via parent Singleton provider
        instance = super()._provide(args, kwargs)

        # Phase 2: Call initialize() once on first access
        if not self._initialized:
            if hasattr(instance, self._init_method):
                init_method = getattr(instance, self._init_method)

                try:
                    logger.debug(
                        f"Initializing {instance.__class__.__name__} via "
                        f"{self._init_method}() with args={self._init_args}, "
                        f"kwargs={self._init_kwargs}"
                    )

                    # Call initialize() with provided arguments
                    init_method(*self._init_args, **self._init_kwargs)

                    self._initialized = True

                    logger.info(
                        f"Successfully initialized {instance.__class__.__name__}"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to initialize {instance.__class__.__name__} "
                        f"via {self._init_method}(): {e}"
                    )
                    raise

            else:
                raise AttributeError(
                    f"{instance.__class__.__name__} does not have method "
                    f"'{self._init_method}'. Cannot use InitializableProvider."
                )

        return instance

    def reset(self):
        """
        Reset the provider to allow reinitialization.

        This clears the cached instance and resets the initialization flag.
        Useful for testing or when you need to recreate the instance with
        different initialization parameters.

        Examples
        --------
        >>> provider = InitializableProvider(MyClass, ...)
        >>> instance1 = provider()
        >>> provider.reset()
        >>> instance2 = provider()  # New instance created
        >>> instance1 is not instance2
        True
        """
        super().reset()
        self._initialized = False

    def __deepcopy__(self, memo):
        """
        Properly handle deepcopy for dependency-injector container creation.

        This ensures our custom attributes are preserved during copy.
        """
        copied = super().__deepcopy__(memo)
        copied._init_method = self._init_method
        copied._init_args = self._init_args
        copied._init_kwargs = self._init_kwargs
        copied._initialized = False
        return copied


class ThreadSafeInitializableProvider(providers.ThreadSafeSingleton):
    """
    Thread-safe version of InitializableProvider.

    Use this when the instance will be accessed from multiple threads
    and you need guaranteed thread-safe initialization.

    Behavior is identical to InitializableProvider but with additional
    thread-safety guarantees provided by ThreadSafeSingleton.

    Parameters
    ----------
    Same as InitializableProvider

    Examples
    --------
    >>> # For multi-threaded applications
    >>> engine = ThreadSafeInitializableProvider(
    ...     RemoteSGLangEngine,
    ...     config=config,
    ...     init_kwargs={'train_data_parallel_size': 4}
    ... )
    >>>
    >>> # Safe to access from multiple threads
    >>> import threading
    >>> def worker():
    ...     engine_instance = container.engine()  # Thread-safe!
    >>> threads = [threading.Thread(target=worker) for _ in range(10)]
    >>> for t in threads: t.start()
    >>> for t in threads: t.join()
    """

    def __init__(
        self,
        provides=None,
        *args,
        init_method: str = "initialize",
        init_args: tuple | None = None,
        init_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize thread-safe provider."""
        if provides is not None:
            super().__init__(provides, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self._init_method = init_method
        self._init_args = init_args or ()
        self._init_kwargs = init_kwargs or {}
        self._initialized = False
        import threading

        self._init_lock = threading.Lock()

    def _provide(self, args, kwargs):
        """Thread-safe provide with two-phase initialization."""
        # Phase 1: Call __init__() via parent ThreadSafeSingleton provider
        instance = super()._provide(args, kwargs)

        # Phase 2: Call initialize() once (thread-safe with dedicated lock)
        with self._init_lock:
            if not self._initialized:
                if hasattr(instance, self._init_method):
                    init_method = getattr(instance, self._init_method)

                    try:
                        logger.debug(
                            f"[Thread-safe] Initializing {instance.__class__.__name__} "
                            f"via {self._init_method}()"
                        )

                        init_method(*self._init_args, **self._init_kwargs)
                        self._initialized = True

                        logger.info(
                            f"[Thread-safe] Successfully initialized "
                            f"{instance.__class__.__name__}"
                        )

                    except Exception as e:
                        logger.error(
                            f"[Thread-safe] Failed to initialize "
                            f"{instance.__class__.__name__}: {e}"
                        )
                        raise

                else:
                    raise AttributeError(
                        f"{instance.__class__.__name__} does not have method "
                        f"'{self._init_method}'"
                    )

        return instance

    def reset(self):
        """Reset the provider and initialization flag."""
        super().reset()
        self._initialized = False

    def __deepcopy__(self, memo):
        """Properly handle deepcopy for dependency-injector container creation."""
        copied = super().__deepcopy__(memo)
        copied._init_method = self._init_method
        copied._init_args = self._init_args
        copied._init_kwargs = self._init_kwargs
        copied._initialized = False
        return copied

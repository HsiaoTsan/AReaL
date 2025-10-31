"""Queue abstraction with filter support and pluggable backends.

This module provides a queue.Queue compatible interface with additional features:
- Filter support: Accept/reject items during put() operations
- Pluggable backends: Easy to swap from local to distributed queues
- Event integration: Optional event firing for queue operations

Key Design Decisions:
- Phase 1: Uses stdlib queue.Queue (zero external dependencies)
- Phase 2: Can swap to Kombu/Redis for distributed queues
- Filters run on put() operations (not consumer-side)
- Thread-safe with explicit locking
"""

import logging
import queue
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FilterableQueue(Generic[T]):
    """
    Thread-safe queue with filter support and pluggable backends.

    This queue is compatible with queue.Queue but adds filter support.
    Filters are functions that accept/reject items during put() operations.

    Parameters
    ----------
    name : str, default 'default'
        Queue name for identification and logging
    maxsize : int, default 0
        Maximum queue size. 0 means unlimited.
    backend : {'memory', 'redis://...', 'amqp://...'}, default 'memory'
        Queue backend. Currently only 'memory' (stdlib queue.Queue) is implemented.
        Other backends are reserved for Phase 2 distributed support.
    filters : list of callable, optional
        List of filter functions. Each takes an item and returns bool.
        Item is rejected if any filter returns False.
    fire_events : bool, default False
        If True, fire events for queue operations (requires event bus initialization)

    Examples
    --------
    >>> # Basic usage (like queue.Queue)
    >>> q = FilterableQueue(maxsize=10)
    >>> q.put("hello")
    >>> item = q.get()

    >>> # With filters
    >>> def no_empty_strings(item):
    ...     return len(item) > 0
    >>> q = FilterableQueue(filters=[no_empty_strings])
    >>> q.put("")  # Returns False (filtered out)
    False
    >>> q.put("hello")  # Returns True (accepted)
    True

    >>> # Add filters dynamically
    >>> q.add_filter(lambda x: x > 0)
    """

    def __init__(
        self,
        name: str = "default",
        maxsize: int = 0,
        backend: str = "memory",
        filters: list[Callable[[T], bool]] | None = None,
        fire_events: bool = False,
    ):
        """Initialize filterable queue."""
        self.name = name
        self.maxsize = maxsize
        self.backend = backend
        self.filters = filters or []
        self.fire_events = fire_events
        self._lock = threading.RLock()

        # Create backend queue
        if backend == "memory":
            # Phase 1: Use stdlib queue.Queue
            self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
            self._backend_type = "stdlib"
        elif backend.startswith("redis://"):
            # Phase 2: Kombu with Redis backend
            raise NotImplementedError(
                f"Redis backend not yet implemented. Use backend='memory' for now. "
                f"Requested backend: {backend}"
            )
        elif backend.startswith("amqp://"):
            # Phase 2: Kombu with RabbitMQ backend
            raise NotImplementedError(
                f"AMQP backend not yet implemented. Use backend='memory' for now. "
                f"Requested backend: {backend}"
            )
        else:
            raise ValueError(
                f"Invalid backend: {backend}. "
                f"Supported: 'memory', 'redis://...', 'amqp://...'"
            )

    def put(self, item: T, block: bool = True, timeout: float | None = None) -> bool:
        """
        Put item into queue with filter support.

        Unlike queue.Queue.put(), this returns a boolean indicating whether
        the item was accepted or filtered out.

        Parameters
        ----------
        item : T
            Item to add to queue
        block : bool, default True
            If True, block if queue is full
        timeout : float, optional
            Timeout in seconds for blocking put

        Returns
        -------
        bool
            True if item was accepted and added to queue,
            False if item was rejected by filters.

        Raises
        ------
        queue.Full
            If queue is full and block=False, or timeout exceeded
        """
        with self._lock:
            # Apply filters
            for filter_fn in self.filters:
                try:
                    if not filter_fn(item):
                        # Item rejected by filter
                        logger.debug(
                            f"Queue '{self.name}': Item filtered out by {filter_fn}"
                        )

                        # Fire filter event
                        if self.fire_events:
                            self._fire_event(
                                "item-filtered", item=item, filter=filter_fn
                            )

                        return False
                except Exception as e:
                    logger.error(
                        f"Queue '{self.name}': Filter {filter_fn} raised exception: {e}. "
                        f"Rejecting item."
                    )
                    return False

            # All filters passed - add to queue
            self._queue.put(item, block=block, timeout=timeout)

            # Fire added event
            if self.fire_events:
                self._fire_event("item-added", item=item)

            return True

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """
        Remove and return an item from the queue.

        Parameters
        ----------
        block : bool, default True
            If True, block if queue is empty
        timeout : float, optional
            Timeout in seconds for blocking get

        Returns
        -------
        T
            Item from queue

        Raises
        ------
        queue.Empty
            If queue is empty and block=False, or timeout exceeded
        """
        with self._lock:
            item = self._queue.get(block=block, timeout=timeout)

            # Fire removed event
            if self.fire_events:
                self._fire_event("item-removed", item=item)

            return item

    def put_nowait(self, item: T) -> bool:
        """
        Put item into queue without blocking.

        Equivalent to put(item, block=False).

        Parameters
        ----------
        item : T
            Item to add to queue

        Returns
        -------
        bool
            True if item was accepted and added to queue,
            False if item was rejected by filters.

        Raises
        ------
        queue.Full
            If queue is full
        """
        return self.put(item, block=False)

    def get_nowait(self) -> T:
        """
        Remove and return an item from the queue without blocking.

        Equivalent to get(block=False).

        Returns
        -------
        T
            Item from queue

        Raises
        ------
        queue.Empty
            If queue is empty
        """
        return self.get(block=False)

    def qsize(self) -> int:
        """
        Return approximate size of queue.

        Returns
        -------
        int
            Number of items in queue (approximate)

        Notes
        -----
        qsize() is approximate and not reliable in multi-threaded contexts.
        """
        return self._queue.qsize()

    def empty(self) -> bool:
        """
        Return True if queue is empty.

        Returns
        -------
        bool
            True if queue is empty, False otherwise

        Notes
        -----
        empty() is not reliable in multi-threaded contexts.
        """
        return self._queue.empty()

    def full(self) -> bool:
        """
        Return True if queue is full.

        Returns
        -------
        bool
            True if queue is full, False otherwise

        Notes
        -----
        full() is not reliable in multi-threaded contexts.
        """
        return self._queue.full()

    def add_filter(self, filter_fn: Callable[[T], bool]):
        """
        Add a filter to the queue.

        Filters are applied in the order they were added.

        Parameters
        ----------
        filter_fn : callable
            Function that takes an item and returns bool.
            Item is rejected if function returns False.

        Examples
        --------
        >>> q = FilterableQueue()
        >>> q.add_filter(lambda x: x > 0)
        >>> q.add_filter(lambda x: x < 100)
        >>> q.put(-1)  # False (rejected by first filter)
        False
        >>> q.put(50)   # True (passes both filters)
        True
        >>> q.put(200)  # False (rejected by second filter)
        False
        """
        with self._lock:
            self.filters.append(filter_fn)
            logger.debug(f"Queue '{self.name}': Added filter {filter_fn}")

    def remove_filter(self, filter_fn: Callable[[T], bool]):
        """
        Remove a filter from the queue.

        Parameters
        ----------
        filter_fn : callable
            Filter function to remove

        Raises
        ------
        ValueError
            If filter is not in the list
        """
        with self._lock:
            self.filters.remove(filter_fn)
            logger.debug(f"Queue '{self.name}': Removed filter {filter_fn}")

    def clear_filters(self):
        """Remove all filters from the queue."""
        with self._lock:
            self.filters.clear()
            logger.debug(f"Queue '{self.name}': Cleared all filters")

    def _fire_event(self, event_type: str, **kwargs):
        """Fire event for queue operation (if event bus is initialized)."""
        try:
            from .events import QueueEvents, get_event_bus

            bus = get_event_bus()

            if event_type == "item-added":
                bus.send(
                    QueueEvents.ITEM_ADDED, sender=self, queue_name=self.name, **kwargs
                )
            elif event_type == "item-removed":
                bus.send(
                    QueueEvents.ITEM_REMOVED,
                    sender=self,
                    queue_name=self.name,
                    **kwargs,
                )
            elif event_type == "item-filtered":
                bus.send(
                    QueueEvents.ITEM_FILTERED,
                    sender=self,
                    queue_name=self.name,
                    **kwargs,
                )

        except RuntimeError:
            # Event bus not initialized - skip events
            pass
        except Exception as e:
            logger.error(f"Failed to fire event '{event_type}': {e}")

    def close(self):
        """
        Close the queue and cleanup resources.

        For stdlib queue.Queue, this is a no-op. For distributed backends
        (Phase 2), this will close connections.
        """
        if self._backend_type == "stdlib":
            # No cleanup needed for stdlib queue
            pass
        else:
            # Future: Close Kombu connections
            pass

    def __repr__(self) -> str:
        return (
            f"FilterableQueue(name='{self.name}', maxsize={self.maxsize}, "
            f"backend='{self.backend}', filters={len(self.filters)})"
        )

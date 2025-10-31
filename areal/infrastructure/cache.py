"""Cache abstraction with list-compatible interface and pluggable backends.

This module provides a Protocol-based cache interface that is compatible with
Python's built-in list[] but adds:
- Thread-safe operations
- Event integration
- Pluggable backends (in-memory, Redis, etc.)

Key Design Decisions:
- Phase 1: ListCache wraps Python list with thread-safety
- Phase 2: Add Redis/distributed cache backends
- Protocol-based for duck typing (no inheritance required)
- 100% compatible with existing list[] operations
"""

import logging
import threading
from collections.abc import Iterator
from typing import Generic, Protocol, TypeVar, overload

logger = logging.getLogger(__name__)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class Cache(Protocol[T_co]):
    """
    Protocol defining cache interface compatible with list operations.

    This protocol ensures that any cache implementation can be used as a
    drop-in replacement for Python's list[]. It uses structural subtyping
    (duck typing) rather than inheritance.

    Implementations must support:
    - append/extend for adding items
    - indexing and slicing for retrieval
    - len/iter for inspection
    - clear for cleanup
    """

    def append(self, item: T_co) -> None:
        """Add item to end of cache."""
        ...

    def extend(self, items: list[T_co]) -> None:
        """Extend cache with multiple items."""
        ...

    def __getitem__(self, key: int | slice) -> T_co | list[T_co]:
        """Get item(s) by index or slice."""
        ...

    def __setitem__(self, key: int | slice, value: T_co | list[T_co]) -> None:
        """Set item(s) by index or slice."""
        ...

    def __delitem__(self, key: int | slice) -> None:
        """Delete item(s) by index or slice."""
        ...

    def __len__(self) -> int:
        """Return number of items in cache."""
        ...

    def __iter__(self) -> Iterator[T_co]:
        """Iterate over items in cache."""
        ...

    def clear(self) -> None:
        """Remove all items from cache."""
        ...


class ListCache(Generic[T]):
    """
    Thread-safe list-based cache implementation.

    This cache wraps a Python list with thread-safe operations and optional
    event firing. It is 100% compatible with list[] operations.

    Parameters
    ----------
    initial : list, optional
        Initial items for the cache
    fire_events : bool, default False
        If True, fire events for cache operations

    Examples
    --------
    >>> # Use like a list
    >>> cache = ListCache()
    >>> cache.append("hello")
    >>> cache.extend(["world", "!"])
    >>> len(cache)
    3
    >>> cache[0]
    'hello'
    >>> cache[1:3]
    ['world', '!']

    >>> # Thread-safe operations
    >>> import threading
    >>> def worker():
    ...     for i in range(100):
    ...         cache.append(i)
    >>> threads = [threading.Thread(target=worker) for _ in range(10)]
    >>> for t in threads: t.start()
    >>> for t in threads: t.join()
    >>> len(cache)  # 1000 (no race conditions)
    1000
    """

    def __init__(
        self,
        initial: list[T] | None = None,
        fire_events: bool = False,
    ):
        """Initialize list-based cache."""
        self._data: list[T] = initial or []
        self._lock = threading.RLock()
        self.fire_events = fire_events

    def append(self, item: T) -> None:
        """
        Append item to end of cache.

        Parameters
        ----------
        item : T
            Item to append

        Examples
        --------
        >>> cache = ListCache()
        >>> cache.append(42)
        >>> len(cache)
        1
        """
        with self._lock:
            self._data.append(item)

            if self.fire_events:
                self._fire_event("item-added", item=item)

    def extend(self, items: list[T]) -> None:
        """
        Extend cache with multiple items.

        Parameters
        ----------
        items : list
            Items to add

        Examples
        --------
        >>> cache = ListCache([1, 2])
        >>> cache.extend([3, 4, 5])
        >>> len(cache)
        5
        """
        with self._lock:
            self._data.extend(items)

            if self.fire_events:
                for item in items:
                    self._fire_event("item-added", item=item)

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> list[T]: ...

    def __getitem__(self, key: int | slice) -> T | list[T]:
        """
        Get item(s) by index or slice.

        Parameters
        ----------
        key : int or slice
            Index or slice

        Returns
        -------
        T or list[T]
            Item or list of items

        Examples
        --------
        >>> cache = ListCache([1, 2, 3, 4, 5])
        >>> cache[0]
        1
        >>> cache[1:4]
        [2, 3, 4]
        >>> cache[-1]
        5
        """
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: int | slice, value: T | list[T]) -> None:
        """
        Set item(s) by index or slice.

        Parameters
        ----------
        key : int or slice
            Index or slice
        value : T or list[T]
            Value(s) to set

        Examples
        --------
        >>> cache = ListCache([1, 2, 3])
        >>> cache[0] = 10
        >>> cache[1:3] = [20, 30]
        >>> list(cache)
        [10, 20, 30]
        """
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: int | slice) -> None:
        """
        Delete item(s) by index or slice.

        Parameters
        ----------
        key : int or slice
            Index or slice to delete

        Examples
        --------
        >>> cache = ListCache([1, 2, 3, 4, 5])
        >>> del cache[0]
        >>> del cache[1:3]
        >>> list(cache)
        [2, 5]
        """
        with self._lock:
            del self._data[key]

            if self.fire_events:
                self._fire_event("item-removed")

    def __len__(self) -> int:
        """
        Return number of items in cache.

        Returns
        -------
        int
            Number of items

        Examples
        --------
        >>> cache = ListCache([1, 2, 3])
        >>> len(cache)
        3
        """
        with self._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[T]:
        """
        Iterate over items in cache.

        Returns a copy to ensure thread safety during iteration.

        Returns
        -------
        Iterator[T]
            Iterator over cached items

        Examples
        --------
        >>> cache = ListCache([1, 2, 3])
        >>> for item in cache:
        ...     print(item)
        1
        2
        3
        """
        with self._lock:
            # Return iterator over a copy for thread safety
            return iter(self._data[:])

    def clear(self) -> None:
        """
        Remove all items from cache.

        Examples
        --------
        >>> cache = ListCache([1, 2, 3])
        >>> cache.clear()
        >>> len(cache)
        0
        """
        with self._lock:
            self._data.clear()

            if self.fire_events:
                self._fire_event("cache-cleared")

    def sort(self, *args, **kwargs):
        """
        Sort items in cache in-place.

        Accepts same arguments as list.sort().

        Examples
        --------
        >>> cache = ListCache([3, 1, 2])
        >>> cache.sort()
        >>> list(cache)
        [1, 2, 3]
        >>> cache.sort(reverse=True)
        >>> list(cache)
        [3, 2, 1]
        """
        with self._lock:
            self._data.sort(*args, **kwargs)

    def remove(self, item: T) -> None:
        """
        Remove first occurrence of item.

        Parameters
        ----------
        item : T
            Item to remove

        Raises
        ------
        ValueError
            If item is not in cache

        Examples
        --------
        >>> cache = ListCache([1, 2, 3, 2])
        >>> cache.remove(2)
        >>> list(cache)
        [1, 3, 2]
        """
        with self._lock:
            self._data.remove(item)

            if self.fire_events:
                self._fire_event("item-removed", item=item)

    def pop(self, index: int = -1) -> T:
        """
        Remove and return item at index.

        Parameters
        ----------
        index : int, default -1
            Index of item to remove

        Returns
        -------
        T
            Removed item

        Examples
        --------
        >>> cache = ListCache([1, 2, 3])
        >>> cache.pop()
        3
        >>> cache.pop(0)
        1
        >>> list(cache)
        [2]
        """
        with self._lock:
            item = self._data.pop(index)

            if self.fire_events:
                self._fire_event("item-removed", item=item)

            return item

    def insert(self, index: int, item: T) -> None:
        """
        Insert item at index.

        Parameters
        ----------
        index : int
            Index where item should be inserted
        item : T
            Item to insert

        Examples
        --------
        >>> cache = ListCache([1, 3])
        >>> cache.insert(1, 2)
        >>> list(cache)
        [1, 2, 3]
        """
        with self._lock:
            self._data.insert(index, item)

            if self.fire_events:
                self._fire_event("item-added", item=item)

    def copy(self) -> "ListCache[T]":
        """
        Create a shallow copy of the cache.

        Returns
        -------
        ListCache[T]
            A new cache with copied items

        Examples
        --------
        >>> cache = ListCache([1, 2, 3])
        >>> cache_copy = cache.copy()
        >>> cache_copy[0] = 10
        >>> cache[0]
        1
        """
        with self._lock:
            return ListCache(self._data[:], fire_events=self.fire_events)

    def _fire_event(self, event_type: str, **kwargs):
        """Fire event for cache operation (if event bus is initialized)."""
        try:
            from .events import CacheEvents, get_event_bus

            bus = get_event_bus()

            if event_type == "item-added":
                bus.send(CacheEvents.ITEM_ADDED, sender=self, **kwargs)
            elif event_type == "item-removed":
                bus.send(CacheEvents.ITEM_REMOVED, sender=self, **kwargs)
            elif event_type == "cache-cleared":
                bus.send(CacheEvents.CACHE_CLEARED, sender=self, **kwargs)

        except RuntimeError:
            # Event bus not initialized - skip events
            pass
        except Exception as e:
            logger.error(f"Failed to fire cache event '{event_type}': {e}")

    def __repr__(self) -> str:
        with self._lock:
            return f"ListCache({self._data!r})"

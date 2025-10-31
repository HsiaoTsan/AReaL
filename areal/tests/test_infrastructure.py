"""Unit tests for AReaL infrastructure components.

Tests cover:
- Event bus (local events, synchronous dispatch)
- Filterable queues
- List-compatible caches
- Two-phase initialization provider
- Dependency injection container
- Component integration
"""

import threading
import time
from dataclasses import dataclass

import pytest

# ==============================================================================
# Test Event System
# ==============================================================================


class TestEventBus:
    """Tests for EventBus."""

    def test_event_bus_initialization(self):
        """Test event bus can be initialized."""
        from areal.infrastructure.events import EventBus

        bus = EventBus(mode="local")
        assert bus.mode == "local"

    def test_connect_and_send(self):
        """Test connecting handler and sending events."""
        from areal.infrastructure.events import EventBus

        bus = EventBus()
        handler_called = []

        def handler(sender, **kwargs):
            handler_called.append(kwargs)

        bus.connect("test-event", handler)
        bus.send("test-event", sender=None, data="hello")

        assert len(handler_called) == 1
        assert handler_called[0]["data"] == "hello"

    def test_synchronous_dispatch(self):
        """Test events are dispatched synchronously (blocks until complete)."""
        from areal.infrastructure.events import EventBus

        bus = EventBus()
        order = []

        def slow_handler(sender, **kwargs):
            time.sleep(0.1)
            order.append("handler")

        bus.connect("test-event", slow_handler)

        order.append("before_send")
        bus.send("test-event", sender=None)
        order.append("after_send")

        # Handler must complete before send() returns
        assert order == ["before_send", "handler", "after_send"]

    def test_multiple_handlers(self):
        """Test multiple handlers receive same event."""
        from areal.infrastructure.events import EventBus

        bus = EventBus()
        calls = []

        bus.connect("test-event", lambda sender, **kw: calls.append("h1"))
        bus.connect("test-event", lambda sender, **kw: calls.append("h2"))
        bus.connect("test-event", lambda sender, **kw: calls.append("h3"))

        bus.send("test-event", sender=None)

        # All handlers should be called (order not guaranteed by blinker)
        assert len(calls) == 3
        assert set(calls) == {"h1", "h2", "h3"}

    def test_sender_filtering(self):
        """Test handlers can filter by sender."""
        from areal.infrastructure.events import EventBus

        bus = EventBus()
        calls = []

        sender_a = object()
        sender_b = object()

        bus.connect("test-event", lambda s, **kw: calls.append("any"))
        bus.connect("test-event", lambda s, **kw: calls.append("a"), sender=sender_a)

        bus.send("test-event", sender=sender_a)
        assert calls == ["any", "a"]

        calls.clear()
        bus.send("test-event", sender=sender_b)
        assert calls == ["any"]  # sender_a handler not called

    def test_disconnect(self):
        """Test disconnecting handlers."""
        from areal.infrastructure.events import EventBus

        bus = EventBus()
        calls = []

        def handler1(s, **kw):
            calls.append("h1")

        def handler2(s, **kw):
            calls.append("h2")

        bus.connect("test-event", handler1)
        bus.connect("test-event", handler2)

        # Both handlers connected
        bus.send("test-event", sender=None)
        assert calls == ["h1", "h2"]

        # Disconnect one
        calls.clear()
        bus.disconnect("test-event", handler1)
        bus.send("test-event", sender=None)
        assert calls == ["h2"]

    def test_disconnect_with_sender(self):
        """Test disconnecting handler with sender filter."""
        from areal.infrastructure.events import EventBus

        bus = EventBus()
        calls = []
        sender = object()

        def handler(s, **kw):
            calls.append("h")

        bus.connect("test-event", handler, sender=sender)
        bus.send("test-event", sender=sender)
        assert len(calls) == 1

        # Disconnect with sender
        calls.clear()
        bus.disconnect("test-event", handler, sender=sender)
        bus.send("test-event", sender=sender)
        assert len(calls) == 0

    def test_has_receivers(self):
        """Test checking if event has receivers."""
        from areal.infrastructure.events import EventBus

        bus = EventBus()

        # No receivers initially
        assert bus.has_receivers("test-event") is False

        # Add receiver
        def handler(s, **kw):
            pass

        bus.connect("test-event", handler)
        assert bus.has_receivers("test-event") is True

        # Remove receiver
        bus.disconnect("test-event", handler)
        assert bus.has_receivers("test-event") is False

    def test_send_async_not_implemented(self):
        """Test send_async raises NotImplementedError."""
        import pytest

        from areal.infrastructure.events import EventBus

        bus = EventBus()

        with pytest.raises(
            NotImplementedError, match="Async event handlers not yet implemented"
        ):
            import asyncio

            asyncio.run(bus.send_async("test-event", sender=None))

    def test_invalid_mode(self):
        """Test invalid mode raises ValueError."""
        import pytest

        from areal.infrastructure.events import EventBus

        with pytest.raises(
            ValueError, match="Invalid mode.*Must be 'local' or 'distributed'"
        ):
            EventBus(mode="invalid")

    def test_distributed_mode_warning(self):
        """Test distributed mode logs warning."""
        from areal.infrastructure.events import EventBus

        # Distributed mode should create bus with warning logged
        bus = EventBus(mode="distributed")
        assert bus.mode == "distributed"


class TestEventBusModuleFunctions:
    """Tests for module-level event bus functions."""

    def test_get_event_bus_not_initialized(self):
        """Test get_event_bus raises error when not initialized."""
        import pytest

        # Reset global state
        import areal.infrastructure.events as events_module
        from areal.infrastructure.events import get_event_bus

        old_bus = events_module._event_bus
        events_module._event_bus = None

        try:
            with pytest.raises(RuntimeError, match="Event bus not initialized"):
                get_event_bus()
        finally:
            events_module._event_bus = old_bus

    def test_initialize_and_get_event_bus(self):
        """Test initializing and getting event bus."""
        from areal.infrastructure.events import get_event_bus, initialize_event_bus

        initialize_event_bus(mode="local")
        bus = get_event_bus()

        assert bus is not None
        assert bus.mode == "local"

    def test_event_name_constants(self):
        """Test event name constants are defined."""
        from areal.infrastructure import CacheEvents, QueueEvents, WorkflowEvents

        # QueueEvents
        assert QueueEvents.ITEM_ADDED == "queue-item-added"
        assert QueueEvents.ITEM_REMOVED == "queue-item-removed"
        assert QueueEvents.ITEM_FILTERED == "queue-item-filtered"

        # CacheEvents
        assert CacheEvents.ITEM_ADDED == "cache-item-added"
        assert CacheEvents.ITEM_REMOVED == "cache-item-removed"
        assert CacheEvents.CACHE_CLEARED == "cache-cleared"

        # WorkflowEvents
        assert WorkflowEvents.ROLLOUT_STARTED == "rollout-started"
        assert WorkflowEvents.ROLLOUT_COMPLETED == "rollout-completed"
        assert WorkflowEvents.BATCH_READY == "batch-ready"
        assert WorkflowEvents.PRE_WEIGHT_UPDATE == "pre-weight-update"
        assert WorkflowEvents.POST_WEIGHT_UPDATE == "post-weight-update"
        assert WorkflowEvents.MODEL_VERSION_UPDATED == "model-version-updated"
        assert WorkflowEvents.TRAINING_STEP_COMPLETED == "training-step-completed"
        assert WorkflowEvents.CAPACITY_CHANGED == "capacity-changed"


# ==============================================================================
# Test Filterable Queue
# ==============================================================================


class TestFilterableQueue:
    """Tests for FilterableQueue."""

    def test_basic_queue_operations(self):
        """Test basic put/get operations."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue(maxsize=10)

        assert q.put("item1") is True
        assert q.put("item2") is True
        assert q.qsize() == 2

        assert q.get() == "item1"
        assert q.get() == "item2"
        assert q.empty()

    def test_filter_acceptance(self):
        """Test filters can accept/reject items."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()

        # Add filter: only positive numbers
        q.add_filter(lambda x: x > 0)

        assert q.put(5) is True
        assert q.put(-3) is False  # Filtered out
        assert q.put(10) is True

        assert q.qsize() == 2
        assert q.get() == 5
        assert q.get() == 10

    def test_multiple_filters(self):
        """Test multiple filters (all must pass)."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()

        q.add_filter(lambda x: x > 0)  # Positive
        q.add_filter(lambda x: x < 100)  # Less than 100

        assert q.put(50) is True  # Passes both
        assert q.put(-5) is False  # Fails first
        assert q.put(200) is False  # Fails second
        assert q.put(30) is True  # Passes both

        assert q.qsize() == 2

    def test_thread_safety(self):
        """Test queue is thread-safe."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()

        def worker(worker_id):
            for i in range(100):
                q.put(f"worker_{worker_id}_item_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert q.qsize() == 500  # All items added successfully

    def test_full(self):
        """Test full() method."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue(maxsize=2)
        assert q.full() is False

        q.put("item1")
        assert q.full() is False

        q.put("item2")
        assert q.full() is True

    def test_remove_filter(self):
        """Test removing a specific filter."""
        from areal.infrastructure.queue import FilterableQueue

        def filter1(x):
            return x > 0

        def filter2(x):
            return x < 100

        q = FilterableQueue()
        q.add_filter(filter1)
        q.add_filter(filter2)

        assert q.put(50) is True  # Passes both
        assert q.put(-5) is False  # Fails first

        # Remove first filter
        q.remove_filter(filter1)
        assert q.put(-5) is True  # Now passes (only filter2)
        assert q.put(200) is False  # Still fails filter2

    def test_clear_filters(self):
        """Test clearing all filters."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        q.add_filter(lambda x: x > 0)
        q.add_filter(lambda x: x < 100)

        assert q.put(-5) is False  # Filtered

        q.clear_filters()
        assert q.put(-5) is True  # Now accepted
        assert q.put(200) is True  # Now accepted

    def test_invalid_backend(self):
        """Test invalid backend raises ValueError."""
        import pytest

        from areal.infrastructure.queue import FilterableQueue

        with pytest.raises(ValueError, match="Invalid backend.*Supported"):
            FilterableQueue(backend="invalid")

    def test_redis_backend_not_implemented(self):
        """Test Redis backend raises NotImplementedError."""
        import pytest

        from areal.infrastructure.queue import FilterableQueue

        with pytest.raises(
            NotImplementedError, match="Redis backend not yet implemented"
        ):
            FilterableQueue(backend="redis://localhost:6379")

    def test_amqp_backend_not_implemented(self):
        """Test AMQP backend raises NotImplementedError."""
        import pytest

        from areal.infrastructure.queue import FilterableQueue

        with pytest.raises(
            NotImplementedError, match="AMQP backend not yet implemented"
        ):
            FilterableQueue(backend="amqp://localhost")

    def test_filter_exception_handling(self):
        """Test filter that raises exception is caught and logged."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()

        def bad_filter(x):
            raise ValueError("Filter error")

        q.add_filter(bad_filter)

        # Filter exception should be caught and logged, item rejected
        result = q.put("item")
        assert result is False  # Item rejected due to filter exception

    def test_close(self):
        """Test close method."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        q.put("item")

        # Close should work without error
        q.close()

        # After close, queue operations may still work (depends on backend)
        # For memory backend, operations continue to work

    def test_scan_all_items(self):
        """Test scan returns all items without predicate."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        q.put(1)
        q.put(2)
        q.put(3)

        result = q.scan()
        assert result == [1, 2, 3]
        # Items still in queue
        assert q.qsize() == 3

    def test_scan_with_predicate(self):
        """Test scan filters items with predicate."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        for i in range(10):
            q.put(i)

        # Scan for even numbers
        evens = q.scan(lambda x: x % 2 == 0)
        assert evens == [0, 2, 4, 6, 8]
        # All items still in queue
        assert q.qsize() == 10

    def test_scan_empty_queue(self):
        """Test scan on empty queue."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        result = q.scan()
        assert result == []

    def test_scan_thread_safe(self):
        """Test scan is thread-safe."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        for i in range(100):
            q.put(i)

        results = []

        def scan_worker():
            result = q.scan(lambda x: x % 2 == 0)
            results.append(len(result))

        threads = [threading.Thread(target=scan_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get same result
        assert all(r == 50 for r in results)

    def test_scan_and_update_basic(self):
        """Test scan_and_update modifies items."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        q.put({"value": 1})
        q.put({"value": 2})
        q.put({"value": 3})

        def add_flag(item):
            item["flag"] = True
            return item

        updated = q.scan_and_update(add_flag)
        assert updated == 3

        # Check items were modified
        items = []
        while not q.empty():
            items.append(q.get())
        assert all(item["flag"] is True for item in items)

    def test_scan_and_update_remove_items(self):
        """Test scan_and_update can remove items by returning None."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        for i in range(10):
            q.put(i)

        def keep_evens(item):
            return item if item % 2 == 0 else None

        updated = q.scan_and_update(keep_evens)
        assert updated == 5  # 5 even numbers kept
        assert q.qsize() == 5

        # Check only evens remain
        items = []
        while not q.empty():
            items.append(q.get())
        assert items == [0, 2, 4, 6, 8]

    def test_scan_and_update_with_dict_items(self):
        """Test scan_and_update with dictionary items."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        q.put({"version": 4, "data": "a"})
        q.put({"version": 5, "data": "b"})
        q.put({"version": 4, "data": "c"})

        def update_old_version(item):
            if item["version"] == 4:
                item["updated"] = True
            return item

        updated = q.scan_and_update(update_old_version)
        assert updated == 3

        # Check updates
        items = []
        while not q.empty():
            items.append(q.get())
        assert items[0].get("updated") is True
        assert items[1].get("updated") is None
        assert items[2].get("updated") is True

    def test_scan_and_update_thread_safe(self):
        """Test scan_and_update is thread-safe."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        for i in range(100):
            q.put({"value": i, "processed": False})

        def mark_processed(item):
            item["processed"] = True
            return item

        # Only one thread should process (due to lock)
        threads = [
            threading.Thread(target=lambda: q.scan_and_update(mark_processed))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All items should be marked (last thread wins)
        items = []
        while not q.empty():
            items.append(q.get())
        assert all(item["processed"] for item in items)

    def test_scan_uses_reentrant_lock(self):
        """Test that scan uses RLock (allows reentrancy from same thread)."""
        from areal.infrastructure.queue import FilterableQueue

        q = FilterableQueue()
        q.put(1)

        # This should not deadlock because RLock allows same thread to re-acquire
        def reentrant_predicate(item):
            # Try to acquire lock again (implicitly via scan)
            # This tests that lock is reentrant
            _ = q.qsize()  # qsize might use lock internally
            return True

        result = q.scan(reentrant_predicate)
        assert result == [1]


# ==============================================================================
# Test Cache
# ==============================================================================


class TestListCache:
    """Tests for ListCache."""

    def test_list_operations(self):
        """Test list-compatible operations."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache()

        # Append
        cache.append(1)
        cache.append(2)
        cache.append(3)
        assert len(cache) == 3

        # Indexing
        assert cache[0] == 1
        assert cache[-1] == 3

        # Slicing
        assert cache[1:3] == [2, 3]

        # Extend
        cache.extend([4, 5])
        assert len(cache) == 5

    def test_cache_operations(self):
        """Test cache-specific operations."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3, 4, 5])

        # Sort
        cache.sort(reverse=True)
        assert list(cache) == [5, 4, 3, 2, 1]

        # Remove
        cache.remove(3)
        assert 3 not in list(cache)

        # Pop
        item = cache.pop()
        assert item == 1
        assert len(cache) == 3

        # Clear
        cache.clear()
        assert len(cache) == 0

    def test_thread_safety(self):
        """Test cache is thread-safe."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache()

        def worker():
            for i in range(100):
                cache.append(i)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(cache) == 1000  # All items added

    def test_setitem_by_index(self):
        """Test setting items by index."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3])
        cache[0] = 10
        cache[-1] = 30

        assert cache[0] == 10
        assert cache[2] == 30

    def test_setitem_by_slice(self):
        """Test setting items by slice."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3, 4, 5])
        cache[1:3] = [20, 30]

        assert list(cache) == [1, 20, 30, 4, 5]

    def test_delitem_by_index(self):
        """Test deleting items by index."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3, 4, 5])
        del cache[2]

        assert list(cache) == [1, 2, 4, 5]

    def test_delitem_by_slice(self):
        """Test deleting items by slice."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3, 4, 5])
        del cache[1:3]

        assert list(cache) == [1, 4, 5]

    def test_insert(self):
        """Test inserting items at specific position."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 3, 4])
        cache.insert(1, 2)

        assert list(cache) == [1, 2, 3, 4]

    def test_iterator(self):
        """Test iteration over cache."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3])
        items = [x for x in cache]

        assert items == [1, 2, 3]

    def test_contains(self):
        """Test 'in' operator."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3])

        assert 2 in cache
        assert 5 not in cache

    def test_pop_with_index(self):
        """Test pop with custom index."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3, 4])
        item = cache.pop(1)

        assert item == 2
        assert list(cache) == [1, 3, 4]

    def test_remove_not_found(self):
        """Test remove raises ValueError when item not found."""
        import pytest

        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3])

        with pytest.raises(ValueError):
            cache.remove(999)

    def test_sort_with_key(self):
        """Test sort with key parameter."""
        from areal.infrastructure.cache import ListCache

        @dataclass
        class Item:
            value: int

        cache = ListCache([Item(3), Item(1), Item(2)])
        cache.sort(key=lambda x: x.value)

        assert [x.value for x in cache] == [1, 2, 3]

    def test_copy(self):
        """Test copy method."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3])
        cache_copy = cache.copy()

        assert list(cache_copy) == [1, 2, 3]
        assert cache_copy is not cache

    def test_scan_all_items(self):
        """Test scan returns all items without predicate."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3, 4, 5])
        result = cache.scan()
        assert result == [1, 2, 3, 4, 5]
        # Items still in cache
        assert len(cache) == 5

    def test_scan_with_predicate(self):
        """Test scan filters items with predicate."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache(list(range(10)))
        evens = cache.scan(lambda x: x % 2 == 0)
        assert evens == [0, 2, 4, 6, 8]
        # All items still in cache
        assert len(cache) == 10

    def test_scan_empty_cache(self):
        """Test scan on empty cache."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache()
        result = cache.scan()
        assert result == []

    def test_scan_thread_safe(self):
        """Test scan is thread-safe."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache(list(range(100)))
        results = []

        def scan_worker():
            result = cache.scan(lambda x: x % 2 == 0)
            results.append(len(result))

        threads = [threading.Thread(target=scan_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get same result
        assert all(r == 50 for r in results)

    def test_scan_and_update_basic(self):
        """Test scan_and_update modifies items."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([{"value": 1}, {"value": 2}, {"value": 3}])

        def add_flag(item):
            item["flag"] = True
            return item

        updated = cache.scan_and_update(add_flag)
        assert updated == 3
        assert all(item["flag"] is True for item in cache)

    def test_scan_and_update_remove_items(self):
        """Test scan_and_update can remove items by returning None."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache(list(range(10)))

        def keep_evens(item):
            return item if item % 2 == 0 else None

        updated = cache.scan_and_update(keep_evens)
        assert updated == 5  # 5 even numbers kept
        assert len(cache) == 5
        assert list(cache) == [0, 2, 4, 6, 8]

    def test_scan_and_update_with_dict_items(self):
        """Test scan_and_update with dictionary items."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache(
            [
                {"version": 4, "data": "a"},
                {"version": 5, "data": "b"},
                {"version": 4, "data": "c"},
            ]
        )

        def update_old_version(item):
            if item["version"] == 4:
                item["updated"] = True
            return item

        updated = cache.scan_and_update(update_old_version)
        assert updated == 3
        assert cache[0].get("updated") is True
        assert cache[1].get("updated") is None
        assert cache[2].get("updated") is True

    def test_scan_and_update_thread_safe(self):
        """Test scan_and_update is thread-safe."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([{"value": i, "processed": False} for i in range(100)])

        def mark_processed(item):
            item["processed"] = True
            return item

        threads = [
            threading.Thread(target=lambda: cache.scan_and_update(mark_processed))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All items should be marked
        assert all(item["processed"] for item in cache)

    def test_scan_uses_reentrant_lock(self):
        """Test that scan uses RLock (allows reentrancy from same thread)."""
        from areal.infrastructure.cache import ListCache

        cache = ListCache([1, 2, 3])

        # This should not deadlock because RLock allows same thread to re-acquire
        def reentrant_predicate(item):
            # Try to acquire lock again (implicitly)
            _ = len(cache)  # len might use lock internally
            return True

        result = cache.scan(reentrant_predicate)
        assert len(result) == 3


# ==============================================================================
# Test InitializableProvider
# ==============================================================================


class TestInitializableProvider:
    """Tests for InitializableProvider."""

    def test_two_phase_initialization(self):
        """Test provider calls both __init__ and initialize()."""
        from dependency_injector import containers, providers

        from areal.infrastructure.providers import InitializableProvider

        class MockComponent:
            def __init__(self, config):
                self.config = config
                self.initialized = False

            def initialize(self, dp_size=1):
                self.initialized = True
                self.dp_size = dp_size

        class TestContainer(containers.DeclarativeContainer):
            config = providers.Configuration()

            component = InitializableProvider(
                MockComponent,
                config=config.component_config,
                init_kwargs={"dp_size": 4},
            )

        container = TestContainer()
        container.config.from_dict({"component_config": "test_config"})

        # Get instance - should be fully initialized
        instance = container.component()

        assert instance.config == "test_config"  # __init__ was called
        assert instance.initialized is True  # initialize() was called
        assert instance.dp_size == 4  # init_kwargs passed

    def test_singleton_behavior(self):
        """Test provider returns same instance on multiple calls."""
        from dependency_injector import containers

        from areal.infrastructure.providers import InitializableProvider

        class MockComponent:
            def __init__(self):
                pass

            def initialize(self):
                pass

        class TestContainer(containers.DeclarativeContainer):
            component = InitializableProvider(MockComponent)

        container = TestContainer()

        instance1 = container.component()
        instance2 = container.component()

        assert instance1 is instance2  # Same instance

    def test_reset(self):
        """Test reset clears cached instance."""
        from dependency_injector import containers

        from areal.infrastructure.providers import InitializableProvider

        class MockComponent:
            def __init__(self):
                self.initialized = False

            def initialize(self):
                self.initialized = True

        class TestContainer(containers.DeclarativeContainer):
            component = InitializableProvider(MockComponent)

        container = TestContainer()

        instance1 = container.component()
        container.component.reset()
        instance2 = container.component()

        assert instance1 is not instance2

    def test_custom_init_method(self):
        """Test custom initialization method name."""
        from dependency_injector import containers

        from areal.infrastructure.providers import InitializableProvider

        class MockComponent:
            def __init__(self):
                self.setup_called = False

            def setup(self):
                self.setup_called = True

        class TestContainer(containers.DeclarativeContainer):
            component = InitializableProvider(MockComponent, init_method="setup")

        container = TestContainer()
        instance = container.component()

        assert instance.setup_called is True

    def test_missing_init_method(self):
        """Test missing init method raises AttributeError."""
        import pytest
        from dependency_injector import containers

        from areal.infrastructure.providers import InitializableProvider

        class MockComponent:
            def __init__(self):
                pass

        class TestContainer(containers.DeclarativeContainer):
            component = InitializableProvider(MockComponent, init_method="nonexistent")

        container = TestContainer()

        with pytest.raises(AttributeError, match="does not have method 'nonexistent'"):
            container.component()

    def test_init_method_exception(self):
        """Test exception in init method propagates."""
        import pytest
        from dependency_injector import containers

        from areal.infrastructure.providers import InitializableProvider

        class MockComponent:
            def __init__(self):
                pass

            def initialize(self):
                raise RuntimeError("Init failed")

        class TestContainer(containers.DeclarativeContainer):
            component = InitializableProvider(MockComponent)

        container = TestContainer()

        with pytest.raises(RuntimeError, match="Init failed"):
            container.component()

    def test_init_args(self):
        """Test positional arguments passed to init method."""
        from dependency_injector import containers

        from areal.infrastructure.providers import InitializableProvider

        class MockComponent:
            def __init__(self):
                self.args = None

            def initialize(self, arg1, arg2):
                self.args = (arg1, arg2)

        class TestContainer(containers.DeclarativeContainer):
            component = InitializableProvider(MockComponent, init_args=("a", "b"))

        container = TestContainer()
        instance = container.component()

        assert instance.args == ("a", "b")


class TestThreadSafeInitializableProvider:
    """Tests for ThreadSafeInitializableProvider."""

    def test_thread_safe_initialization(self):
        """Test thread-safe initialization with concurrent access."""
        import threading

        from dependency_injector import containers

        from areal.infrastructure.providers import ThreadSafeInitializableProvider

        call_count = [0]

        class MockComponent:
            def __init__(self):
                pass

            def initialize(self):
                call_count[0] += 1
                time.sleep(0.01)  # Simulate slow initialization

        class TestContainer(containers.DeclarativeContainer):
            component = ThreadSafeInitializableProvider(MockComponent)

        container = TestContainer()
        instances = []

        def worker():
            instances.append(container.component())

        # Multiple threads try to get instance simultaneously
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only initialize once
        assert call_count[0] == 1

        # All threads should get same instance
        assert all(inst is instances[0] for inst in instances)

    def test_thread_safe_reset(self):
        """Test thread-safe reset."""
        from dependency_injector import containers

        from areal.infrastructure.providers import ThreadSafeInitializableProvider

        class MockComponent:
            def __init__(self):
                pass

            def initialize(self):
                pass

        class TestContainer(containers.DeclarativeContainer):
            component = ThreadSafeInitializableProvider(MockComponent)

        container = TestContainer()

        instance1 = container.component()
        container.component.reset()
        instance2 = container.component()

        assert instance1 is not instance2


# ==============================================================================
# Test Container
# ==============================================================================


class TestContainer:
    """Tests for InfrastructureContainer."""

    def test_container_initialization(self):
        """Test container can be created and configured."""
        from areal.infrastructure.container import InfrastructureContainer

        container = InfrastructureContainer()
        container.config.from_dict({"max_queue_size": 500})

        # Get components
        queue = container.task_input_queue()
        cache = container.result_cache()
        bus = container.event_bus()

        assert queue is not None
        assert cache is not None
        assert bus is not None

    def test_factory_vs_singleton(self):
        """Test Factory creates new instances, Singleton reuses."""
        from areal.infrastructure.container import InfrastructureContainer

        container = InfrastructureContainer()

        # Factory - new instance each time
        queue1 = container.task_input_queue()
        queue2 = container.task_input_queue()
        assert queue1 is not queue2

        # Singleton - same instance
        bus1 = container.event_bus()
        bus2 = container.event_bus()
        assert bus1 is bus2

    def test_all_providers(self):
        """Test all container providers can be accessed."""
        from areal.infrastructure.container import InfrastructureContainer

        container = InfrastructureContainer()
        container.config.from_dict(
            {
                "event_bus_mode": "local",
                "max_queue_size": 1000,
                "queue_backend": "memory",
                "fire_queue_events": False,
                "fire_cache_events": False,
            }
        )

        # Test all providers can be instantiated
        assert container.event_bus() is not None
        assert container.task_input_queue() is not None
        assert container.task_output_queue() is not None
        assert container.result_cache() is not None
        assert container.pending_results() is not None
        assert container.pending_inputs() is not None

    def test_queue_configuration(self):
        """Test queue respects configuration."""
        from areal.infrastructure.container import InfrastructureContainer

        container = InfrastructureContainer()
        container.config.from_dict({"max_queue_size": 42})

        queue = container.task_input_queue()
        assert queue.maxsize == 42

    def test_event_bus_configuration(self):
        """Test event bus respects configuration."""
        from areal.infrastructure.container import InfrastructureContainer

        container = InfrastructureContainer()
        container.config.from_dict({"event_bus_mode": "distributed"})

        bus = container.event_bus()
        assert bus.mode == "distributed"


class TestInitialization:
    """Tests for initialize_infrastructure function."""

    def test_initialize_with_defaults(self):
        """Test initialization with no config uses defaults."""
        from areal.infrastructure import get_event_bus, initialize_infrastructure

        returned_container = initialize_infrastructure()

        assert returned_container is not None
        bus = get_event_bus()
        assert bus.mode == "local"

    def test_initialize_with_partial_config(self):
        """Test initialization merges with defaults."""
        from areal.infrastructure import container, initialize_infrastructure

        initialize_infrastructure({"max_queue_size": 999})

        queue = container.task_input_queue()
        assert queue.maxsize == 999
        assert queue.backend == "memory"  # Default

    def test_default_config_values(self):
        """Test default configuration values."""
        from areal.infrastructure import container, initialize_infrastructure

        initialize_infrastructure()

        # Verify defaults
        queue = container.task_input_queue()
        assert queue.maxsize == 10240  # Default
        assert queue.backend == "memory"  # Default

    def test_module_exports(self):
        """Test all public APIs are exported."""
        import areal.infrastructure as infra

        # Event system
        assert hasattr(infra, "EventBus")
        assert hasattr(infra, "initialize_event_bus")
        assert hasattr(infra, "get_event_bus")
        assert hasattr(infra, "QueueEvents")
        assert hasattr(infra, "CacheEvents")
        assert hasattr(infra, "WorkflowEvents")

        # Queues
        assert hasattr(infra, "FilterableQueue")

        # Caches
        assert hasattr(infra, "Cache")
        assert hasattr(infra, "ListCache")

        # DI
        assert hasattr(infra, "InitializableProvider")
        assert hasattr(infra, "ThreadSafeInitializableProvider")
        assert hasattr(infra, "InfrastructureContainer")
        assert hasattr(infra, "container")

        # Initialization
        assert hasattr(infra, "initialize_infrastructure")

    def test_package_metadata(self):
        """Test package metadata is defined."""
        import areal.infrastructure as infra

        assert hasattr(infra, "__version__")
        assert hasattr(infra, "__author__")
        assert hasattr(infra, "__doc_url__")


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for infrastructure components."""

    def test_event_bus_with_container_components(self):
        """Test event bus integrated with queues and caches from container."""
        from areal.infrastructure import (
            container,
            get_event_bus,
            initialize_infrastructure,
        )

        # Initialize
        initialize_infrastructure({"max_queue_size": 100})

        # Get components from container
        bus = get_event_bus()
        queue = container.task_input_queue()
        cache = container.result_cache()

        # Create a simple handler that uses queue and cache
        results = []

        def handler(sender, **kwargs):
            # Handler receives event data
            item = kwargs.get("item")
            results.append(item)
            # Put to queue
            queue.put(f"queued_{item}")
            # Add to cache
            cache.append(f"cached_{item}")

        # Connect handler
        bus.connect("test-event", handler)

        # Send events
        bus.send("test-event", sender=None, item="a")
        bus.send("test-event", sender=None, item="b")

        # Verify handler was called
        assert results == ["a", "b"]

        # Verify queue received items
        assert queue.qsize() == 2
        assert queue.get() == "queued_a"
        assert queue.get() == "queued_b"

        # Verify cache received items
        assert len(cache) == 2
        assert cache[0] == "cached_a"
        assert cache[1] == "cached_b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# AReaL Infrastructure Framework Recommendations

## Executive Summary

Based on analysis of the AReaL codebase (11+ queues, 3+ list caches across async
components), I recommend a minimal-change architecture using mature Python frameworks:

1. **Kombu** - Queue abstraction with distributed backend support
1. **blinker** - Lightweight event system
1. **dependency-injector** - IoC container for clean instantiation
1. **typing.Protocol** - Native Python for cache abstraction

## Current Architecture Analysis

### Queue Usage (11+ instances)

- **AsyncTaskRunner**: `input_queue`, `output_queue` for task execution
- **ModelWorker**: 5 queues (`__request_queue`, `__reply_queue`, `train_step`,
  `inference`, `generate`)
- **StreamDataset**: `data_queue` for ZMQ buffering
- **NameResolve**: 3 internal queues for distributed discovery

### Cache Usage (3+ instances)

- **AsyncTaskRunner**: `result_cache` for completed tasks (Lines 467, 490)
- **WorkflowExecutor**: `_pending_results`, `_pending_inputs` for rollout buffering
  (Lines 523, 531)
- **SGLangEngine**: `result_cache` for response accumulation

______________________________________________________________________

## Recommended Framework Stack

### 1. Queue Abstraction: **Custom Wrapper** (Start) → **Kombu** (Future)

#### Why Custom Wrapper First?

- ✅ **Zero dependencies**: Built on stdlib `queue.Queue`
- ✅ **Native filter support**: Filters run on `put()` operations (Kombu doesn't support
  this)
- ✅ **queue.Queue-compatible**: 100% API compatible
- ✅ **Future-proof**: Easy to swap in Kombu/Redis backend later

#### Why Kombu for Future?

- ✅ **Production-ready**: Used by Celery, battle-tested at scale
- ✅ **Pluggable transports**: Redis, RabbitMQ, Amazon SQS, ZooKeeper
- ⚠️ **Filter limitation**: Kombu filters work at consumer-side (after dequeue), not on
  `put()` operations
- ✅ **Solution**: Wrap Kombu with filter layer (same as stdlib queue)

#### ⚠️ Important Note About Kombu Filters

**Kombu does NOT natively support filtering on `put()` operations**. Kombu's filtering
happens at:

- **Routing level**: Via exchange routing keys (e.g., `routing_key='*.stock.#'`)
- **Consumer level**: Via `message.reject()` after dequeuing

**Your requirement**: Filters that accept/reject items **during** `put()` operations.

**Solution**: We wrap `queue.Queue` (or Kombu) with custom filter logic. This gives you:

1. Immediate filter support with zero dependencies (stdlib `queue.Queue`)
1. Easy backend swap to Kombu/Redis later (just change `self._queue` implementation)

#### Installation

```bash
# Start with zero dependencies (stdlib only)
# Later, when you need distributed:
pip install kombu redis
```

#### Implementation Example

```python
# areal/infrastructure/queue.py
from typing import Protocol, TypeVar, Callable, Any
from kombu import Queue, Connection, Exchange
from kombu.simple import SimpleQueue
import threading

T = TypeVar('T')

class FilterableQueue(Protocol[T]):
    """Protocol defining queue interface with filter support"""

    def put(self, item: T, block: bool = True, timeout: float | None = None) -> bool:
        """Put item into queue. Returns False if filtered out."""
        ...

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """Get item from queue."""
        ...

    def qsize(self) -> int:
        """Return approximate queue size."""
        ...

    def empty(self) -> bool:
        """Return True if queue is empty."""
        ...

class FilterableQueue:
    """Queue implementation with filter support and pluggable backends"""

    def __init__(
        self,
        name: str = 'default',
        maxsize: int = 0,
        backend: str = 'memory',  # 'memory', 'redis://...', 'amqp://...'
        filters: list[Callable[[Any], bool]] | None = None,
    ):
        self.name = name
        self.maxsize = maxsize
        self.filters = filters or []
        self._lock = threading.RLock()

        # Pluggable backend support
        if backend == 'memory':
            # Standard library queue (zero dependencies)
            import queue
            self._queue = queue.Queue(maxsize=maxsize)
            self._backend = 'stdlib'
        else:
            # Kombu for distributed backends (requires: pip install kombu)
            from kombu import Connection
            self._connection = Connection(backend)
            self._queue = self._connection.SimpleQueue(name)
            self._backend = 'kombu'

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> bool:
        """Put item with filter support"""
        with self._lock:
            # Apply filters
            for filter_fn in self.filters:
                if not filter_fn(item):
                    return False  # Filtered out

            if self._backend == 'stdlib':
                self._queue.put(item, block=block, timeout=timeout)
            else:
                self._queue.put(item, block=block, timeout=timeout)

            return True

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Get item from queue"""
        with self._lock:
            if self._backend == 'stdlib':
                return self._queue.get(block=block, timeout=timeout)
            else:
                msg = self._queue.get(block=block, timeout=timeout)
                return msg.payload

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def add_filter(self, filter_fn: Callable[[Any], bool]):
        """Add a filter to the queue"""
        with self._lock:
            self.filters.append(filter_fn)

    def close(self):
        """Cleanup resources"""
        if self._backend == 'kombu':
            self._queue.close()
            self._connection.release()
```

#### Migration Path

```python
# Before (areal/core/async_task_runner.py:180)
self.input_queue: queue.Queue[_TaskInput[T]] = queue.Queue(maxsize=queue_size)

# After
self.input_queue: FilterableQueue[_TaskInput[T]] = container.task_input_queue()
```

______________________________________________________________________

### 2. Event System: **blinker**

#### Why blinker?

- ✅ **Lightweight**: ~500 lines, zero dependencies
- ✅ **Mature**: Used by Flask, Werkzeug, and others since 2010
- ✅ **Async-compatible**: Works with asyncio
- ✅ **Weak references**: Automatic cleanup of dead listeners
- ✅ **Flexible**: Supports sender filtering and custom data

#### Installation

```bash
pip install blinker
```

#### Implementation Example

```python
# areal/infrastructure/events.py
from blinker import signal
from typing import Any

# Define system events
class QueueEvents:
    """Queue-related events"""
    item_added = signal('queue-item-added')
    item_removed = signal('queue-item-removed')
    item_filtered = signal('queue-item-filtered')

class CacheEvents:
    """Cache-related events"""
    item_added = signal('cache-item-added')
    item_removed = signal('cache-item-removed')
    cache_cleared = signal('cache-cleared')

class WorkflowEvents:
    """Workflow lifecycle events"""
    rollout_started = signal('rollout-started')
    rollout_completed = signal('rollout-completed')
    batch_ready = signal('batch-ready')

# Example handler registration
def on_batch_ready(sender, **kwargs):
    """Handler for batch ready event"""
    batch_data = kwargs.get('batch')
    print(f"Batch ready with {len(batch_data)} items")

# Connect handler
WorkflowEvents.batch_ready.connect(on_batch_ready)

# Fire event (from WorkflowExecutor)
WorkflowEvents.batch_ready.send(
    sender=self,
    batch=results,
    version=current_version
)
```

#### Integration with Queues/Caches

```python
class EventAwareQueue(KombuQueue):
    """Queue with event notifications"""

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> bool:
        accepted = super().put(item, block, timeout)

        if accepted:
            QueueEvents.item_added.send(self, item=item)
        else:
            QueueEvents.item_filtered.send(self, item=item)

        return accepted

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        item = super().get(block, timeout)
        QueueEvents.item_removed.send(self, item=item)
        return item
```

**Alternative**: If you need more asyncio integration:

- **asyncio_dispatch**: Better for pure async workflows
- **PyDispatcher**: More powerful but heavier

______________________________________________________________________

### 3. Cache Abstraction: **typing.Protocol + Wrapper**

#### Why Protocol-based?

- ✅ **Zero dependencies**: Uses Python 3.8+ typing.Protocol
- ✅ **Duck typing**: Compatible with list\[\] without subclassing
- ✅ **Pluggable backends**: Easy to swap implementations
- ✅ **Type-safe**: Full mypy/pyright support

#### Implementation Example

```python
# areal/infrastructure/cache.py
from typing import Protocol, TypeVar, Generic, Iterator
import threading
from collections.abc import MutableSequence

T = TypeVar('T')

class Cache(Protocol[T]):
    """Protocol for cache implementations"""

    def append(self, item: T) -> None: ...
    def extend(self, items: list[T]) -> None: ...
    def __getitem__(self, key: int | slice) -> T | list[T]: ...
    def __setitem__(self, key: int | slice, value: T | list[T]) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def clear(self) -> None: ...

class ListCache(Generic[T]):
    """Thread-safe list-based cache with event support"""

    def __init__(self, initial: list[T] | None = None):
        self._data: list[T] = initial or []
        self._lock = threading.RLock()

    def append(self, item: T) -> None:
        with self._lock:
            self._data.append(item)
            CacheEvents.item_added.send(self, item=item)

    def extend(self, items: list[T]) -> None:
        with self._lock:
            self._data.extend(items)
            for item in items:
                CacheEvents.item_added.send(self, item=item)

    def __getitem__(self, key: int | slice) -> T | list[T]:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: int | slice, value: T | list[T]) -> None:
        with self._lock:
            self._data[key] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[T]:
        with self._lock:
            return iter(self._data[:])  # Return copy for thread safety

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            CacheEvents.cache_cleared.send(self)

    def sort(self, *args, **kwargs):
        """Support sorting like list"""
        with self._lock:
            self._data.sort(*args, **kwargs)

class RedisCache(Generic[T]):
    """Redis-backed cache implementation (future)"""

    def __init__(self, redis_url: str, key_prefix: str):
        import redis
        self.client = redis.from_url(redis_url)
        self.prefix = key_prefix
        self._lock = threading.RLock()

    # Implement same interface as ListCache...
```

#### Migration Example

```python
# Before (areal/core/async_task_runner.py:188)
self.result_cache: list[_TimedResult[T]] = []

# After
self.result_cache: Cache[_TimedResult[T]] = container.result_cache()

# Usage remains identical!
self.result_cache.append(result)
items = self.result_cache[:10]
self.result_cache.sort(key=lambda x: x.created_time)
```

______________________________________________________________________

### 4. Dependency Injection: **dependency-injector**

#### Why dependency-injector?

- ✅ **Mature**: Most comprehensive Python DI framework
- ✅ **Container-based**: Clean separation of configuration
- ✅ **Factory support**: Easy to create instances on-demand
- ✅ **Configuration**: YAML/JSON/ENV support
- ✅ **Testing**: Easy to override dependencies

#### Installation

```bash
pip install dependency-injector
```

#### Implementation Example

```python
# areal/infrastructure/container.py
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from .queue import KombuQueue, FilterableQueue
from .cache import ListCache, Cache
from .events import QueueEvents, CacheEvents

class InfrastructureConfig:
    """Configuration for infrastructure components"""
    queue_backend: str = 'memory'  # 'memory' (stdlib), 'redis://localhost:6379/0', 'amqp://...'
    cache_backend: str = 'memory'  # 'memory' (list), 'redis' (future)
    max_queue_size: int = 10240
    enable_events: bool = True

class InfrastructureContainer(containers.DeclarativeContainer):
    """Container for infrastructure dependencies"""

    config = providers.Configuration()

    # Queue factories
    task_input_queue = providers.Factory(
        FilterableQueue,
        name='task_input',
        maxsize=config.max_queue_size,
        backend=config.queue_backend,
    )

    task_output_queue = providers.Factory(
        FilterableQueue,
        name='task_output',
        maxsize=config.max_queue_size,
        backend=config.queue_backend,
    )

    # Cache factories
    result_cache = providers.Factory(
        ListCache,
    )

    pending_results_cache = providers.Factory(
        ListCache,
    )

    pending_inputs_cache = providers.Factory(
        ListCache,
    )

# areal/infrastructure/__init__.py
from .container import InfrastructureContainer, InfrastructureConfig
from .queue import FilterableQueue
from .cache import Cache

# Global container instance
container = InfrastructureContainer()

def initialize_infrastructure(config: InfrastructureConfig | None = None):
    """Initialize infrastructure with configuration"""
    if config:
        container.config.from_dict({
            'max_queue_size': config.max_queue_size,
            'queue_backend': config.queue_backend,
            'cache_backend': config.cache_backend,
        })
    container.wire(modules=[
        'areal.core.async_task_runner',
        'areal.core.workflow_executor',
    ])

__all__ = [
    'InfrastructureContainer',
    'InfrastructureConfig',
    'FilterableQueue',
    'Cache',
    'container',
    'initialize_infrastructure',
]
```

______________________________________________________________________

## Migration Strategy: Minimal Changes

### Step 1: Add Infrastructure Layer (No code changes)

```bash
mkdir -p areal/infrastructure
touch areal/infrastructure/__init__.py
touch areal/infrastructure/queue.py
touch areal/infrastructure/cache.py
touch areal/infrastructure/events.py
touch areal/infrastructure/container.py
```

### Step 2: Update Type Annotations Only

```python
# areal/core/async_task_runner.py
from areal.infrastructure import FilterableQueue, Cache, container
from dependency_injector.wiring import inject, Provide

class AsyncTaskRunner(Generic[T]):
    @inject
    def __init__(
        self,
        ...,
        input_queue: FilterableQueue[_TaskInput[T]] = Provide[
            InfrastructureContainer.task_input_queue
        ],
        output_queue: FilterableQueue[_TimedResult[T]] = Provide[
            InfrastructureContainer.task_output_queue
        ],
        result_cache: Cache[_TimedResult[T]] = Provide[
            InfrastructureContainer.result_cache
        ],
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.result_cache = result_cache
        # Rest remains identical!
```

### Step 3: Initialize at Application Entry

```python
# examples/train_*.py or areal/api/controller_api.py
from areal.infrastructure import initialize_infrastructure, InfrastructureConfig

def main():
    # Initialize infrastructure (starts with stdlib queue.Queue)
    config = InfrastructureConfig(
        queue_backend='memory',  # Switch to 'redis://localhost:6379/0' for distributed
        max_queue_size=10240,
    )
    initialize_infrastructure(config)

    # Rest of application code remains unchanged!
    controller = Controller(...)
    controller.train()
```

______________________________________________________________________

## Benefits of This Approach

### 1. **Zero Breaking Changes**

- All existing code continues to work
- Queue/list operations remain identical (`put()`, `get()`, `append()`, `[:]`)
- Type annotations provide IDE support

### 2. **Future-Proof**

- Switch from in-memory to Redis: **1 line config change**
- Add distributed queues: **no code changes**
- Implement new cache backends: **implement Protocol interface**

### 3. **Testability**

```python
# tests/test_workflow.py
from areal.infrastructure import container
from unittest.mock import Mock

def test_workflow_executor():
    # Override dependencies for testing
    mock_queue = Mock(spec=FilterableQueue)
    mock_cache = Mock(spec=Cache)

    with container.task_input_queue.override(mock_queue):
        with container.result_cache.override(mock_cache):
            executor = WorkflowExecutor(...)
            # Test with mocks
```

### 4. **Event-Driven Extensions**

```python
# Add staleness filtering via events (no queue code changes!)
from areal.infrastructure.events import QueueEvents

@QueueEvents.item_added.connect
def check_staleness(sender, item, **kwargs):
    if is_stale(item):
        # Trigger cleanup or rejection
        pass
```

### 5. **Distributed Ready**

```python
# Switch to distributed mode
config = InfrastructureConfig(
    queue_backend='redis://localhost:6379/0',
    cache_backend='redis',
)
# That's it! All queues now use Redis backend
```

______________________________________________________________________

## Implementation Checklist

- [ ] Install dependencies: `pip install blinker dependency-injector` (Kombu optional,
  only needed for distributed queues)
- [ ] Create `areal/infrastructure/` directory structure
- [ ] Implement `queue.py` with `FilterableQueue` (wraps stdlib `queue.Queue` with
  filter support)
- [ ] Implement `cache.py` with `ListCache` and `Cache` protocol
- [ ] Implement `events.py` with blinker signals
- [ ] Implement `container.py` with dependency-injector container
- [ ] Update `AsyncTaskRunner` to use injected dependencies (type annotations only)
- [ ] Update `WorkflowExecutor` to use injected dependencies
- [ ] Add initialization call in application entry points
- [ ] Write tests with mocked dependencies
- [ ] Document configuration options (in-memory vs Redis)

______________________________________________________________________

## Alternative Frameworks Considered

| Requirement    | Primary Choice                                      | Alternatives                   | Why Not Alternative                                                                                |
| -------------- | --------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------- |
| Queue          | **Custom Wrapper** (stdlib `queue.Queue` + filters) | Kombu, Celery, RQ              | Kombu doesn't support put-time filtering (only consumer-side), Celery too heavy, RQ requires Redis |
| Queue (Future) | **Kombu** (when distributed needed)                 | multiprocessing.Queue, RQ      | Kombu most flexible for multiple backends                                                          |
| Events         | **blinker**                                         | asyncio_dispatch, PyDispatcher | asyncio_dispatch less mature, PyDispatcher heavier                                                 |
| DI             | **dependency-injector**                             | injector, pinject              | injector less features, pinject deprecated                                                         |
| Cache          | **Protocol + ListCache**                            | cachetools, dogpile.cache      | Too heavy for simple list abstraction                                                              |

______________________________________________________________________

## Example: Complete Migration of AsyncTaskRunner

```python
# Before
import queue

class AsyncTaskRunner(Generic[T]):
    def __init__(self, queue_size: int = 0):
        self.input_queue: queue.Queue[_TaskInput[T]] = queue.Queue(maxsize=queue_size)
        self.output_queue: queue.Queue[_TimedResult[T]] = queue.Queue(maxsize=queue_size)
        self.result_cache: list[_TimedResult[T]] = []

# After
from areal.infrastructure import FilterableQueue, Cache, container
from dependency_injector.wiring import inject, Provide

class AsyncTaskRunner(Generic[T]):
    @inject
    def __init__(
        self,
        input_queue: FilterableQueue[_TaskInput[T]] | None = None,
        output_queue: FilterableQueue[_TimedResult[T]] | None = None,
        result_cache: Cache[_TimedResult[T]] | None = None,
    ):
        # Use injected or create default (for backward compatibility)
        self.input_queue = input_queue or container.task_input_queue()
        self.output_queue = output_queue or container.task_output_queue()
        self.result_cache = result_cache or container.result_cache()

        # All other code remains IDENTICAL
```

______________________________________________________________________

## Performance Considerations

1. **In-memory mode**: Near-zero overhead (falls back to `queue.Queue` and `list[]`)
1. **Redis mode**: Network overhead, but enables true distributed queues
1. **Event system**: blinker has minimal overhead (~µs per signal)
1. **DI container**: Instantiation overhead only, runtime is zero-cost

______________________________________________________________________

## Next Steps

1. **Prototype**: Implement infrastructure layer in separate branch
1. **Test**: Verify backward compatibility with existing tests
1. **Migrate**: Update one component at a time (start with `AsyncTaskRunner`)
1. **Validate**: Run full training pipeline to ensure no regressions
1. **Document**: Add configuration guide for switching backends

______________________________________________________________________

## References

- **Kombu**: https://docs.celeryq.dev/projects/kombu/en/stable/
- **blinker**: https://blinker.readthedocs.io/
- **dependency-injector**: https://python-dependency-injector.ets-labs.org/
- **typing.Protocol**: https://peps.python.org/pep-0544/

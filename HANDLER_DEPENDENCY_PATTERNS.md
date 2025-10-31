# Event Handler Dependency Injection Patterns

## Problem Statement

When a handler needs to access queue, cache, and other dependencies (like
`inference_engine`), how do we pass them?

```python
# Handler needs access to these:
def on_pre_weight_update(sender, **kwargs):
    # How to access queue, cache, inference_engine here?
    for item in cache:  # ❌ Where does cache come from?
        if should_recompute(item):
            result = inference_engine.process(item)  # ❌ Where does inference_engine come from?
            queue.put(result)  # ❌ Where does queue come from?
```

______________________________________________________________________

## Solution Options

### ✅ **Option 1: Handler as Class (Recommended for AReaL)**

**Why**: Clean, testable, explicit dependencies, Pythonic

```python
# areal/infrastructure/handlers.py
from typing import Protocol
from areal.infrastructure.queue import FilterableQueue
from areal.infrastructure.cache import Cache
from areal.api.engine_api import InferenceEngine

class PreWeightUpdateHandler:
    """Handler for pre-weight-update event with dependency injection"""

    def __init__(
        self,
        queue: FilterableQueue,
        cache: Cache,
        inference_engine: InferenceEngine,
    ):
        self.queue = queue
        self.cache = cache
        self.inference_engine = inference_engine

    def __call__(self, sender, **kwargs):
        """Called when 'pre_weight_update' event fires"""
        model_version = kwargs.get('version')

        # Access dependencies via self
        for item in self.cache:
            if self._should_recompute(item, model_version):
                # Recompute using inference engine
                result = self.inference_engine.generate(item)

                # Update cache
                self.cache.update(item.id, result)

                # Optionally add back to queue
                self.queue.put(result)

    def _should_recompute(self, item, model_version):
        # Your staleness logic here
        return item.version < model_version
```

**Registration via Dependency Injection Container**:

```python
# areal/infrastructure/container.py
from dependency_injector import containers, providers
from .handlers import PreWeightUpdateHandler

class InfrastructureContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Dependencies
    queue = providers.Factory(FilterableQueue, ...)
    cache = providers.Factory(Cache, ...)
    inference_engine = providers.Factory(InferenceEngine, ...)
    event_bus = providers.Singleton(EventBus, ...)

    # Handler with auto-injected dependencies
    pre_weight_update_handler = providers.Factory(
        PreWeightUpdateHandler,
        queue=queue,
        cache=cache,
        inference_engine=inference_engine,
    )


# In application initialization
from areal.infrastructure import container
from areal.infrastructure.events import WorkflowEvents

# Create handler with injected dependencies
handler = container.pre_weight_update_handler()

# Register to event bus
bus = container.event_bus()
bus.connect(WorkflowEvents.PRE_WEIGHT_UPDATE, handler)
```

**Benefits**:

- ✅ Explicit dependencies (clear what handler needs)
- ✅ Easy to test (mock dependencies in `__init__`)
- ✅ Pythonic (`__call__` makes it callable)
- ✅ Type-safe (full IDE support)
- ✅ Works with DI container

______________________________________________________________________

### ✅ **Option 2: Closure Factory (Lightweight)**

**Why**: Simple, functional style, good for simple handlers

```python
# areal/infrastructure/handlers.py

def make_pre_weight_update_handler(queue, cache, inference_engine):
    """Factory function that captures dependencies in closure"""

    def handler(sender, **kwargs):
        """Actual handler with access to captured dependencies"""
        model_version = kwargs.get('version')

        # Access dependencies from closure
        for item in cache:
            if item.version < model_version:
                result = inference_engine.generate(item)
                cache.update(item.id, result)
                queue.put(result)

    return handler


# Registration
from areal.infrastructure import container

handler = make_pre_weight_update_handler(
    queue=container.queue(),
    cache=container.cache(),
    inference_engine=container.inference_engine(),
)

bus.connect(WorkflowEvents.PRE_WEIGHT_UPDATE, handler)
```

**Benefits**:

- ✅ Simple and concise
- ✅ Captures dependencies automatically
- ✅ Good for small handlers

**Drawbacks**:

- ⚠️ Harder to test (can't easily mock dependencies)
- ⚠️ Less explicit (dependencies hidden in closure)

______________________________________________________________________

### ✅ **Option 3: Partial Application (Functional Style)**

```python
from functools import partial

def pre_weight_update_handler(queue, cache, inference_engine, sender, **kwargs):
    """Handler with explicit dependency parameters"""
    model_version = kwargs.get('version')

    for item in cache:
        if item.version < model_version:
            result = inference_engine.generate(item)
            cache.update(item.id, result)
            queue.put(result)


# Registration with partial
from functools import partial

handler = partial(
    pre_weight_update_handler,
    queue=container.queue(),
    cache=container.cache(),
    inference_engine=container.inference_engine(),
)

bus.connect(WorkflowEvents.PRE_WEIGHT_UPDATE, handler)
```

______________________________________________________________________

### ✅ **Option 4: Dependency Injection Decorator (Advanced)**

**Why**: Auto-inject dependencies from container

```python
from dependency_injector.wiring import inject, Provide
from areal.infrastructure.container import InfrastructureContainer

@inject
def on_pre_weight_update(
    sender,
    queue: FilterableQueue = Provide[InfrastructureContainer.queue],
    cache: Cache = Provide[InfrastructureContainer.cache],
    inference_engine: InferenceEngine = Provide[InfrastructureContainer.inference_engine],
    **kwargs
):
    """Handler with auto-injected dependencies"""
    model_version = kwargs.get('version')

    for item in cache:
        if item.version < model_version:
            result = inference_engine.generate(item)
            cache.update(item.id, result)
            queue.put(result)


# Registration (dependencies auto-injected when called!)
bus.connect(WorkflowEvents.PRE_WEIGHT_UPDATE, on_pre_weight_update)
```

**Note**: Requires wiring the module:

```python
container.wire(modules=['areal.infrastructure.handlers'])
```

**Benefits**:

- ✅ Clean syntax
- ✅ Auto-injection via container
- ✅ Type-safe

**Drawbacks**:

- ⚠️ Magic behavior (dependencies injected invisibly)
- ⚠️ Requires container wiring setup

______________________________________________________________________

### ❌ **Option 5: Pass Dependencies in Event Data (NOT Recommended)**

```python
# Sender passes dependencies in event
bus.send(
    'pre_weight_update',
    sender=self,
    version=42,
    queue=queue,          # ❌ Anti-pattern!
    cache=cache,          # ❌ Anti-pattern!
    inference_engine=engine,  # ❌ Anti-pattern!
)

def handler(sender, queue, cache, inference_engine, **kwargs):
    # Use dependencies from event data
    pass
```

**Why NOT**:

- ❌ Couples sender and handler
- ❌ Sender must know handler's dependencies
- ❌ Violates separation of concerns
- ❌ Hard to add new handlers with different dependencies

______________________________________________________________________

## Recommended Approach for AReaL

### **Use Option 1: Handler as Class + DI Container** ✅

**Implementation**:

```python
# areal/infrastructure/handlers.py
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)


class PreWeightUpdateHandler:
    """Recomputes stale samples in cache/queue before weight update"""

    def __init__(
        self,
        pending_results: Cache,
        pending_inputs: Cache,
        inference_engine: InferenceEngine,
        staleness_manager: StalenessManager,
    ):
        self.pending_results = pending_results
        self.pending_inputs = pending_inputs
        self.inference_engine = inference_engine
        self.staleness_manager = staleness_manager

    def __call__(self, sender, **kwargs):
        """Handle pre-weight-update event"""
        new_version = kwargs.get('version')
        logger.info(f"Pre-weight-update: new version={new_version}")

        # Scan pending results
        stale_count = 0
        for item in self.pending_results:
            if self._is_stale(item, new_version):
                # Option A: Remove stale item
                self.pending_results.remove(item)
                stale_count += 1

                # Option B: Recompute (if needed)
                # recomputed = self.inference_engine.generate(item.input)
                # self.pending_results.update(item.id, recomputed)

        logger.info(f"Removed {stale_count} stale results")

        # Update capacity based on new version
        self.staleness_manager.update_capacity(new_version)

    def _is_stale(self, item, new_version):
        """Check if item is too stale based on version difference"""
        return (new_version - item.version) > self.staleness_manager.max_staleness


class CacheCleanupHandler:
    """Cleans up cache periodically"""

    def __init__(self, cache: Cache, max_size: int = 1000):
        self.cache = cache
        self.max_size = max_size

    def __call__(self, sender, **kwargs):
        """Handle cache-cleanup event"""
        if len(self.cache) > self.max_size:
            # Keep only recent items
            self.cache[:] = self.cache[-self.max_size:]
            logger.info(f"Trimmed cache to {self.max_size} items")


class MetricsCollector:
    """Collects metrics from events"""

    def __init__(self):
        self.metrics = {}

    def __call__(self, sender, **kwargs):
        """Handle any event and collect metrics"""
        event_name = kwargs.get('_event_name', 'unknown')
        self.metrics[event_name] = self.metrics.get(event_name, 0) + 1
```

**Container Configuration**:

```python
# areal/infrastructure/container.py

class InfrastructureContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Core components
    event_bus = providers.Singleton(EventBus, mode=config.event_bus_mode)

    pending_results_cache = providers.Factory(ListCache)
    pending_inputs_cache = providers.Factory(ListCache)

    inference_engine = providers.Singleton(
        InferenceEngine,
        config=config.inference_config,
    )

    staleness_manager = providers.Singleton(
        StalenessManager,
        config=config.staleness_config,
    )

    # Handlers with auto-injected dependencies
    pre_weight_update_handler = providers.Factory(
        PreWeightUpdateHandler,
        pending_results=pending_results_cache,
        pending_inputs=pending_inputs_cache,
        inference_engine=inference_engine,
        staleness_manager=staleness_manager,
    )

    cache_cleanup_handler = providers.Factory(
        CacheCleanupHandler,
        cache=pending_results_cache,
        max_size=config.max_cache_size,
    )

    metrics_collector = providers.Singleton(MetricsCollector)


def register_handlers(container: InfrastructureContainer):
    """Register all event handlers"""
    bus = container.event_bus()

    # Register pre-weight-update handler
    bus.connect(
        WorkflowEvents.PRE_WEIGHT_UPDATE,
        container.pre_weight_update_handler()
    )

    # Register cache cleanup handler
    bus.connect(
        CacheEvents.CACHE_CLEANUP_REQUESTED,
        container.cache_cleanup_handler()
    )

    # Register metrics collector (listens to all events)
    metrics = container.metrics_collector()
    for event_name in [WorkflowEvents.PRE_WEIGHT_UPDATE,
                       WorkflowEvents.BATCH_READY,
                       CacheEvents.CACHE_CLEANUP_REQUESTED]:
        bus.connect(event_name, metrics)
```

**Application Initialization**:

```python
# areal/api/controller_api.py or examples/train_*.py

from areal.infrastructure import container, register_handlers
from areal.infrastructure.events import WorkflowEvents

def main():
    # Initialize container
    container.config.from_dict({
        'event_bus_mode': 'local',
        'max_cache_size': 1000,
        # ... other config
    })

    # Register all handlers
    register_handlers(container)

    # Now events will be handled automatically!
    bus = container.event_bus()

    # Training loop
    for step in range(num_steps):
        # Trigger pre-weight-update event
        bus.send(
            WorkflowEvents.PRE_WEIGHT_UPDATE,
            sender=self,
            version=step + 1,
        )

        # Do weight update
        model.update_weights(...)

        # Trigger batch-ready event
        bus.send(
            WorkflowEvents.BATCH_READY,
            sender=self,
            batch_size=128,
        )
```

______________________________________________________________________

## Testing Handlers

**Option 1 makes testing easy**:

```python
# tests/test_handlers.py
from unittest.mock import Mock
from areal.infrastructure.handlers import PreWeightUpdateHandler

def test_pre_weight_update_handler():
    # Mock dependencies
    mock_cache = Mock()
    mock_cache.__iter__ = Mock(return_value=iter([
        Mock(version=40, id=1),
        Mock(version=38, id=2),  # Stale
    ]))

    mock_engine = Mock()
    mock_staleness = Mock()
    mock_staleness.max_staleness = 2

    # Create handler with mocks
    handler = PreWeightUpdateHandler(
        pending_results=mock_cache,
        pending_inputs=Mock(),
        inference_engine=mock_engine,
        staleness_manager=mock_staleness,
    )

    # Trigger handler
    handler(sender=None, version=42)

    # Assert behavior
    assert mock_cache.remove.called
    # ... more assertions
```

______________________________________________________________________

## Summary: Comparison Table

| Pattern                     | Complexity | Testability  | Type Safety | Best For           |
| --------------------------- | ---------- | ------------ | ----------- | ------------------ |
| **Class-based (Option 1)**  | Medium     | ✅ Excellent | ✅ Full     | **Recommended**    |
| **Closure (Option 2)**      | Low        | ⚠️ Moderate  | ⚠️ Partial  | Simple handlers    |
| **Partial (Option 3)**      | Low        | ⚠️ Moderate  | ✅ Full     | Functional style   |
| **DI Decorator (Option 4)** | High       | ✅ Good      | ✅ Full     | Advanced use cases |
| **Event Data (Option 5)**   | Low        | ❌ Poor      | ❌ None     | ❌ Never use       |

**For AReaL**: Use **Option 1 (Class-based handlers with DI container)** ✅

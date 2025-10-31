# AReaL Infrastructure

Foundational infrastructure for AReaL providing queues, caches, events, and dependency
injection.

## Features

### ✅ Event System (`events.py`)

- **Synchronous event dispatching** using blinker
- Thread-safe in-process events
- Blocks until all handlers complete
- Pre-defined event namespaces (`QueueEvents`, `CacheEvents`, `WorkflowEvents`)
- **Phase 2**: Distributed events via ZMQ/Redis (coming soon)

### ✅ Queues (`queue.py`)

- **Filter support**: Accept/reject items during `put()` operations
- **queue.Queue compatible** API
- Thread-safe with explicit locking
- **Phase 1**: Stdlib `queue.Queue` backend (zero dependencies)
- **Phase 2**: Pluggable backends (Kombu/Redis/RabbitMQ) (coming soon)

### ✅ Caches (`cache.py`)

- **List-compatible** interface (`Cache` protocol)
- Thread-safe operations
- 100% compatible with Python `list[]`
- **Phase 1**: In-memory `ListCache`
- **Phase 2**: Redis backend (coming soon)

### ✅ Dependency Injection (`container.py`, `providers.py`)

- Container-based DI using `dependency-injector`
- **Custom providers** for two-phase initialization
- Easy configuration and testing
- Type-safe with full IDE support

## Installation

Dependencies are automatically installed with AReaL:

```bash
pip install blinker>=1.9 dependency-injector>=4.48.0
```

Or install from AReaL's requirements:

```bash
pip install -e .
```

## Quick Start

### 1. Initialize Infrastructure

```python
from areal.infrastructure import initialize_infrastructure

# Initialize at application startup
initialize_infrastructure({
    'event_bus_mode': 'local',
    'max_queue_size': 10240,
})
```

### 2. Use Event Bus

```python
from areal.infrastructure import get_event_bus, WorkflowEvents

# Get event bus
bus = get_event_bus()

# Define handler
def on_batch_ready(sender, batch_size, **kwargs):
    print(f"Batch ready: {batch_size} items")

# Connect handler
bus.connect(WorkflowEvents.BATCH_READY, on_batch_ready)

# Send event (blocks until handler completes)
bus.send(WorkflowEvents.BATCH_READY, sender=self, batch_size=128)
```

### 3. Use Queues with Filters

```python
from areal.infrastructure import container

# Get queue
queue = container.task_input_queue()

# Add filters
queue.add_filter(lambda x: x > 0)  # Only positive
queue.add_filter(lambda x: x < 100)  # Only < 100

# Use queue
queue.put(-5)   # Returns False (filtered out)
queue.put(50)   # Returns True (accepted)
queue.put(200)  # Returns False (filtered out)

item = queue.get()  # 50
```

### 4. Use Caches (List-Compatible)

```python
from areal.infrastructure import container

# Get cache
cache = container.result_cache()

# Use like a list
cache.append(result1)
cache.extend([result2, result3])
items = cache[:10]  # Slice
cache.sort(key=lambda x: x.version)
```

### 5. Dependency Injection

```python
from areal.infrastructure import container

# Define handler with dependencies
class PreWeightUpdateHandler:
    def __init__(self, queue, cache, inference_engine):
        self.queue = queue
        self.cache = cache
        self.inference_engine = inference_engine

    def __call__(self, sender, **kwargs):
        # Use dependencies
        version = self.inference_engine.get_version()
        for item in self.cache:
            if item.version < version:
                recomputed = self.inference_engine.generate(item)
                self.queue.put(recomputed)

# Register in container (container.py)
from dependency_injector import providers

class InfrastructureContainer(containers.DeclarativeContainer):
    # ... other providers ...

    pre_weight_update_handler = providers.Factory(
        PreWeightUpdateHandler,
        queue=task_input_queue,
        cache=pending_results,
        inference_engine=inference_engine,
    )

# Use
handler = container.pre_weight_update_handler()
bus.connect(WorkflowEvents.PRE_WEIGHT_UPDATE, handler)
```

## Architecture

```
areal/infrastructure/
├── __init__.py              # Public API and initialization
├── events.py                # Event system (blinker)
├── queue.py                 # Filterable queues
├── cache.py                 # List-compatible caches
├── providers.py             # Custom DI providers
├── container.py             # DI container
├── example_usage.py         # Usage examples
└── README.md                # This file
```

## Event-Driven Architecture

```
Training Loop
    ↓
Send Event: 'pre-weight-update'
    ↓
Handler: PreWeightUpdateHandler
    ↓ (accesses dependencies)
    ├─ Queue (filter stale items)
    ├─ Cache (scan for recomputation)
    └─ InferenceEngine (recompute if needed)
    ↓
Event completes (handler returns)
    ↓
Update weights (safe, cache is clean)
```

## Two-Phase Initialization

AReaL components follow this pattern:

```python
# Phase 1: __init__() - Lightweight
engine = RemoteSGLangEngine(config=config)

# Phase 2: initialize() - Heavy (GPU, network)
engine.initialize(train_data_parallel_size=4)
```

The `InitializableProvider` handles both automatically:

```python
# Container configuration
inference_engine = InitializableProvider(
    RemoteSGLangEngine,
    config=config,
    init_kwargs={'train_data_parallel_size': 4}
)

# Usage - fully initialized!
engine = container.inference_engine()
engine.submit(...)  # Ready to use
```

## Testing

```python
from unittest.mock import Mock

# Easy to mock dependencies
mock_engine = Mock()
mock_engine.get_version.return_value = 42

handler = PreWeightUpdateHandler(
    queue=Mock(),
    cache=Mock(),
    inference_engine=mock_engine
)

handler(sender=None, version=43)
assert mock_engine.get_version.called
```

Or override container providers:

```python
with container.inference_engine.override(Mock()):
    handler = container.pre_weight_update_handler()
    # handler has mocked engine
```

## Examples

Run the example file to see all features in action:

```bash
python -m areal.infrastructure.example_usage
```

Examples include:

1. Basic usage (events, queues, caches)
1. Queue filters
1. Handlers with dependencies
1. Cache operations (list-compatible)
1. Thread-safe operations

## Phase 2 Roadmap

Coming soon:

- [ ] **Distributed events** via ZMQ Pub/Sub or Redis Pub/Sub
- [ ] **Distributed queues** via Kombu (Redis/RabbitMQ backends)
- [ ] **Redis cache** backend for distributed caching
- [ ] **Event persistence** and replay
- [ ] **Monitoring** and metrics integration

## Best Practices

### 1. Initialize Once at Startup

```python
# In main() or application entry point
initialize_infrastructure(config)
```

### 2. Use Descriptive Provider Names

```python
# Good
task_input_queue = providers.Factory(...)
model_request_queue = providers.Factory(...)

# Bad
queue1 = providers.Factory(...)
queue2 = providers.Factory(...)
```

### 3. Handlers as Classes (Not Functions)

```python
# Good - testable, explicit dependencies
class MyHandler:
    def __init__(self, queue, cache):
        self.queue = queue
        self.cache = cache

    def __call__(self, sender, **kwargs):
        pass

# Less good - dependencies from closure (harder to test)
def make_handler(queue, cache):
    def handler(sender, **kwargs):
        pass  # Uses queue, cache from closure
    return handler
```

### 4. Keep Handlers Fast

Event dispatch is synchronous - handlers should complete quickly:

```python
# Good - fast handler
def on_event(sender, **kwargs):
    item.mark_stale()  # Quick flag update

# Bad - slow handler (blocks event dispatch)
def on_event(sender, **kwargs):
    time.sleep(5)  # Blocks other handlers!
```

For long operations, use background threads or make handlers trigger async tasks.

### 5. Handle Exceptions in Handlers

```python
class MyHandler:
    def __call__(self, sender, **kwargs):
        try:
            # Handler logic
            pass
        except Exception as e:
            logger.error(f"Handler failed: {e}")
            # Don't let exception break other handlers
```

## Documentation

See the markdown files in the repository root for detailed design documentation:

- `FRAMEWORK_RECOMMENDATION.md` - Framework choices and rationale
- `BLINKER_SYNC_ASYNC_BEHAVIOR.md` - Event system behavior
- `HANDLER_DEPENDENCY_PATTERNS.md` - DI patterns for handlers
- `DEPENDENCY_INJECTOR_NAMING.md` - DI naming conventions
- `LAZY_INITIALIZATION_ANALYSIS.md` - Two-phase init analysis
- `POLYMORPHIC_DI_SOLUTION.md` - Multiple engine types support

## Contributing

When adding new infrastructure components:

1. Follow existing patterns (Protocol-based, thread-safe)
1. Add type hints and docstrings
1. Update container with providers
1. Add examples to `example_usage.py`
1. Update this README

## License

Apache-2.0 (same as AReaL)

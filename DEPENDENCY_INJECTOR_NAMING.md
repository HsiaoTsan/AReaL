# Dependency Injector: By Type vs By Name

## TL;DR

**`dependency-injector` injects BY NAME (parameter name), NOT by type** ✅

When you have multiple instances of the same type (e.g., multiple queues), it matches:

- **Parameter name** in `__init__()` → **Provider name** in container

______________________________________________________________________

## How It Works

### Example: Multiple Queues of Same Type

```python
# You have TWO queues (same type, different purposes)
class AsyncTaskRunner:
    def __init__(
        self,
        input_queue,   # ← Parameter name: "input_queue"
        output_queue,  # ← Parameter name: "output_queue"
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
```

### Container: Named Providers

```python
from dependency_injector import containers, providers

class InfrastructureContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Provider name: "input_queue"
    input_queue = providers.Factory(
        FilterableQueue,
        name='input',
        maxsize=config.max_queue_size,
    )

    # Provider name: "output_queue"
    output_queue = providers.Factory(
        FilterableQueue,
        name='output',
        maxsize=config.max_queue_size,
    )

    # Provider name: "task_runner"
    task_runner = providers.Factory(
        AsyncTaskRunner,
        input_queue=input_queue,   # ← Explicitly wire by name!
        output_queue=output_queue, # ← Explicitly wire by name!
    )
```

**How it resolves**:

```
AsyncTaskRunner.__init__(input_queue, output_queue)
                         ↓            ↓
                         Matches by parameter name
                         ↓            ↓
container.input_queue    container.output_queue
```

______________________________________________________________________

## Three Ways to Inject Dependencies

### ✅ **Method 1: Explicit Wiring in Container (Recommended)**

**Most common and explicit**:

```python
class InfrastructureContainer(containers.DeclarativeContainer):
    # Define providers with unique names
    task_input_queue = providers.Factory(FilterableQueue, name='task_input')
    task_output_queue = providers.Factory(FilterableQueue, name='task_output')
    result_cache = providers.Factory(ListCache)
    pending_results_cache = providers.Factory(ListCache)

    # Wire dependencies explicitly by parameter name
    pre_weight_update_handler = providers.Factory(
        PreWeightUpdateHandler,
        queue=task_input_queue,                    # parameter "queue" gets this provider
        cache=pending_results_cache,               # parameter "cache" gets this provider
        inference_engine=inference_engine,         # parameter "inference_engine" gets this provider
    )
```

**Usage**:

```python
# Create instance with dependencies auto-injected
handler = container.pre_weight_update_handler()

# Under the hood, it does:
# handler = PreWeightUpdateHandler(
#     queue=container.task_input_queue(),
#     cache=container.pending_results_cache(),
#     inference_engine=container.inference_engine(),
# )
```

**Benefits**:

- ✅ Explicit and clear
- ✅ Easy to understand dependency graph
- ✅ No magic
- ✅ Type-safe with IDE support

______________________________________________________________________

### ✅ **Method 2: `@inject` Decorator with `Provide[]` (Auto-injection)**

**More magical but cleaner code**:

```python
from dependency_injector.wiring import inject, Provide
from areal.infrastructure.container import InfrastructureContainer

class PreWeightUpdateHandler:
    @inject
    def __init__(
        self,
        queue: FilterableQueue = Provide[InfrastructureContainer.task_input_queue],
        cache: Cache = Provide[InfrastructureContainer.pending_results_cache],
        inference_engine: InferenceEngine = Provide[InfrastructureContainer.inference_engine],
    ):
        self.queue = queue
        self.cache = cache
        self.inference_engine = inference_engine
```

**Container (simpler)**:

```python
class InfrastructureContainer(containers.DeclarativeContainer):
    task_input_queue = providers.Factory(FilterableQueue, ...)
    pending_results_cache = providers.Factory(ListCache, ...)
    inference_engine = providers.Factory(InferenceEngine, ...)

    # No need to wire explicitly! @inject does it
    pre_weight_update_handler = providers.Factory(PreWeightUpdateHandler)
```

**Must wire the module**:

```python
# In initialization
container.wire(modules=['areal.infrastructure.handlers'])
```

**Usage**:

```python
# Dependencies auto-injected via @inject decorator!
handler = container.pre_weight_update_handler()

# Or even without container (if wired):
handler = PreWeightUpdateHandler()  # Auto-injected!
```

**Benefits**:

- ✅ Clean syntax
- ✅ Type annotations show dependencies
- ✅ Can override at call time

**Drawbacks**:

- ⚠️ Requires wiring setup
- ⚠️ Magic behavior (dependencies injected invisibly)
- ⚠️ Default values are markers, not actual defaults

______________________________________________________________________

### ✅ **Method 3: Manual Instantiation**

**No container needed**:

```python
# Create dependencies manually
task_queue = FilterableQueue(name='task_input', maxsize=1000)
results_cache = ListCache()
engine = InferenceEngine(config=...)

# Pass to constructor
handler = PreWeightUpdateHandler(
    queue=task_queue,
    cache=results_cache,
    inference_engine=engine,
)
```

**Use case**: Testing, simple scripts, when you don't need DI container

______________________________________________________________________

## Matching Rules: By Parameter Name

### Rule: Parameter name → Provider name

```python
# Handler constructor
class Handler:
    def __init__(self, my_queue, my_cache):
        #              ↑          ↑
        #              Parameter names

# Container
class Container(containers.DeclarativeContainer):
    my_queue = providers.Factory(Queue)  # ← Must match parameter name!
    my_cache = providers.Factory(Cache)  # ← Must match parameter name!

    handler = providers.Factory(
        Handler,
        my_queue=my_queue,  # ← Explicit mapping by name
        my_cache=my_cache,
    )
```

### What if names don't match?

**Problem**:

```python
class Handler:
    def __init__(self, queue):  # ← Parameter name: "queue"
        pass

class Container(containers.DeclarativeContainer):
    task_input_queue = providers.Factory(Queue)  # ← Provider name different!

    handler = providers.Factory(Handler)  # ❌ Won't work! No "queue" provider
```

**Solution**: Explicitly map:

```python
class Container(containers.DeclarativeContainer):
    task_input_queue = providers.Factory(Queue)

    handler = providers.Factory(
        Handler,
        queue=task_input_queue,  # ← Map "queue" parameter to "task_input_queue" provider
    )
```

______________________________________________________________________

## Real-World Example: AReaL

### Multiple Queues and Caches (Same Types)

```python
# areal/core/async_task_runner.py
class AsyncTaskRunner:
    def __init__(
        self,
        input_queue: FilterableQueue,   # ← Same type
        output_queue: FilterableQueue,  # ← Same type
        result_cache: Cache,            # ← Same type
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.result_cache = result_cache

# areal/core/workflow_executor.py
class WorkflowExecutor:
    def __init__(
        self,
        task_runner: AsyncTaskRunner,
        pending_results: Cache,         # ← Same type as result_cache!
        pending_inputs: Cache,          # ← Same type!
    ):
        self.task_runner = task_runner
        self.pending_results = pending_results
        self.pending_inputs = pending_inputs
```

### Container with Named Providers

```python
# areal/infrastructure/container.py
class InfrastructureContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # ==== Queues (same type, different names) ====
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

    # ==== Caches (same type, different names) ====
    result_cache = providers.Factory(ListCache)

    pending_results = providers.Factory(ListCache)

    pending_inputs = providers.Factory(ListCache)

    # ==== Engines ====
    inference_engine = providers.Singleton(
        InferenceEngine,
        config=config.inference_config,
    )

    # ==== Components ====
    task_runner = providers.Factory(
        AsyncTaskRunner,
        input_queue=task_input_queue,    # Map to specific queue
        output_queue=task_output_queue,  # Map to specific queue
        result_cache=result_cache,       # Map to specific cache
    )

    workflow_executor = providers.Factory(
        WorkflowExecutor,
        task_runner=task_runner,
        pending_results=pending_results,  # Different cache instance!
        pending_inputs=pending_inputs,    # Different cache instance!
    )

    # ==== Handlers ====
    pre_weight_update_handler = providers.Factory(
        PreWeightUpdateHandler,
        queue=task_input_queue,           # Inject specific queue
        cache=pending_results,            # Inject specific cache
        inference_engine=inference_engine,
    )
```

**Key Points**:

- ✅ Multiple instances of `FilterableQueue` with unique provider names
- ✅ Multiple instances of `Cache` with unique provider names
- ✅ Each component gets the correct instances via explicit wiring
- ✅ Clear dependency graph

______________________________________________________________________

## Advanced: Runtime Overrides

You can override dependencies at call time:

```python
# Create with default dependencies
handler = container.pre_weight_update_handler()

# Override at call time
custom_cache = ListCache()
handler = container.pre_weight_update_handler(cache=custom_cache)

# Or even override configuration
container.config.max_queue_size.from_value(5000)
queue = container.task_input_queue()  # Uses new maxsize!
```

______________________________________________________________________

## Testing: Easy Mocking

Because dependencies are explicit, testing is straightforward:

```python
# tests/test_handlers.py
from unittest.mock import Mock

def test_pre_weight_update_handler():
    # Create mocks
    mock_queue = Mock(spec=FilterableQueue)
    mock_cache = Mock(spec=Cache)
    mock_engine = Mock(spec=InferenceEngine)

    # Inject mocks directly (no container needed!)
    handler = PreWeightUpdateHandler(
        queue=mock_queue,
        cache=mock_cache,
        inference_engine=mock_engine,
    )

    # Test
    handler(sender=None, version=42)

    # Assert
    assert mock_cache.update.called
```

**Or override container providers**:

```python
def test_with_container():
    # Override providers with mocks
    with container.task_input_queue.override(Mock(spec=FilterableQueue)):
        with container.pending_results.override(Mock(spec=Cache)):
            handler = container.pre_weight_update_handler()
            # handler has mocked dependencies!
```

______________________________________________________________________

## Comparison with Other DI Libraries

| Library                 | Injection Method                    | Multiple Same Type           |
| ----------------------- | ----------------------------------- | ---------------------------- |
| **dependency-injector** | By parameter name (explicit wiring) | ✅ Named providers           |
| **injector**            | By type + `Annotated`               | ✅ `Annotated[Type, 'name']` |
| **pinject**             | By parameter name                   | ✅ Named bindings            |
| **FastAPI**             | By type + `Depends()`               | ⚠️ Limited support           |

**dependency-injector** is the most explicit and clear for multiple instances!

______________________________________________________________________

## Best Practices for AReaL

### 1. Use Descriptive Provider Names

```python
# ✅ Good: Clear purpose
task_input_queue = providers.Factory(...)
model_worker_request_queue = providers.Factory(...)
stream_data_queue = providers.Factory(...)

# ❌ Bad: Generic names
queue1 = providers.Factory(...)
queue2 = providers.Factory(...)
```

### 2. Group Related Providers

```python
class InfrastructureContainer(containers.DeclarativeContainer):
    # ==== Configuration ====
    config = providers.Configuration()

    # ==== Event System ====
    event_bus = providers.Singleton(EventBus, ...)

    # ==== Queues ====
    task_input_queue = providers.Factory(...)
    task_output_queue = providers.Factory(...)

    # ==== Caches ====
    result_cache = providers.Factory(...)
    pending_results = providers.Factory(...)

    # ==== Engines ====
    inference_engine = providers.Singleton(...)
    train_engine = providers.Singleton(...)

    # ==== Handlers ====
    pre_weight_update_handler = providers.Factory(...)
```

### 3. Explicit Wiring (No Magic)

```python
# ✅ Recommended: Explicit
handler = providers.Factory(
    PreWeightUpdateHandler,
    queue=task_input_queue,           # Clear which queue
    cache=pending_results,            # Clear which cache
    inference_engine=inference_engine,
)

# ⚠️ Avoid: Implicit (requires @inject and wiring)
handler = providers.Factory(PreWeightUpdateHandler)
```

### 4. Document Dependencies

```python
class PreWeightUpdateHandler:
    """
    Handles pre-weight-update event.

    Dependencies:
        queue: Task input queue (for resubmitting recomputed items)
        cache: Pending results cache (for scanning stale samples)
        inference_engine: For recomputing stale samples
    """
    def __init__(self, queue, cache, inference_engine):
        ...
```

______________________________________________________________________

## Summary: Answer to Your Question

### **Q: Will the injector inject queue and cache by type? Or by name?**

**A: By NAME (parameter name)** ✅

```python
# Parameter names in constructor
class Handler:
    def __init__(self, my_queue, my_cache):
        #              ↑          ↑
        #              These parameter names...

# Must match provider names (or be explicitly mapped)
class Container(containers.DeclarativeContainer):
    my_queue = providers.Factory(Queue)  # ← Match!
    my_cache = providers.Factory(Cache)  # ← Match!

    handler = providers.Factory(
        Handler,
        my_queue=my_queue,  # ← Explicit mapping
        my_cache=my_cache,
    )
```

**Key Points**:

- ✅ **Not by type** - you can have multiple instances of same type
- ✅ **By parameter name** - parameter name maps to provider name
- ✅ **Explicit wiring** - you control which instance goes where
- ✅ **Clear and testable** - no ambiguity

This is perfect for AReaL where you have multiple queues and caches of the same type! 🎯

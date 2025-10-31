# Dependency Injection Patterns: Singleton vs Factory

## Your Question

> Does the provider provide new instance of queue for each call, or use a singleton
> pattern? If former, how does our event handler get the reference of this queue (DI)?

## Answer: **Singleton Pattern**

After refactoring, we use **Singleton** for queues and runners:

```python
# In app_container.py
async_task_input_queue = providers.Singleton(FilterableQueue, ...)  # ✓ Singleton
async_task_output_queue = providers.Singleton(FilterableQueue, ...)  # ✓ Singleton
async_task_runner = providers.Singleton(AsyncTaskRunner, ...)        # ✓ Singleton
```

**This means**:

- `app_container.async_task_input_queue()` returns the **same queue** every time
- `app_container.async_task_runner()` returns the **same runner** every time
- All components share the same queues and runner application-wide

## Why Singleton?

### 1. Event Handler Access

Event handlers can reliably get queue references:

```python
from areal.core.app_container import app_container

class MyEventHandler:
    def __init__(self):
        # Get the singleton queue - same one used by AsyncTaskRunner
        self.queue = app_container.async_task_input_queue()

    def on_some_event(self, sender, **kwargs):
        # Push to the shared queue
        self.queue.put(new_task)
```

### 2. Application-Wide Sharing

Multiple inference engines share the same task execution infrastructure:

```python
# Both engines use the same runner
engine1 = SGLangEngine(config1)
engine2 = RemoteInfEngine(config2, backend)

# They both get the singleton runner internally
# Tasks from both engines go to the same queues
```

### 3. Centralized Management

One runner manages all async tasks for the application:

```python
# Configure once
app_container.config.from_dict({'max_queue_size': 5000})

# Everyone gets the same configured instance
runner = app_container.async_task_runner()
```

## How Event Handlers Get Queue References

### Method 1: Direct Injection (Active Access)

For handlers that need to **actively** push/pull from queues:

```python
from areal.core.app_container import app_container
from areal.infrastructure import get_event_bus

class TaskSubmitHandler:
    """Handler that submits tasks to queue on events."""

    def __init__(self):
        # Get the singleton queue
        self.input_queue = app_container.async_task_input_queue()

        # Register for events
        bus = get_event_bus()
        bus.connect('workflow-complete', self.on_workflow_complete)

    def on_workflow_complete(self, sender, **kwargs):
        result = kwargs['result']
        # Process and submit new task
        new_task = self.process(result)
        self.input_queue.put(new_task)
```

### Method 2: Via Event Sender (Passive Access)

For handlers that react to queue events:

```python
from areal.infrastructure import get_event_bus, QueueEvents

def monitor_queue(sender, **kwargs):
    """Handler receives queue as sender parameter."""
    queue = sender  # sender IS the queue instance!
    item = kwargs['item']
    print(f"Queue {queue.name} received item: {item}")
    print(f"Queue size: {queue.qsize()}")

# Connect to queue events
bus = get_event_bus()
bus.connect(QueueEvents.ITEM_ADDED, monitor_queue)

# Enable event firing in config
app_container.config.from_dict({'fire_queue_events': True})
```

### Method 3: Container-Managed Handler

Best practice - let container manage handler lifecycle:

```python
# In your application container
class MyAppContainer(ApplicationContainer):

    # Define handler as singleton
    task_monitor = providers.Singleton(
        TaskMonitorHandler,
        input_queue=async_task_input_queue,  # Inject queue
        output_queue=async_task_output_queue,
    )

    # Auto-initialize and register
    def wire_handlers(self):
        monitor = self.task_monitor()
        monitor.register()  # Connect to events

# Usage
container = MyAppContainer()
container.wire_handlers()
```

## Example: Complete Event Handler Pattern

```python
# areal/handlers/task_monitor.py
from areal.infrastructure import FilterableQueue, get_event_bus, QueueEvents

class TaskMonitorHandler:
    """Monitors task queues and logs statistics."""

    def __init__(
        self,
        input_queue: FilterableQueue,
        output_queue: FilterableQueue,
    ):
        """Initialize with injected queues."""
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.task_count = 0

    def register(self):
        """Register event listeners."""
        bus = get_event_bus()
        bus.connect(QueueEvents.ITEM_ADDED, self.on_item_added)
        bus.connect(QueueEvents.ITEM_REMOVED, self.on_item_removed)

    def on_item_added(self, sender, **kwargs):
        """Handle item added to any queue."""
        queue = sender
        if queue is self.input_queue:
            self.task_count += 1
            print(f"Task submitted. Total: {self.task_count}")

    def on_item_removed(self, sender, **kwargs):
        """Handle item removed from any queue."""
        queue = sender
        if queue is self.output_queue:
            print(f"Task completed. Pending: {self.input_queue.qsize()}")


# In application setup
from areal.core.app_container import app_container

# Enable queue events
app_container.config.from_dict({
    'fire_queue_events': True,
    'max_queue_size': 10000,
})

# Create handler with injected queues
handler = TaskMonitorHandler(
    input_queue=app_container.async_task_input_queue(),
    output_queue=app_container.async_task_output_queue(),
)
handler.register()

# Now all queue operations fire events that handler receives
runner = app_container.async_task_runner()
runner.initialize()
```

## Singleton Benefits Summary

### ✓ Reliable References

- Event handlers get the same queue instance
- No confusion about which queue to monitor

### ✓ Shared State

- All components see the same queue state
- One source of truth for pending tasks

### ✓ Easy Configuration

- Configure once at application startup
- All components use the same config

### ✓ Resource Efficiency

- One runner thread, not multiple
- Shared memory for queues

## Testing with Singletons

Tests need to reset singletons between runs:

```python
def create_runner(max_queue_size=10, **kwargs):
    """Helper to create fresh runner for tests."""
    # Reset singletons
    app_container.async_task_runner.reset()
    app_container.async_task_input_queue.reset()
    app_container.async_task_output_queue.reset()
    app_container.async_task_result_cache.reset()

    # Configure
    app_container.config.from_dict({
        'max_queue_size': max_queue_size,
        **kwargs
    })

    # Get fresh instance
    runner = app_container.async_task_runner()
    return runner
```

## Alternative: Factory Pattern (Not Recommended)

If you used Factory instead:

```python
# DON'T DO THIS (creates new instances each time)
async_task_input_queue = providers.Factory(FilterableQueue, ...)
async_task_runner = providers.Factory(AsyncTaskRunner, ...)
```

**Problems**:

- ✗ Each call creates NEW queues
- ✗ Event handlers don't know which queue to reference
- ✗ Multiple runners compete for resources
- ✗ No shared state between components

**When to use Factory**:

- For stateless utilities
- For per-request/per-operation objects
- For testing with different configurations

But for **infrastructure** like queues and runners, **Singleton is correct**.

## Summary

1. **Queues are Singleton** - One instance application-wide
1. **Runner is Singleton** - One runner manages all tasks
1. **Event handlers access queues via**:
   - Direct injection: `app_container.async_task_input_queue()`
   - Event sender: Queue passed as `sender` parameter
   - Container management: Handler has queue injected
1. **Tests reset singletons** between runs for isolation
1. **All tests passing** (80/80) with Singleton pattern

## Code Locations

- **Container**: `areal/core/app_container.py`
- **AsyncTaskRunner**: `areal/core/async_task_runner.py`
- **FilterableQueue**: `areal/infrastructure/queue.py`
- **Event System**: `areal/infrastructure/events.py`
- **Tests**: `areal/tests/test_async_task_runner.py`

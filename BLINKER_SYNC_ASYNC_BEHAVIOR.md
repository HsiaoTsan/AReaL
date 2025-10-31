# Blinker: Synchronous vs Asynchronous Event Handling

## TL;DR

✅ **Yes, blinker supports SYNC (blocking) mode by default** ✅ **`signal.send()` blocks
until ALL handlers complete** ✅ **Handlers are called sequentially in registration
order** ✅ **Also supports async handlers via `send_async()` (added in recent versions)**

______________________________________________________________________

## Question 2: Sync vs Async Behavior

### Default Behavior: **Synchronous and Blocking** ✅

```python
from blinker import signal

# Create signal
pre_weight_update = signal('pre_weight_update')

# Connect handlers
@pre_weight_update.connect
def handler1(sender, **kwargs):
    print("Handler 1 start")
    time.sleep(1)  # Simulate work
    print("Handler 1 done")

@pre_weight_update.connect
def handler2(sender, **kwargs):
    print("Handler 2 start")
    time.sleep(1)  # Simulate work
    print("Handler 2 done")

# Send signal - BLOCKS until all handlers complete!
print("Sending signal...")
pre_weight_update.send(None, version=42)
print("All handlers completed!")  # This prints AFTER handlers finish

# Output:
# Sending signal...
# Handler 1 start
# Handler 1 done
# Handler 2 start
# Handler 2 done
# All handlers completed!
```

**Execution Flow**:

```
send() called
    ↓
Handler 1 executes (blocks)
    ↓
Handler 2 executes (blocks)
    ↓
send() returns
```

______________________________________________________________________

## Perfect for Your Use Case! ✅

Your `pre_weight_update` handler needs to:

1. Scan cache/queue for stale samples
1. Recompute using inference engine
1. Update cache/queue
1. **Then proceed with weight update**

**Blinker's synchronous behavior is ideal**:

```python
from areal.infrastructure.events import get_event_bus, WorkflowEvents

# In training loop
def train_step(self):
    # 1. Send pre-weight-update event (BLOCKS until handlers finish)
    bus = get_event_bus()
    bus.send(
        WorkflowEvents.PRE_WEIGHT_UPDATE,
        sender=self,
        version=self.current_version + 1
    )
    # ✅ At this point, ALL handlers have completed
    # ✅ Cache/queue have been cleaned up
    # ✅ Stale samples removed/recomputed

    # 2. Safe to proceed with weight update
    self.model.update_weights(gradients)
    self.current_version += 1

    # 3. Continue training
    ...
```

______________________________________________________________________

## Async Support (If Needed)

Blinker added `send_async()` for async handlers:

### Using `send_async()` with Async Handlers

```python
from blinker import signal
import asyncio

pre_weight_update = signal('pre_weight_update')

# Async handler
@pre_weight_update.connect
async def async_handler(sender, **kwargs):
    print("Async handler start")
    await asyncio.sleep(1)  # Async work
    print("Async handler done")

# Send to async handlers
async def main():
    await pre_weight_update.send_async(None, version=42)
    print("All async handlers completed!")

asyncio.run(main())
```

### Mixing Sync and Async Handlers

```python
# Sync handler
@pre_weight_update.connect
def sync_handler(sender, **kwargs):
    print("Sync handler")

# Async handler
@pre_weight_update.connect
async def async_handler(sender, **kwargs):
    await asyncio.sleep(1)
    print("Async handler")

# Option 1: Call from async context with wrapper for sync handlers
async def main():
    await pre_weight_update.send_async(
        None,
        _sync_wrapper=lambda fn: asyncio.to_thread(fn)  # Run sync in thread
    )

# Option 2: Call from sync context with wrapper for async handlers
def main_sync():
    pre_weight_update.send(
        None,
        _async_wrapper=lambda coro: asyncio.run(coro)  # Run async in event loop
    )
```

______________________________________________________________________

## Handler Execution Order and Error Handling

### Sequential Execution

Handlers execute in **registration order**:

```python
signal = signal('test')

@signal.connect
def handler1(sender, **kwargs):
    print("Handler 1")

@signal.connect
def handler2(sender, **kwargs):
    print("Handler 2")

@signal.connect
def handler3(sender, **kwargs):
    print("Handler 3")

signal.send(None)

# Output:
# Handler 1
# Handler 2
# Handler 3
```

### Error Handling: All or Nothing

**Important**: If one handler raises an exception, subsequent handlers **won't run**:

```python
@signal.connect
def handler1(sender, **kwargs):
    print("Handler 1 - OK")

@signal.connect
def handler2(sender, **kwargs):
    print("Handler 2 - ERROR")
    raise ValueError("Oops!")

@signal.connect
def handler3(sender, **kwargs):
    print("Handler 3 - Never runs!")

signal.send(None)

# Output:
# Handler 1 - OK
# Handler 2 - ERROR
# ValueError: Oops!
# (Handler 3 never executes)
```

### Solution: Catch Exceptions in Handlers

```python
class PreWeightUpdateHandler:
    def __call__(self, sender, **kwargs):
        try:
            # Your logic here
            self._process_cache()
            self._recompute_stale()
        except Exception as e:
            logger.error(f"Error in pre-weight-update handler: {e}")
            # Continue despite error (don't block other handlers)
```

### Or Wrap at Event Bus Level

```python
# areal/infrastructure/events.py

class EventBus:
    def send(self, event_name: str, sender=None, **kwargs):
        """Send event with error handling"""
        signal = self.signal(event_name)

        # Get all receivers
        receivers = signal.receivers_for(sender)

        results = []
        for receiver in receivers:
            try:
                result = receiver(sender, **kwargs)
                results.append((receiver, result))
            except Exception as e:
                logger.error(f"Error in handler {receiver}: {e}")
                results.append((receiver, e))
                # Continue to next handler

        return results
```

______________________________________________________________________

## Performance Characteristics

### Blocking Time

For `N` handlers each taking `T` seconds:

- **Total time**: `N × T` seconds (sequential execution)

```python
# 3 handlers, each taking 1 second
handler1: 1s
handler2: 1s
handler3: 1s
-----------
Total: 3s (blocked)
```

### If You Need Non-Blocking

**Option 1**: Make handlers fast (recommended)

```python
class PreWeightUpdateHandler:
    def __call__(self, sender, **kwargs):
        # Quick validation only
        self._mark_stale_items()  # Fast: just mark, don't recompute

    def _mark_stale_items(self):
        # O(1) or O(n) with simple checks
        for item in self.cache:
            if item.version < self.target_version:
                item.stale = True  # Just mark, don't process
```

**Option 2**: Use background threads in handler (if really needed)

```python
import threading

class PreWeightUpdateHandler:
    def __call__(self, sender, **kwargs):
        # Launch background thread for heavy work
        thread = threading.Thread(
            target=self._recompute_stale_async,
            args=(kwargs.get('version'),),
            daemon=True
        )
        thread.start()
        # Handler returns immediately (non-blocking)

    def _recompute_stale_async(self, version):
        # Heavy recomputation in background
        pass
```

**Option 3**: Use `send_async()` with async handlers

```python
async def async_handler(sender, **kwargs):
    # Can use await for I/O-bound operations
    results = await self.inference_engine.agenerate(items)

await bus.send_async('pre_weight_update', version=42)
```

______________________________________________________________________

## Recommendation for AReaL

### ✅ **Use Default Synchronous Mode**

**Why**:

1. ✅ Guarantees handlers complete before weight update
1. ✅ Simple to reason about (no concurrency bugs)
1. ✅ Matches training loop semantics (step-by-step)
1. ✅ Easy to debug (sequential execution)

**Implementation**:

```python
# areal/infrastructure/events.py

class EventBus:
    """Event bus with synchronous (blocking) behavior by default"""

    def __init__(self, mode='local'):
        self.mode = mode
        self._local_signals = {}

    def signal(self, name: str):
        """Get or create a signal"""
        if name not in self._local_signals:
            from blinker import signal
            self._local_signals[name] = signal(name)
        return self._local_signals[name]

    def connect(self, event_name: str, handler, sender=None):
        """Connect handler to event"""
        self.signal(event_name).connect(handler, sender=sender)

    def send(self, event_name: str, sender=None, **kwargs):
        """
        Send event and BLOCK until all handlers complete.

        This is the default behavior - ensures all event processing
        is complete before continuing execution.
        """
        self.signal(event_name).send(sender, **kwargs)
        # Returns only after all handlers finish

    async def send_async(self, event_name: str, sender=None, **kwargs):
        """
        Send event to async handlers.

        Use this if handlers need to await async operations.
        """
        await self.signal(event_name).send_async(sender, **kwargs)
```

**Usage in Training Loop**:

```python
# Training step - synchronous and clear
def train_step(self):
    # 1. Prepare batch
    batch = self.prepare_batch()

    # 2. Fire pre-weight-update event (BLOCKS)
    bus.send(
        WorkflowEvents.PRE_WEIGHT_UPDATE,
        sender=self,
        version=self.version + 1,
        batch=batch,
    )
    # ✅ All handlers completed here

    # 3. Update weights
    self.model.update_weights(batch)
    self.version += 1

    # 4. Fire post-weight-update event (BLOCKS)
    bus.send(
        WorkflowEvents.POST_WEIGHT_UPDATE,
        sender=self,
        version=self.version,
    )
    # ✅ All handlers completed here
```

______________________________________________________________________

## Async Use Case: AReaL's Existing Async Workflow

Your `WorkflowExecutor` already uses `asyncio` for rollout generation:

```python
# areal/core/workflow_executor.py
async def arun_episode(self, workflow, *args, **kwargs):
    # This is already async!
    pass
```

**You can mix sync events and async workflows**:

```python
class WorkflowExecutor:
    async def arun_episode(self, workflow, *args, **kwargs):
        # Async workflow execution
        result = await workflow.arun_episode(*args, **kwargs)

        # Fire sync event when done (from async context)
        bus.send(
            WorkflowEvents.ROLLOUT_COMPLETED,
            sender=self,
            result=result,
        )
        # Event handlers run synchronously

        return result
```

**Or use async events if handlers need async operations**:

```python
class WorkflowExecutor:
    async def arun_episode(self, workflow, *args, **kwargs):
        # Fire async event
        await bus.send_async(
            WorkflowEvents.ROLLOUT_COMPLETED,
            sender=self,
            result=result,
        )
        # Async handlers can await operations

        return result
```

______________________________________________________________________

## Summary: Answers to Your Questions

### Q1: How do handlers access queue, cache, inference_engine?

**A**: Use **class-based handlers with dependency injection** (see
HANDLER_DEPENDENCY_PATTERNS.md)

```python
class PreWeightUpdateHandler:
    def __init__(self, queue, cache, inference_engine):
        self.queue = queue
        self.cache = cache
        self.inference_engine = inference_engine

    def __call__(self, sender, **kwargs):
        # Access via self.queue, self.cache, self.inference_engine
        pass

# Register with DI container
handler = container.pre_weight_update_handler()
bus.connect(WorkflowEvents.PRE_WEIGHT_UPDATE, handler)
```

### Q2: Are events sync/async? Does blinker support blocking mode?

**A**:

- ✅ **Yes, blinker is synchronous/blocking by default**
- ✅ **`send()` blocks until ALL handlers complete**
- ✅ **Handlers execute sequentially in registration order**
- ✅ **Also supports async via `send_async()` if needed**

**Perfect for your use case**: Ensures cache/queue cleanup completes before weight
update!

______________________________________________________________________

## Recommended Implementation

```python
# Phase 1: Sync events only (simple, blocking)
bus = EventBus(mode='local')
bus.send('pre_weight_update', version=42)  # Blocks until done

# Phase 2 (if needed): Mix sync and async
bus = EventBus(mode='local')
bus.send('pre_weight_update', version=42)  # Sync handlers
await bus.send_async('rollout_completed', result=data)  # Async handlers
```

**Start with sync (Phase 1)** - it's simpler and matches your use case perfectly! ✅

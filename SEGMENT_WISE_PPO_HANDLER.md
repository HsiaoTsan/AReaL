# Segment-Wise Decoupled PPO Handler Implementation

## Overview

This document describes the implementation of the ProxTLogprobHandler, which enables
segment-wise decoupled PPO by computing proximal policy logprobs at the prox_t version
(behavior + 1) before weight updates.

## Algorithm Background

### Problem: High Variance in Importance Weights

In standard decoupled off-policy PPO, the importance weight is:

```
importance_weight = π_prox / π_behavior
```

With staleness window = 16, tokens generated at version 0 might be trained against
policy version 15, leading to high variance in the importance weight.

### Solution: Segment-Wise Decoupled PPO

Use **prox_t** (proximate at t+1) instead of prox:

```
importance_weight = π_prox_t / π_behavior
```

Where prox_t is always exactly 1 version ahead of behavior, significantly reducing
variance.

### Challenge

Since only ONE policy version exists in rollout workers at any time, we cannot recompute
log_prob_prox_t at training time. Instead, we must compute it **before the weight
update**, when the current policy IS the prox_t for tokens from the previous version.

## Implementation Architecture

### Components

1. **Infrastructure Layer**

   - `FilterableQueue.scan()` - Non-destructive queue iteration
   - `FilterableQueue.scan_and_update()` - In-place queue updates
   - `ListCache.scan()` - Non-destructive cache iteration
   - `ListCache.scan_and_update()` - In-place cache updates
   - `WorkflowEvents.BEFORE_WEIGHT_UPDATE` - Event fired before weight update
   - `WorkflowEvents.AFTER_WEIGHT_UPDATE` - Event fired after weight update

1. **Handler Layer**

   - `ProxTLogprobHandler` - Event handler that recomputes prox_t logprobs
   - Registered in `app_container` as singleton
   - Injected with `output_queue` and `result_cache`

1. **Engine Layer**

   - `SGLangEngine._update_weights()` - Fires BEFORE/AFTER_WEIGHT_UPDATE events
   - `RemoteInfEngine.update_weights_from_disk()` - Fires BEFORE/AFTER_WEIGHT_UPDATE
     events
   - Must implement `recompute_output_logprobs_sync(ids: list[int]) -> list[float]`

### Data Flow

```
1. Training loop calls engine.update_weights(meta)
   ↓
2. Engine fires BEFORE_WEIGHT_UPDATE event
   ↓
3. ProxTLogprobHandler.on_before_weight_update()
   ├─ Scans output_queue for tokens with version = current - 1
   ├─ Scans result_cache for tokens with version = current - 1
   ├─ Filters out tokens already recomputed (check _recompute_version)
   ├─ Calls engine.recompute_output_logprobs_sync(ids)
   ├─ Patches proximal_logprobs_t[0, output_indices] = new_logprobs
   └─ Marks as recomputed (_recompute_version = current)
   ↓
4. Engine applies weight update
   ↓
5. Engine fires AFTER_WEIGHT_UPDATE event
   ↓
6. Training continues with patched prox_t values
```

## Code Structure

### File Organization

```
areal/
├── infrastructure/
│   ├── events.py           # WorkflowEvents with BEFORE/AFTER_WEIGHT_UPDATE
│   ├── queue.py            # FilterableQueue with scan methods
│   └── cache.py            # ListCache with scan methods
├── handlers/
│   ├── __init__.py
│   └── prox_t_handler.py   # ProxTLogprobHandler implementation
├── core/
│   ├── app_container.py    # Container with prox_t_handler provider
│   └── remote_inf_engine.py  # Fires weight update events
└── experimental/
    └── sglang_engine.py    # Fires weight update events
```

### Key Methods

#### 1. FilterableQueue.scan()

```python
def scan(self, predicate: Callable[[T], bool] | None = None) -> list[T]:
    """Non-destructive scan of queue contents."""
    with self._lock:
        matches = []
        if hasattr(self._queue, 'queue'):
            for item in self._queue.queue:
                if predicate is None or predicate(item):
                    matches.append(item)
        return matches
```

#### 2. FilterableQueue.scan_and_update()

```python
def scan_and_update(self, update_fn: Callable[[T], T | None]) -> int:
    """Scan and update queue items in-place."""
    with self._lock:
        updated_count = 0
        if hasattr(self._queue, 'queue'):
            new_deque = []
            for item in self._queue.queue:
                result = update_fn(item)
                if result is not None:
                    new_deque.append(result)
                    updated_count += 1
            self._queue.queue.clear()
            self._queue.queue.extend(new_deque)
        return updated_count
```

#### 3. ProxTLogprobHandler.on_before_weight_update()

```python
def on_before_weight_update(self, sender, **kwargs):
    """Handle BEFORE_WEIGHT_UPDATE event."""
    inference_engine = sender
    current_version = kwargs.get("current_version")
    target_version = current_version - 1

    total_patched = 0

    # Scan and update output_queue
    def process_item(item):
        nonlocal total_patched
        if self._try_recompute_item(item, target_version, current_version, inference_engine):
            total_patched += 1
        return item  # Keep all items

    self.output_queue.scan_and_update(process_item)
    self.result_cache.scan_and_update(process_item)

    logger.info(f"Recomputed prox_t for {total_patched} items")
```

#### 4. ProxTLogprobHandler.\_try_recompute_item()

```python
def _try_recompute_item(self, td, target_version, current_version, engine):
    """Try to recompute prox_t logprobs for an item."""
    # Check if already recomputed
    if td.get('_recompute_version') == current_version:
        return False

    # Extract fields
    versions = td['versions'][0]
    loss_mask = td['loss_mask'][0]

    # Find output tokens with target version
    output_indices = [i for i in range(len(versions))
                      if loss_mask[i] == 1 and versions[i] == target_version]

    if not output_indices:
        return False

    # Recompute logprobs for full sequence
    input_ids = td['input_ids'][0].tolist()
    new_logprobs = engine.recompute_output_logprobs_sync(input_ids)

    # Patch prox_t at output positions
    for idx in output_indices:
        td['proximal_logprobs_t'][0, idx] = new_logprobs[idx]

    # Mark as recomputed
    td['_recompute_version'] = current_version
    return True
```

#### 5. Engine Event Firing

```python
# In SGLangEngine._update_weights()
def _update_weights(self, meta):
    # Fire BEFORE event
    from areal.infrastructure import WorkflowEvents, get_event_bus
    bus = get_event_bus()
    current_version = self.get_version()
    bus.send(
        WorkflowEvents.BEFORE_WEIGHT_UPDATE,
        sender=self,
        current_version=current_version,
        next_version=meta.model_version,
    )

    # Apply weight update
    self.engine.update_weights_from_disk(model_path=meta.path)
    self.set_version(meta.model_version)

    # Fire AFTER event
    bus.send(
        WorkflowEvents.AFTER_WEIGHT_UPDATE,
        sender=self,
        new_version=meta.model_version,
    )
```

## Usage

### Basic Setup

```python
from areal.core.app_container import app_container
from areal.infrastructure import initialize_infrastructure, initialize_event_bus

# 1. Initialize infrastructure
initialize_event_bus(mode='local')
initialize_infrastructure({
    'max_queue_size': 10240,
    'fire_queue_events': False,
    'fire_cache_events': False,
})

# 2. Get and register the handler
handler = app_container.prox_t_handler()
handler.register()

# 3. Create engine (will fire events on weight updates)
engine = app_container.create_sglang_engine(config, engine_args)
engine.initialize()

# 4. Training loop
for step in range(num_steps):
    # ... rollout ...

    # Weight update triggers handler automatically
    engine.update_weights(weight_meta)  # Handler fires automatically!

    # ... training ...
```

### Automatic vs Manual Registration

#### Automatic (Recommended)

```python
# Handler registers itself when accessed
handler = app_container.prox_t_handler()
handler.register()
```

#### Manual

```python
from areal.handlers import ProxTLogprobHandler

# Create with manual dependency injection
handler = ProxTLogprobHandler(
    output_queue=app_container.async_task_output_queue(),
    result_cache=app_container.async_task_result_cache(),
)
handler.register()
```

## Engine Requirements

For the handler to work, inference engines must implement:

```python
class InferenceEngine:
    def recompute_output_logprobs_sync(self, input_ids: list[int]) -> list[float]:
        """
        Recompute output logprobs for a sequence.

        Parameters
        ----------
        input_ids : list[int]
            Full sequence including prompt and completion

        Returns
        -------
        list[float]
            Logprobs for each token position
        """
        raise NotImplementedError
```

## Data Structure Requirements

Rollout results (TensorDict) must contain:

```python
{
    'input_ids': Tensor[batch, seq_len],      # Token IDs
    'versions': Tensor[batch, seq_len],       # Policy version per token
    'loss_mask': Tensor[batch, seq_len],      # 1 for output tokens, 0 for prompt
    'proximal_logprobs_t': Tensor[batch, seq_len],  # Prox_t logprobs (to be patched)
    'attention_mask': Tensor[batch, seq_len], # Optional: for valid length
    '_recompute_version': Tensor[batch, 1],   # Tracking field (auto-added)
}
```

## Benefits

### 1. **Decoupled Business Logic**

Handler is completely separate from core infrastructure:

- Infrastructure: Generic queue/cache/event system
- Handler: Business logic for segment-wise PPO
- Easy to disable/enable/modify without touching infrastructure

### 2. **Minimal Changes to Existing Code**

Only additions, no modifications to existing rollout/training code:

- AsyncTaskRunner: Unchanged
- WorkflowExecutor: Unchanged (uses old code for reference)
- Engines: Only added event firing (non-breaking)

### 3. **Testable**

Easy to test in isolation:

```python
def test_prox_t_handler():
    # Mock queue and cache
    mock_queue = Mock(spec=FilterableQueue)
    mock_cache = Mock(spec=ListCache)

    # Create handler
    handler = ProxTLogprobHandler(mock_queue, mock_cache)

    # Test event handling
    mock_engine = Mock()
    mock_engine.get_version.return_value = 5
    handler.on_before_weight_update(mock_engine, current_version=5)

    # Verify scans were called
    mock_queue.scan_and_update.assert_called_once()
    mock_cache.scan_and_update.assert_called_once()
```

### 4. **Observable**

Easy to monitor via logging:

```
[ProxTHandler] Processing prox_t recomputation: current_version=5, target_version=4
[ProxTHandler] Recomputation complete: candidates=128, patched=96
```

### 5. **Configurable**

Can be enabled/disabled per experiment:

```python
# Enable for segment-wise PPO
if config.use_segment_wise_ppo:
    handler = app_container.prox_t_handler()
    handler.register()

# Disable for standard PPO
# (just don't register the handler)
```

## Performance Considerations

### Thread Safety

- All queue/cache scans are protected by locks
- No race conditions during recomputation
- Safe for concurrent rollouts

### Memory

- Scans do not copy queue/cache contents
- Updates happen in-place
- Minimal memory overhead

### Latency

- Recomputation happens BEFORE weight update (blocking)
- Only processes tokens from previous version (target_version = current - 1)
- Parallelizable if engine supports batch recomputation

## Debugging

### Enable Detailed Logging

```python
import logging
logging.getLogger('areal.handlers.prox_t_handler').setLevel(logging.DEBUG)
```

### Check Handler Registration

```python
from areal.infrastructure import get_event_bus

bus = get_event_bus()
print(bus.has_receivers('before-weight-update'))  # Should be True
```

### Monitor Recomputation Stats

```python
# Handler logs:
# - Number of candidates (tokens needing recomputation)
# - Number patched (successfully recomputed)
# - Any errors during recomputation
```

## Future Enhancements

### 1. Batch Recomputation

```python
# Collect all sequences needing recomputation
sequences = [item['input_ids'] for item in candidates]

# Batch recompute (if engine supports)
batch_logprobs = engine.recompute_output_logprobs_batch(sequences)
```

### 2. Async Recomputation

```python
# Non-blocking recomputation in background
async def on_before_weight_update_async(self, sender, **kwargs):
    await asyncio.gather(*[
        self._recompute_async(item) for item in candidates
    ])
```

### 3. Distributed Events

```python
# Fire events across nodes (Phase 2)
bus = EventBus(mode='distributed')
bus.send(WorkflowEvents.BEFORE_WEIGHT_UPDATE, ...)  # Propagates to all nodes
```

## Summary

The ProxTLogprobHandler demonstrates how our new infrastructure enables clean separation
of concerns:

- **Infrastructure**: Generic, reusable queue/cache/event primitives
- **Handler**: Business logic for segment-wise PPO algorithm
- **Minimal invasiveness**: Only adds event firing, no core changes
- **Fully testable**: All components independently testable
- **DI-based**: All dependencies injected via container

This architecture makes it easy to add/remove/modify business logic without touching the
core infrastructure.

# Distributed Event System Recommendation for AReaL

## Problem Statement

**Blinker does NOT support distributed events** - it only works within a single Python
process.

From the official description: "A fast Python **in-process** signal/event dispatching
system"

## Recommended Solution: Hybrid Approach

### Phase 1: Local Events with blinker ✅

Use blinker for in-process events (same node, different threads/coroutines)

### Phase 2: Distributed Events with Redis Pub/Sub or ZMQ ✅

Extend to distributed events when needed across multiple nodes

______________________________________________________________________

## Architecture: Two-Tier Event System

```
┌─────────────────────────────────────────────────────────────┐
│                    Node 1 (Model Worker)                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Local Event Bus (blinker)                    │  │
│  │  • queue-item-added                                  │  │
│  │  • cache-updated                                     │  │
│  │  • task-completed                                    │  │
│  └─────────────────┬────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Distributed Event Bridge                        │  │
│  │  • Subscribes to local events                        │  │
│  │  • Publishes to Redis Pub/Sub / ZMQ                  │  │
│  └─────────────────┬────────────────────────────────────┘  │
│                    │                                        │
└────────────────────┼────────────────────────────────────────┘
                     │
                     ▼
     ╔══════════════════════════════════════════╗
     ║   Distributed Event Bus                  ║
     ║   (Redis Pub/Sub or ZMQ Pub/Sub)         ║
     ║                                          ║
     ║   Topics:                                ║
     ║   • model-version-updated               ║
     ║   • training-batch-ready                ║
     ║   • rollout-completed                   ║
     ╚══════════════════════════════════════════╝
                     │
                     ▼
┌────────────────────┼────────────────────────────────────────┐
│                    │                                        │
│  ┌─────────────────▼────────────────────────────────────┐  │
│  │      Distributed Event Bridge                        │  │
│  │  • Receives from Redis Pub/Sub / ZMQ                 │  │
│  │  • Dispatches to local event handlers                │  │
│  └─────────────────┬────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Local Event Bus (blinker)                    │  │
│  │  • model-version-updated (from node 1)               │  │
│  │  • training-batch-ready (from node 1)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│                    Node 2 (Rollout Worker)                  │
└─────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Implementation

### Option A: Redis Pub/Sub (Recommended for Simplicity)

**Why Redis Pub/Sub?**

- ✅ Simple to integrate
- ✅ Lightweight protocol
- ✅ Already familiar if using Redis for queues
- ✅ Built-in pub/sub support
- ⚠️ Fire-and-forget (no delivery guarantees)

**Installation**:

```bash
pip install redis blinker
```

**Implementation**:

```python
# areal/infrastructure/events.py
from blinker import signal
import redis
import pickle
import threading
from typing import Callable, Literal, Any
import logging

logger = logging.getLogger(__name__)

class HybridEventBus:
    """Two-tier event bus: local (blinker) + distributed (Redis Pub/Sub)"""

    def __init__(
        self,
        mode: Literal['local', 'distributed'] = 'local',
        redis_url: str = 'redis://localhost:6379/0',
        distributed_events: list[str] | None = None,
    ):
        """
        Args:
            mode: 'local' for single node, 'distributed' for multi-node
            redis_url: Redis connection URL (only used in distributed mode)
            distributed_events: List of event names that should propagate across nodes
        """
        self.mode = mode
        self.distributed_events = set(distributed_events or [])

        # Local event bus (blinker)
        self._local_signals = {}

        # Distributed event bus (Redis Pub/Sub)
        if mode == 'distributed':
            self._redis = redis.from_url(redis_url)
            self._pubsub = self._redis.pubsub()
            self._listener_thread = None
            self._distributed_handlers = {}

    def signal(self, name: str):
        """Get or create a local signal"""
        if name not in self._local_signals:
            self._local_signals[name] = signal(name)
        return self._local_signals[name]

    def connect(self, event_name: str, handler: Callable, sender=None, distributed: bool = False):
        """
        Connect handler to event

        Args:
            event_name: Name of the event to listen for
            handler: Callback function
            sender: Optional sender filter (blinker feature)
            distributed: If True, also listen for this event from remote nodes
        """
        # Always connect to local signal
        self.signal(event_name).connect(handler, sender=sender)

        # If distributed mode and distributed=True, also subscribe to Redis
        if self.mode == 'distributed' and distributed:
            if event_name not in self._distributed_handlers:
                self._distributed_handlers[event_name] = []
                self._pubsub.subscribe(event_name)
                logger.info(f"Subscribed to distributed event: {event_name}")

                # Start listener thread if not running
                if self._listener_thread is None:
                    self._listener_thread = threading.Thread(
                        target=self._listen_distributed,
                        daemon=True,
                        name='EventBusListener'
                    )
                    self._listener_thread.start()

            self._distributed_handlers[event_name].append(handler)

    def send(self, event_name: str, sender=None, distributed: bool = False, **kwargs):
        """
        Send event

        Args:
            event_name: Name of the event
            sender: Event sender
            distributed: If True, propagate to remote nodes
            **kwargs: Event data
        """
        # Always send locally (in-process)
        self.signal(event_name).send(sender, **kwargs)

        # If distributed mode and distributed=True, also publish to Redis
        if self.mode == 'distributed' and distributed:
            try:
                payload = pickle.dumps({'sender': str(sender), 'data': kwargs})
                self._redis.publish(event_name, payload)
                logger.debug(f"Published distributed event: {event_name}")
            except Exception as e:
                logger.error(f"Failed to publish distributed event {event_name}: {e}")

    def _listen_distributed(self):
        """Background thread listening for distributed events"""
        logger.info("Started distributed event listener")
        try:
            for message in self._pubsub.listen():
                if message['type'] == 'message':
                    event_name = message['channel'].decode()

                    try:
                        payload = pickle.loads(message['data'])
                        sender = payload.get('sender')
                        data = payload.get('data', {})

                        # Dispatch to local handlers
                        handlers = self._distributed_handlers.get(event_name, [])
                        for handler in handlers:
                            try:
                                handler(sender=sender, **data)
                            except Exception as e:
                                logger.error(f"Error in distributed event handler: {e}")
                    except Exception as e:
                        logger.error(f"Failed to process distributed event {event_name}: {e}")
        except Exception as e:
            logger.error(f"Distributed event listener crashed: {e}")

    def close(self):
        """Cleanup resources"""
        if self.mode == 'distributed':
            self._pubsub.close()
            self._redis.close()


# Convenience event namespaces
class QueueEvents:
    """Queue-related events (local only)"""
    ITEM_ADDED = 'queue-item-added'
    ITEM_REMOVED = 'queue-item-removed'
    ITEM_FILTERED = 'queue-item-filtered'

class CacheEvents:
    """Cache-related events (local only)"""
    ITEM_ADDED = 'cache-item-added'
    ITEM_REMOVED = 'cache-item-removed'
    CACHE_CLEARED = 'cache-cleared'

class WorkflowEvents:
    """Workflow lifecycle events (can be distributed)"""
    ROLLOUT_STARTED = 'rollout-started'        # local
    ROLLOUT_COMPLETED = 'rollout-completed'    # local
    BATCH_READY = 'batch-ready'                # local

    # Distributed events (cross-node)
    MODEL_VERSION_UPDATED = 'model-version-updated'
    TRAINING_STEP_COMPLETED = 'training-step-completed'
    CAPACITY_CHANGED = 'capacity-changed'


# Global event bus instance
event_bus: HybridEventBus | None = None

def get_event_bus() -> HybridEventBus:
    """Get the global event bus instance"""
    global event_bus
    if event_bus is None:
        raise RuntimeError("Event bus not initialized. Call initialize_event_bus() first.")
    return event_bus

def initialize_event_bus(
    mode: Literal['local', 'distributed'] = 'local',
    redis_url: str = 'redis://localhost:6379/0',
):
    """Initialize the global event bus"""
    global event_bus
    event_bus = HybridEventBus(
        mode=mode,
        redis_url=redis_url,
        distributed_events=[
            WorkflowEvents.MODEL_VERSION_UPDATED,
            WorkflowEvents.TRAINING_STEP_COMPLETED,
            WorkflowEvents.CAPACITY_CHANGED,
        ]
    )
    logger.info(f"Initialized event bus in {mode} mode")
    return event_bus
```

**Usage Example**:

```python
# In your application startup (areal/api/controller_api.py)
from areal.infrastructure.events import initialize_event_bus, get_event_bus, WorkflowEvents

# Initialize event bus
initialize_event_bus(mode='distributed', redis_url='redis://localhost:6379/0')

# Get event bus instance
bus = get_event_bus()

# Connect handlers
def on_model_updated(sender, version, **kwargs):
    print(f"Model updated to version {version}")

bus.connect(
    WorkflowEvents.MODEL_VERSION_UPDATED,
    on_model_updated,
    distributed=True  # Listen for events from remote nodes
)

# Send events
bus.send(
    WorkflowEvents.MODEL_VERSION_UPDATED,
    sender=self,
    distributed=True,  # Propagate to remote nodes
    version=42,
)
```

______________________________________________________________________

### Option B: ZeroMQ Pub/Sub (Recommended for Performance)

**Why ZeroMQ?**

- ✅ **You already use ZMQ** in AReaL (request_reply_stream.py, push_pull_stream.py)
- ✅ Extremely fast (zero-copy, kernel bypass)
- ✅ No external broker needed
- ✅ Flexible transport (TCP, IPC, inproc)
- ✅ Built-in patterns (PUB/SUB, PUSH/PULL, REQ/REP)

**Installation**:

```bash
pip install pyzmq blinker
# Already installed in your project!
```

**Implementation**:

```python
# areal/infrastructure/events_zmq.py
from blinker import signal
import zmq
import pickle
import threading
from typing import Callable, Literal
import logging

logger = logging.getLogger(__name__)

class ZMQEventBus:
    """Two-tier event bus: local (blinker) + distributed (ZMQ Pub/Sub)"""

    def __init__(
        self,
        mode: Literal['local', 'distributed'] = 'local',
        pub_endpoint: str = "tcp://*:5556",
        sub_endpoints: list[str] | None = None,
    ):
        """
        Args:
            mode: 'local' for single node, 'distributed' for multi-node
            pub_endpoint: ZMQ endpoint for publishing events (e.g., "tcp://*:5556")
            sub_endpoints: List of ZMQ endpoints to subscribe to (e.g., ["tcp://node1:5556"])
        """
        self.mode = mode
        self._local_signals = {}

        if mode == 'distributed':
            self.context = zmq.Context()

            # Publisher socket (send events to other nodes)
            self.publisher = self.context.socket(zmq.PUB)
            self.publisher.bind(pub_endpoint)
            logger.info(f"ZMQ publisher bound to {pub_endpoint}")

            # Subscriber socket (receive events from other nodes)
            self.subscriber = self.context.socket(zmq.SUB)
            for endpoint in (sub_endpoints or []):
                self.subscriber.connect(endpoint)
                logger.info(f"ZMQ subscriber connected to {endpoint}")

            self._distributed_handlers = {}
            self._listener_thread = None

    def signal(self, name: str):
        """Get or create a local signal"""
        if name not in self._local_signals:
            self._local_signals[name] = signal(name)
        return self._local_signals[name]

    def connect(self, event_name: str, handler: Callable, sender=None, distributed: bool = False):
        """Connect handler to event"""
        # Local connection
        self.signal(event_name).connect(handler, sender=sender)

        # Distributed subscription
        if self.mode == 'distributed' and distributed:
            if event_name not in self._distributed_handlers:
                self._distributed_handlers[event_name] = []
                self.subscriber.setsockopt_string(zmq.SUBSCRIBE, event_name)
                logger.info(f"Subscribed to distributed event: {event_name}")

                if self._listener_thread is None:
                    self._listener_thread = threading.Thread(
                        target=self._listen_distributed,
                        daemon=True,
                        name='ZMQEventListener'
                    )
                    self._listener_thread.start()

            self._distributed_handlers[event_name].append(handler)

    def send(self, event_name: str, sender=None, distributed: bool = False, **kwargs):
        """Send event"""
        # Local dispatch
        self.signal(event_name).send(sender, **kwargs)

        # Distributed publish
        if self.mode == 'distributed' and distributed:
            try:
                payload = pickle.dumps({'sender': str(sender), 'data': kwargs})
                self.publisher.send_multipart([event_name.encode(), payload])
                logger.debug(f"Published distributed event: {event_name}")
            except Exception as e:
                logger.error(f"Failed to publish distributed event {event_name}: {e}")

    def _listen_distributed(self):
        """Background thread for receiving distributed events"""
        logger.info("Started ZMQ distributed event listener")
        try:
            while True:
                event_name_bytes, payload_bytes = self.subscriber.recv_multipart()
                event_name = event_name_bytes.decode()

                try:
                    payload = pickle.loads(payload_bytes)
                    sender = payload.get('sender')
                    data = payload.get('data', {})

                    # Dispatch to handlers
                    handlers = self._distributed_handlers.get(event_name, [])
                    for handler in handlers:
                        try:
                            handler(sender=sender, **data)
                        except Exception as e:
                            logger.error(f"Error in distributed event handler: {e}")
                except Exception as e:
                    logger.error(f"Failed to process distributed event {event_name}: {e}")
        except Exception as e:
            logger.error(f"ZMQ event listener crashed: {e}")

    def close(self):
        """Cleanup resources"""
        if self.mode == 'distributed':
            self.publisher.close()
            self.subscriber.close()
            self.context.term()
```

**Usage Example**:

```python
# Node 1 (Model Worker)
bus = ZMQEventBus(
    mode='distributed',
    pub_endpoint='tcp://*:5556',
    sub_endpoints=['tcp://node2:5556', 'tcp://node3:5556']
)

# Send model update event (propagates to all nodes)
bus.send(
    'model-version-updated',
    sender=self,
    distributed=True,
    version=42,
    timestamp=time.time()
)

# Node 2 (Rollout Worker)
bus = ZMQEventBus(
    mode='distributed',
    pub_endpoint='tcp://*:5556',
    sub_endpoints=['tcp://node1:5556']  # Subscribe to model worker
)

# Listen for model updates
bus.connect(
    'model-version-updated',
    on_model_updated,
    distributed=True
)
```

______________________________________________________________________

## Comparison: Redis Pub/Sub vs ZMQ

| Feature                    | Redis Pub/Sub               | ZeroMQ Pub/Sub                           |
| -------------------------- | --------------------------- | ---------------------------------------- |
| **External Dependency**    | Requires Redis server       | None (peer-to-peer)                      |
| **Performance**            | Fast (~100k msg/s)          | Extremely fast (~1M msg/s)               |
| **Setup Complexity**       | Low (install Redis)         | Low (already in AReaL)                   |
| **Protocol**               | Redis wire protocol         | TCP/IPC/inproc                           |
| **Delivery Guarantee**     | Fire-and-forget             | Fire-and-forget                          |
| **Persistence**            | None (pub/sub is ephemeral) | None (use Redis Streams for persistence) |
| **Integration with AReaL** | New dependency              | **Already used!**                        |
| **Monitoring**             | Redis CLI, RedisInsight     | ZMQ monitoring sockets                   |

______________________________________________________________________

## Recommended Approach for AReaL

### **Start with: blinker (local) + ZMQ Pub/Sub (distributed)**

**Rationale**:

1. ✅ **You already use ZMQ** - reuse existing infrastructure
1. ✅ **Higher performance** - critical for RL training loops
1. ✅ **No additional dependencies** - ZMQ already in your stack
1. ✅ **Consistent architecture** - matches existing request_reply_stream.py patterns

### **Alternative: blinker (local) + Redis Pub/Sub (distributed)**

**When to choose Redis**:

- ✅ You're already using Redis for queues
- ✅ You want centralized event monitoring via Redis CLI
- ✅ Team is more familiar with Redis than ZMQ

______________________________________________________________________

## Integration with Dependency Injection

```python
# areal/infrastructure/container.py
from dependency_injector import containers, providers
from .events_zmq import ZMQEventBus

class InfrastructureContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Event bus factory
    event_bus = providers.Singleton(
        ZMQEventBus,
        mode=config.event_bus_mode,  # 'local' or 'distributed'
        pub_endpoint=config.zmq_pub_endpoint,
        sub_endpoints=config.zmq_sub_endpoints,
    )
```

______________________________________________________________________

## Event Categories

### Local Events (blinker only)

- `queue-item-added` - Queue operations
- `cache-updated` - Cache modifications
- `task-completed` - Async task completion
- `rollout-started` - Single rollout lifecycle

### Distributed Events (blinker + ZMQ/Redis)

- `model-version-updated` - Model weight updates (training → inference)
- `training-step-completed` - Training progress (training → controller)
- `capacity-changed` - Staleness manager capacity updates
- `worker-status-changed` - Worker health/availability

______________________________________________________________________

## Migration Path

### Phase 1: Local Events Only

```python
# All events local (current single-node setup)
event_bus = ZMQEventBus(mode='local')
```

### Phase 2: Add Distributed Events

```python
# Enable distributed events for multi-node deployment
event_bus = ZMQEventBus(
    mode='distributed',
    pub_endpoint='tcp://*:5556',
    sub_endpoints=['tcp://other-node:5556']
)

# Mark specific events as distributed
bus.send('model-version-updated', distributed=True, version=42)
```

______________________________________________________________________

## Next Steps

1. **Choose backend**: ZMQ (recommended) or Redis Pub/Sub
1. **Implement `HybridEventBus`** in `areal/infrastructure/events.py`
1. **Update container** to provide event bus via DI
1. **Start with local mode** - no breaking changes
1. **Test distributed mode** - add config flag to enable
1. **Document event contracts** - which events are distributed vs local

______________________________________________________________________

## Summary

| Requirement                         | Solution                          | Protocol            |
| ----------------------------------- | --------------------------------- | ------------------- |
| **Local events** (in-process)       | **blinker**                       | N/A (in-memory)     |
| **Distributed events** (cross-node) | **ZMQ Pub/Sub** (recommended)     | TCP/IPC             |
|                                     | **Redis Pub/Sub** (alternative)   | Redis wire protocol |
| **Hybrid** (local + distributed)    | **Custom EventBus** wrapping both | Both                |

**Recommendation**: Use **blinker for local + ZMQ for distributed** to leverage your
existing infrastructure and maximize performance.

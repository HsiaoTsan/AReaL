# Proper Dependency Injection Usage

## The Question

> Why bother to implement the instantiation logics inside RemoteInfEngine.initialize()
> instead of moving them into container?

## The Answer: **You're absolutely right!**

The instantiation logic **should be in the container**, not scattered across component
classes. Here's the proper way.

## ❌ Wrong: Manual Instantiation in Component

```python
# In RemoteInfEngine.initialize() - DON'T DO THIS
if self.workflow_executor is None:
    # Component constructs its own dependencies!
    runner = app_container.async_task_runner()
    pending_results = app_container.workflow_pending_results()
    pending_inputs = app_container.workflow_pending_inputs()

    self.workflow_executor = WorkflowExecutor(
        config=self.config,
        inference_engine=self,
        runner=runner,
        pending_results=pending_results,
        pending_inputs=pending_inputs,
    )
```

**Problems**:

- ✗ Component knows about the container (tight coupling)
- ✗ Complex wiring logic scattered across files
- ✗ Hard to test (can't easily mock dependencies)
- ✗ Violates Single Responsibility Principle
- ✗ Duplicated logic in every engine class

## ✅ Right: Factory Method in Container

```python
# In app_container.py - DO THIS
@staticmethod
def create_remote_inf_engine(config, backend, workflow_executor=None):
    """
    Factory method for creating fully-wired RemoteInfEngine.

    Handles the circular dependency between engine and workflow_executor
    by creating them in phases and wiring them together.
    """
    from areal.core.remote_inf_engine import RemoteInfEngine

    # Phase 1: Create engine
    engine = RemoteInfEngine(
        config=config,
        backend=backend,
        workflow_executor=workflow_executor,
    )

    # Phase 2: Create and wire workflow_executor
    if workflow_executor is None:
        engine.workflow_executor = app_container.create_workflow_executor_for_engine(
            config=config,
            inference_engine=engine,
        )

    return engine
```

**Benefits**:

- ✓ Centralized wiring logic
- ✓ Component is clean and simple
- ✓ Easy to test (container is mockable)
- ✓ Clear separation of concerns
- ✓ No code duplication

## Proper Usage

### Creating Engines

```python
from areal.core.app_container import app_container

# Option 1: Use factory method (RECOMMENDED)
engine = app_container.create_remote_inf_engine(config, backend)
engine.initialize()

# Option 2: Use factory for SGLang
engine = app_container.create_sglang_engine(config, engine_args={'model_path': '...'})
engine.initialize()

# Option 3: Pre-create workflow_executor for custom logic
workflow_executor = app_container.create_workflow_executor_for_engine(config, my_engine)
engine = app_container.create_remote_inf_engine(config, backend, workflow_executor)
engine.initialize()
```

### Legacy Code (Backward Compatible)

Existing code still works (fallback logic in engine classes):

```python
# Old way - still works but not recommended
engine = RemoteInfEngine(config, backend)
engine.initialize()  # Creates workflow_executor internally if missing
```

But this should be migrated to use container factory methods.

## Why Circular Dependency?

The circular dependency between engine and workflow_executor exists because:

```python
# WorkflowExecutor needs engine reference
class WorkflowExecutor:
    def __init__(self, inference_engine: InferenceEngine, ...):
        self.inference_engine = inference_engine  # Needs engine!

# InferenceEngine needs workflow_executor
class RemoteInfEngine:
    def __init__(self, workflow_executor: WorkflowExecutor):
        self.workflow_executor = workflow_executor  # Needs executor!
```

**Solution**: Two-phase construction in container:

1. Create engine without executor
1. Create executor with engine reference
1. Wire them together

## Container Factory Methods

### `create_workflow_executor_for_engine(config, inference_engine)`

Creates WorkflowExecutor with all dependencies:

```python
@staticmethod
def create_workflow_executor_for_engine(config, inference_engine):
    """
    Factory for creating WorkflowExecutor for an engine.

    - Gets singleton AsyncTaskRunner
    - Initializes runner if needed
    - Creates new caches for this executor
    - Wires everything together
    """
    runner = app_container.async_task_runner()

    if not hasattr(runner, 'thread') or runner.thread is None:
        logger = getattr(inference_engine, 'logger', None)
        runner.initialize(logger=logger)

    pending_results = app_container.workflow_pending_results()
    pending_inputs = app_container.workflow_pending_inputs()

    return WorkflowExecutor(
        config=config,
        inference_engine=inference_engine,
        runner=runner,
        pending_results=pending_results,
        pending_inputs=pending_inputs,
    )
```

### `create_remote_inf_engine(config, backend, workflow_executor=None)`

Creates fully-wired RemoteInfEngine:

```python
@staticmethod
def create_remote_inf_engine(config, backend, workflow_executor=None):
    """Factory for RemoteInfEngine with workflow_executor."""
    engine = RemoteInfEngine(config, backend, workflow_executor)

    if workflow_executor is None:
        engine.workflow_executor = app_container.create_workflow_executor_for_engine(
            config, engine
        )

    return engine
```

### `create_sglang_engine(config, engine_args=None, workflow_executor=None)`

Creates fully-wired SGLangEngine:

```python
@staticmethod
def create_sglang_engine(config, engine_args=None, workflow_executor=None):
    """Factory for SGLangEngine with workflow_executor."""
    engine = SGLangEngine(config, engine_args, workflow_executor)

    if workflow_executor is None:
        engine.workflow_executor = app_container.create_workflow_executor_for_engine(
            config, engine
        )

    return engine
```

## Migration Guide

### Step 1: Update Engine Creation

**Before**:

```python
from areal.core.remote_inf_engine import RemoteInfEngine

engine = RemoteInfEngine(config, backend)
engine.initialize()
```

**After**:

```python
from areal.core.app_container import app_container

engine = app_container.create_remote_inf_engine(config, backend)
engine.initialize()
```

### Step 2: Remove Manual Instantiation Logic

Once all code uses container factories, remove the fallback logic from engine classes:

```python
# In RemoteInfEngine.initialize()
def initialize(self, ...):
    # Remove this entire block:
    # if self.workflow_executor is None:
    #     ... manual instantiation ...

    # Just use what was injected:
    if self.workflow_executor is None:
        raise RuntimeError(
            "WorkflowExecutor must be injected. "
            "Use app_container.create_remote_inf_engine() to create engine."
        )

    self.workflow_executor.initialize(...)
```

### Step 3: Update Tests

**Before**:

```python
def test_engine():
    engine = RemoteInfEngine(config, backend)
    engine.initialize()
```

**After**:

```python
def test_engine():
    # Use container with test configuration
    app_container.config.from_dict({'max_queue_size': 10})
    engine = app_container.create_remote_inf_engine(config, backend)
    engine.initialize()

# Or mock the workflow_executor
def test_engine_with_mock():
    mock_executor = Mock(spec=WorkflowExecutor)
    engine = app_container.create_remote_inf_engine(config, backend, mock_executor)
    engine.initialize()
```

## Testing with Container

### Override for Testing

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def test_container():
    """Container with test configuration."""
    from areal.core.app_container import app_container

    # Reset singletons
    app_container.async_task_runner.reset()
    app_container.async_task_input_queue.reset()
    app_container.async_task_output_queue.reset()

    # Configure for tests
    app_container.config.from_dict({
        'max_queue_size': 10,
        'enable_rollout_tracing': False,
    })

    return app_container

def test_engine_with_container(test_container):
    """Test using container."""
    engine = test_container.create_sglang_engine(config)
    engine.initialize()
    # Test...
```

## Benefits Summary

### Before (Manual Instantiation)

- ✗ Logic scattered across engine classes
- ✗ Tight coupling to container
- ✗ Hard to test
- ✗ Code duplication
- ✗ Violates SRP

### After (Container Factory)

- ✓ Centralized in container
- ✓ Loose coupling
- ✓ Easy to test
- ✓ No duplication
- ✓ Follows DI principles

## Conclusion

The container factory methods (`create_remote_inf_engine`, `create_sglang_engine`) are
the **proper way** to create engines with all dependencies wired.

The fallback logic in engine classes is only for **backward compatibility** and should
eventually be removed once all code migrates to using container factories.

## File Locations

- **Container**: `areal/core/app_container.py` - All factory methods here
- **RemoteInfEngine**: `areal/core/remote_inf_engine.py` - Simplified (eventually)
- **SGLangEngine**: `areal/experimental/sglang_engine.py` - Simplified (eventually)
- **Tests**: Use `app_container.create_*()` methods

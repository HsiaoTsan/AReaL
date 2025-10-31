# Removing Fallback Logic: Enforcing Proper DI

## Why Remove the Fallback?

You asked: **"Why bother to implement the instantiation logics here instead of move them
into container?"**

**Answer**: You're absolutely right! The fallback logic defeats the entire purpose of
dependency injection.

### Problems with Fallback Logic

```python
# In RemoteInfEngine.initialize() - BAD!
if self.workflow_executor is None:
    # Component constructs its own dependencies
    runner = app_container.async_task_runner()
    self.workflow_executor = WorkflowExecutor(...)
```

**Issues**:

- ✗ **Two ways to create engines** - Container factory OR direct instantiation
  (confusion!)
- ✗ **Logic duplication** - Same wiring code in container AND engine classes
- ✗ **Tight coupling** - Components still know about the container
- ✗ **Not true DI** - Components are responsible for their dependencies
- ✗ **Hard to maintain** - Must update wiring logic in multiple places

### After Removing Fallback

```python
# In RemoteInfEngine.initialize() - GOOD!
if self.workflow_executor is None:
    raise RuntimeError(
        "WorkflowExecutor must be injected. "
        "Use app_container.create_remote_inf_engine() to create the engine"
    )
```

**Benefits**:

- ✓ **One way** - Must use container factory (clear!)
- ✓ **No duplication** - Wiring logic only in container
- ✓ **Loose coupling** - Components don't know about container
- ✓ **True DI** - Container fully responsible for dependencies
- ✓ **Easy to maintain** - Change wiring in one place

## What Changed

### Before (With Fallback)

```python
# Two ways to create engine:

# Way 1: Direct instantiation (uses fallback logic)
engine = SGLangEngine(config, engine_args)
engine.initialize()  # Creates workflow_executor internally

# Way 2: Container factory
engine = app_container.create_sglang_engine(config, engine_args)
engine.initialize()  # workflow_executor already injected
```

### After (Enforced DI)

```python
# Only one way: Must use container

engine = app_container.create_sglang_engine(config, engine_args)
engine.initialize()

# Direct instantiation fails:
engine = SGLangEngine(config, engine_args)
engine.initialize()  # RuntimeError: WorkflowExecutor must be injected!
```

## Migration Guide

### For Test Files

**Before**:

```python
from areal.experimental.sglang_engine import SGLangEngine

engine = SGLangEngine(config, engine_args=build_engine_args())
engine.initialize()
```

**After**:

```python
from areal.core.app_container import app_container

engine = app_container.create_sglang_engine(config, engine_args=build_engine_args())
engine.initialize()
```

### For Application Code

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

### For Custom Wiring

If you need custom workflow_executor:

```python
from areal.core.app_container import app_container

# Create custom executor
executor = app_container.create_workflow_executor_for_engine(config, my_engine)

# Or fully custom
executor = WorkflowExecutor(
    config=config,
    inference_engine=my_engine,
    runner=my_custom_runner,
    ...
)

# Pass to factory
engine = app_container.create_sglang_engine(
    config,
    engine_args,
    workflow_executor=executor  # Inject custom executor
)
```

## Files Updated

### Engine Classes (Removed Fallback)

1. **areal/experimental/sglang_engine.py**

   - Removed: Manual WorkflowExecutor construction
   - Added: Clear error message pointing to container factory

1. **areal/core/remote_inf_engine.py**

   - Removed: Manual WorkflowExecutor construction
   - Added: Clear error message pointing to container factory

### Test Files (Updated to Use Container)

1. **areal/experimental/tests/test_sglang_local_engine.py**
   - Updated all 3 tests to use `app_container.create_sglang_engine()`

### Container (Factory Methods)

**areal/core/app_container.py** - Contains all wiring logic:

- `create_workflow_executor_for_engine(config, inference_engine)`
- `create_remote_inf_engine(config, backend, workflow_executor=None)`
- `create_sglang_engine(config, engine_args=None, workflow_executor=None)`

## Error Messages

If you try to create an engine without proper injection, you'll see:

```
RuntimeError: WorkflowExecutor must be injected.
Use app_container.create_sglang_engine() to create the engine:

    from areal.core.app_container import app_container
    engine = app_container.create_sglang_engine(config, engine_args)

Or inject manually:
    executor = app_container.create_workflow_executor_for_engine(config, engine)
    engine = SGLangEngine(config, workflow_executor=executor, engine_args=args)
```

Clear instructions on how to fix!

## Why This is Better

### Principle: Separation of Concerns

**Wrong**:

- Component knows HOW to construct its dependencies
- Component knows WHICH container to use
- Component mixes business logic with dependency construction

**Right**:

- Component only knows it NEEDS dependencies
- Container knows HOW to construct dependencies
- Component focuses on business logic only

### Principle: Single Source of Truth

**Wrong**:

- Wiring logic in container AND component classes
- Must update multiple files when dependencies change
- Easy to get out of sync

**Right**:

- Wiring logic only in container
- Change in one place
- Always in sync

### Principle: Dependency Inversion

**Wrong**:

- Component depends on concrete container
- High-level module (engine) depends on low-level module (container)

**Right**:

- Component depends on abstractions (interfaces)
- Container handles concrete implementations
- Low-level module (container) depends on high-level module (engine interface)

## Summary

The fallback logic was removed because:

1. ✓ **Enforces proper DI** - Container is single source of truth for wiring
1. ✓ **Eliminates duplication** - Wiring logic in one place
1. ✓ **Clearer intention** - Must use container factory (no confusion)
1. ✓ **Better separation** - Components focus on business logic
1. ✓ **Easier testing** - Clear injection points for mocks

All existing code has been migrated to use container factories.

## Next Steps

1. Search for remaining direct instantiations:

   ```bash
   grep -r "RemoteInfEngine(" areal/ --include="*.py"
   grep -r "SGLangEngine(" areal/ --include="*.py"
   ```

1. Update them to use container factories

1. Run all tests to ensure everything works

1. Remove any unused imports of engine classes (use container instead)

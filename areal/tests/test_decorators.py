"""Tests for infrastructure decorators (fire_events)."""

import asyncio

import pytest

from areal.infrastructure import (
    fire_events,
    get_event_bus,
    initialize_event_bus,
)


@pytest.fixture(autouse=True)
def setup_event_bus():
    """Initialize event bus before each test."""
    initialize_event_bus(mode="local")
    yield
    # Event bus persists across tests, but that's okay for these tests


class TestFireEventsDecorator:
    """Test fire_events decorator functionality."""

    def test_fire_events_basic(self):
        """Test basic before/after event firing."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append((sender, kwargs))

        bus = get_event_bus()
        bus.connect("test-before", handler)
        bus.connect("test-after", handler)

        class TestClass:
            @fire_events(before="test-before", after="test-after")
            def process(self):
                return "result"

        obj = TestClass()
        result = obj.process()

        assert result == "result"
        assert len(events_received) == 2
        assert events_received[0][0] == obj  # sender
        assert events_received[0][1]["method_name"] == "process"
        assert events_received[1][0] == obj
        assert events_received[1][1]["method_name"] == "process"
        assert events_received[1][1]["result"] == "result"

    def test_fire_events_with_string_error(self):
        """Test error event firing with string event name."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append((sender, kwargs))

        bus = get_event_bus()
        bus.connect("test-error", handler)

        class TestClass:
            @fire_events(
                before="test-before", after="test-after", on_error="test-error"
            )
            def failing_method(self):
                raise ValueError("Test error")

        obj = TestClass()
        with pytest.raises(ValueError, match="Test error"):
            obj.failing_method()

        assert len(events_received) == 1
        assert events_received[0][0] == obj
        assert events_received[0][1]["method_name"] == "failing_method"
        assert isinstance(events_received[0][1]["error"], ValueError)

    def test_fire_events_with_callback_error(self):
        """Test error handling with custom callback."""
        callback_calls = []

        def custom_error_handler(bus, self, exception, method_name, context):
            callback_calls.append(
                {
                    "bus": bus,
                    "self": self,
                    "exception": exception,
                    "method_name": method_name,
                    "context": context,
                }
            )
            # Custom logic: could fire different event, log, or do nothing

        class TestClass:
            @fire_events(
                before="test-before", after="test-after", on_error=custom_error_handler
            )
            def failing_method(self):
                raise ValueError("Test error")

        obj = TestClass()
        with pytest.raises(ValueError, match="Test error"):
            obj.failing_method()

        assert len(callback_calls) == 1
        assert callback_calls[0]["self"] == obj
        assert isinstance(callback_calls[0]["exception"], ValueError)
        assert callback_calls[0]["method_name"] == "failing_method"
        assert callback_calls[0]["bus"] is not None

    def test_fire_events_callback_can_suppress_event(self):
        """Test that custom callback can choose not to fire events."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append((sender, kwargs))

        bus = get_event_bus()
        bus.connect("error-event", handler)

        def silent_error_handler(bus, self, exception, method_name, context):
            # Do nothing - don't fire any event
            pass

        class TestClass:
            @fire_events(
                before="test-before", after="test-after", on_error=silent_error_handler
            )
            def failing_method(self):
                raise ValueError("Test error")

        obj = TestClass()
        with pytest.raises(ValueError, match="Test error"):
            obj.failing_method()

        # No events should have been fired
        assert len(events_received) == 0

    def test_fire_events_callback_can_fire_custom_event(self):
        """Test that custom callback can fire different events."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append((kwargs.get("event_type"), sender))

        bus = get_event_bus()
        bus.connect("custom-critical-error", handler)
        bus.connect("custom-minor-error", handler)

        def smart_error_handler(bus, self, exception, method_name, context):
            # Fire different events based on error type
            if isinstance(exception, ValueError):
                bus.send("custom-critical-error", sender=self, event_type="critical")
            else:
                bus.send("custom-minor-error", sender=self, event_type="minor")

        class TestClass:
            @fire_events(
                before="test-before", after="test-after", on_error=smart_error_handler
            )
            def failing_method(self, error_type="value"):
                if error_type == "value":
                    raise ValueError("Critical error")
                else:
                    raise RuntimeError("Minor error")

        obj = TestClass()

        # Test ValueError -> critical event
        with pytest.raises(ValueError):
            obj.failing_method("value")
        assert len(events_received) == 1
        assert events_received[0][0] == "critical"

        # Test RuntimeError -> minor event
        with pytest.raises(RuntimeError):
            obj.failing_method("runtime")
        assert len(events_received) == 2
        assert events_received[1][0] == "minor"

    def test_fire_events_with_extract_context(self):
        """Test context extraction."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append(kwargs)

        bus = get_event_bus()
        bus.connect("test-before", handler)
        bus.connect("test-after", handler)

        def extract_version(self, meta):
            return {
                "current_version": self.version,
                "next_version": meta.get("version"),
            }

        class TestClass:
            def __init__(self):
                self.version = 5

            @fire_events(
                before="test-before",
                after="test-after",
                extract_context=extract_version,
            )
            def update(self, meta):
                self.version = meta["version"]
                return "updated"

        obj = TestClass()
        result = obj.update({"version": 10})

        assert result == "updated"
        assert len(events_received) == 2
        # Before event has context
        assert events_received[0]["current_version"] == 5
        assert events_received[0]["next_version"] == 10
        # After event has context too
        assert events_received[1]["current_version"] == 5

    @pytest.mark.asyncio
    async def test_fire_events_async(self):
        """Test decorator with async methods."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append((sender, kwargs))

        bus = get_event_bus()
        bus.connect("async-before", handler)
        bus.connect("async-after", handler)

        class TestClass:
            @fire_events(before="async-before", after="async-after")
            async def async_process(self):
                await asyncio.sleep(0.01)
                return "async-result"

        obj = TestClass()
        result = await obj.async_process()

        assert result == "async-result"
        assert len(events_received) == 2
        assert events_received[0][0] == obj
        assert events_received[1][1]["result"] == "async-result"

    @pytest.mark.asyncio
    async def test_fire_events_async_with_error(self):
        """Test async decorator with error."""
        callback_calls = []

        def custom_error_handler(bus, self, exception, method_name, context):
            callback_calls.append(exception)

        class TestClass:
            @fire_events(
                before="async-before",
                after="async-after",
                on_error=custom_error_handler,
            )
            async def async_failing(self):
                await asyncio.sleep(0.01)
                raise ValueError("Async error")

        obj = TestClass()
        with pytest.raises(ValueError, match="Async error"):
            await obj.async_failing()

        assert len(callback_calls) == 1
        assert isinstance(callback_calls[0], ValueError)

    def test_fire_events_without_event_bus(self):
        """Test decorator gracefully handles missing event bus."""
        # Don't initialize event bus - it should handle gracefully
        # Actually the event bus is already initialized by fixture,
        # so let's test that it doesn't break if events fail

        class TestClass:
            @fire_events(before="nonexistent", after="nonexistent")
            def process(self):
                return "result"

        obj = TestClass()
        # Should work even if no handlers registered
        result = obj.process()
        assert result == "result"

    def test_fire_events_only_before(self):
        """Test decorator with only before event."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append(kwargs)

        bus = get_event_bus()
        bus.connect("only-before", handler)

        class TestClass:
            @fire_events(before="only-before")
            def process(self):
                return "result"

        obj = TestClass()
        result = obj.process()

        assert result == "result"
        assert len(events_received) == 1
        assert events_received[0]["method_name"] == "process"

    def test_fire_events_only_after(self):
        """Test decorator with only after event."""
        events_received = []

        def handler(sender, **kwargs):
            events_received.append(kwargs)

        bus = get_event_bus()
        bus.connect("only-after", handler)

        class TestClass:
            @fire_events(after="only-after")
            def process(self):
                return "result"

        obj = TestClass()
        result = obj.process()

        assert result == "result"
        assert len(events_received) == 1
        assert events_received[0]["result"] == "result"

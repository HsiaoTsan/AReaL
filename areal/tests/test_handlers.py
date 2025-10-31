"""Tests for event handlers (ProxTLogprobHandler)."""

import pytest
import torch

from areal.handlers.prox_t_handler import ProxTLogprobHandler
from areal.infrastructure import (
    FilterableQueue,
    ListCache,
    WorkflowEvents,
    get_event_bus,
    initialize_event_bus,
)


@pytest.fixture(autouse=True)
def setup_event_bus():
    """Initialize event bus before each test."""
    initialize_event_bus(mode="local")
    yield


@pytest.fixture
def mock_engine():
    """Create mock inference engine."""

    class MockEngine:
        def __init__(self):
            self._version = 5
            self.recompute_calls = []

        def get_version(self):
            return self._version

        def recompute_output_logprobs_sync(self, input_ids):
            """Mock recomputation - returns dummy logprobs."""
            self.recompute_calls.append(input_ids)
            # Return logprobs for each token (all -1.0 for simplicity)
            return [-1.0] * len(input_ids)

    return MockEngine()


@pytest.fixture
def output_queue():
    """Create output queue for testing."""
    return FilterableQueue(name="test_output", maxsize=100)


@pytest.fixture
def result_cache():
    """Create result cache for testing."""
    return ListCache()


class TestProxTLogprobHandler:
    """Test ProxTLogprobHandler functionality."""

    def test_handler_initialization(self, output_queue, result_cache):
        """Test handler can be initialized."""
        handler = ProxTLogprobHandler(
            output_queue=output_queue,
            result_cache=result_cache,
        )
        assert handler.output_queue is output_queue
        assert handler.result_cache is result_cache
        assert handler._registered is False

    def test_handler_registration(self, output_queue, result_cache):
        """Test handler registration to event bus."""
        handler = ProxTLogprobHandler(output_queue, result_cache)

        bus = get_event_bus()
        assert not bus.has_receivers(WorkflowEvents.PRE_WEIGHT_UPDATE)

        handler.register()
        assert handler._registered is True
        assert bus.has_receivers(WorkflowEvents.PRE_WEIGHT_UPDATE)

        # Test double registration is safe
        handler.register()
        assert handler._registered is True

    def test_handler_unregistration(self, output_queue, result_cache):
        """Test handler unregistration."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        bus = get_event_bus()
        assert bus.has_receivers(WorkflowEvents.PRE_WEIGHT_UPDATE)

        handler.unregister()
        assert handler._registered is False
        # Note: blinker may still show receivers if there are other handlers

    def test_handler_ignores_engine_without_get_version(
        self, output_queue, result_cache
    ):
        """Test handler gracefully handles engine without get_version."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        class BadEngine:
            pass

        engine = BadEngine()
        bus = get_event_bus()

        # Should not crash
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=engine)

    def test_handler_ignores_engine_without_recompute_method(
        self, output_queue, result_cache
    ):
        """Test handler gracefully handles engine without recompute method."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        class EngineWithoutRecompute:
            def get_version(self):
                return 5

        engine = EngineWithoutRecompute()
        bus = get_event_bus()

        # Should not crash
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=engine)

    def test_handler_recomputes_tokens_from_previous_version(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler recomputes tokens from previous version."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Create sample tensor dict with tokens from version 4 (target_version)
        td = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "versions": torch.tensor(
                [[4, 4, 4, 5, 5]]
            ),  # First 3 from v4, last 2 from v5
            "loss_mask": torch.tensor([[0, 1, 1, 1, 1]]),  # Skip first token (prompt)
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0]]),
        }

        # Put in queue
        output_queue.put(td)

        # Fire event (current_version=5, target_version=4)
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Check recomputation was called
        assert len(mock_engine.recompute_calls) == 1
        assert mock_engine.recompute_calls[0] == [1, 2, 3, 4, 5]

        # Check that prox_t was patched for target version tokens
        # Tokens at indices 1, 2 have version=4, should be updated to -1.0
        assert td["proximal_logprobs_t"][0, 1].item() == -1.0
        assert td["proximal_logprobs_t"][0, 2].item() == -1.0
        # Tokens at indices 3, 4 have version=5, should remain -2.0
        assert td["proximal_logprobs_t"][0, 3].item() == -2.0
        assert td["proximal_logprobs_t"][0, 4].item() == -2.0

    def test_handler_recomputes_cache_items(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler recomputes items in result cache."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Create sample tensor dict
        td = {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "versions": torch.tensor([[4, 4, 4]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-3.0, -3.0, -3.0]]),
        }

        # Put in cache
        result_cache.append(td)

        # Fire event
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Check recomputation
        assert len(mock_engine.recompute_calls) == 1
        assert mock_engine.recompute_calls[0] == [10, 20, 30]

        # Check patching
        assert td["proximal_logprobs_t"][0, 0].item() == -1.0
        assert td["proximal_logprobs_t"][0, 1].item() == -1.0
        assert td["proximal_logprobs_t"][0, 2].item() == -1.0

    def test_handler_skips_already_recomputed_items(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler skips items already recomputed at current version."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Create item already recomputed at version 5
        td = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[4, 4, 4]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0, -2.0]]),
            "_recompute_version": torch.tensor([[5]]),  # Already recomputed at v5
        }

        output_queue.put(td)

        # Fire event (current_version=5)
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Should NOT recompute
        assert len(mock_engine.recompute_calls) == 0

    def test_handler_processes_multiple_items(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler processes multiple items in queue and cache."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Add items to queue
        for i in range(3):
            td = {
                "input_ids": torch.tensor([[i, i + 1, i + 2]]),
                "versions": torch.tensor([[4, 4, 4]]),
                "loss_mask": torch.tensor([[1, 1, 1]]),
                "proximal_logprobs_t": torch.tensor([[-2.0, -2.0, -2.0]]),
            }
            output_queue.put(td)

        # Add items to cache
        for i in range(2):
            td = {
                "input_ids": torch.tensor([[i + 10, i + 11]]),
                "versions": torch.tensor([[4, 4]]),
                "loss_mask": torch.tensor([[1, 1]]),
                "proximal_logprobs_t": torch.tensor([[-2.0, -2.0]]),
            }
            result_cache.append(td)

        # Fire event
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Should have recomputed 5 items (3 queue + 2 cache)
        assert len(mock_engine.recompute_calls) == 5

    def test_handler_skips_items_from_wrong_version(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler only processes items from target_version."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Add items from different versions
        # Item 1: version 3 (too old, target is 4)
        td1 = {
            "input_ids": torch.tensor([[1, 2]]),
            "versions": torch.tensor([[3, 3]]),
            "loss_mask": torch.tensor([[1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0]]),
        }
        output_queue.put(td1)

        # Item 2: version 4 (target version, should process)
        td2 = {
            "input_ids": torch.tensor([[3, 4]]),
            "versions": torch.tensor([[4, 4]]),
            "loss_mask": torch.tensor([[1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0]]),
        }
        output_queue.put(td2)

        # Item 3: version 5 (current version, skip)
        td3 = {
            "input_ids": torch.tensor([[5, 6]]),
            "versions": torch.tensor([[5, 5]]),
            "loss_mask": torch.tensor([[1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0]]),
        }
        output_queue.put(td3)

        # Fire event (current=5, target=4)
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Should only recompute td2
        assert len(mock_engine.recompute_calls) == 1
        assert mock_engine.recompute_calls[0] == [3, 4]

    def test_handler_respects_loss_mask(self, mock_engine, output_queue, result_cache):
        """Test handler only patches tokens where loss_mask=1."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Create item with mixed loss_mask
        td = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "versions": torch.tensor([[4, 4, 4, 4, 4]]),
            "loss_mask": torch.tensor([[0, 1, 0, 1, 1]]),  # Only indices 1, 3, 4
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0]]),
        }

        output_queue.put(td)

        # Fire event
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Should recompute for full sequence
        assert len(mock_engine.recompute_calls) == 1

        # But only patch where loss_mask=1
        assert td["proximal_logprobs_t"][0, 0].item() == -2.0  # Not patched (mask=0)
        assert td["proximal_logprobs_t"][0, 1].item() == -1.0  # Patched (mask=1)
        assert td["proximal_logprobs_t"][0, 2].item() == -2.0  # Not patched (mask=0)
        assert td["proximal_logprobs_t"][0, 3].item() == -1.0  # Patched (mask=1)
        assert td["proximal_logprobs_t"][0, 4].item() == -1.0  # Patched (mask=1)

    def test_handler_handles_mixed_versions_in_sequence(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler with sequence containing mixed versions."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Sequence with mixed versions (some tokens from v4, some from v5)
        td = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "versions": torch.tensor([[3, 4, 4, 5, 5, 5]]),  # Only indices 1,2 are v4
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]]),
        }

        output_queue.put(td)

        # Fire event (current=5, target=4)
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Should recompute
        assert len(mock_engine.recompute_calls) == 1

        # Only patch indices 1, 2 (version=4)
        assert td["proximal_logprobs_t"][0, 0].item() == -2.0  # v3, not patched
        assert td["proximal_logprobs_t"][0, 1].item() == -1.0  # v4, patched
        assert td["proximal_logprobs_t"][0, 2].item() == -1.0  # v4, patched
        assert td["proximal_logprobs_t"][0, 3].item() == -2.0  # v5, not patched
        assert td["proximal_logprobs_t"][0, 4].item() == -2.0  # v5, not patched

    def test_handler_marks_items_as_recomputed(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler marks items with _recompute_version."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        td = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[4, 4, 4]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0, -2.0]]),
        }

        output_queue.put(td)

        # Fire event (current=5)
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Should have marked as recomputed
        assert "_recompute_version" in td
        assert td["_recompute_version"][0, 0].item() == 5

    def test_handler_handles_empty_queue_and_cache(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler handles empty queue and cache gracefully."""
        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Fire event with empty queue/cache
        bus = get_event_bus()
        bus.send(WorkflowEvents.PRE_WEIGHT_UPDATE, sender=mock_engine)

        # Should not crash, no recomputations
        assert len(mock_engine.recompute_calls) == 0

    def test_handler_integration_with_fire_events_decorator(
        self, mock_engine, output_queue, result_cache
    ):
        """Test handler works with decorator-fired events."""
        from areal.infrastructure import fire_events

        handler = ProxTLogprobHandler(output_queue, result_cache)
        handler.register()

        # Add test data
        td = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "versions": torch.tensor([[4, 4, 4]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "proximal_logprobs_t": torch.tensor([[-2.0, -2.0, -2.0]]),
        }
        output_queue.put(td)

        # Create engine with decorated update method
        class DecoratedEngine:
            def __init__(self):
                self._version = 5
                self.recompute_calls = []

            def get_version(self):
                return self._version

            def recompute_output_logprobs_sync(self, input_ids):
                self.recompute_calls.append(input_ids)
                return [-1.0] * len(input_ids)

            @fire_events(
                before=WorkflowEvents.PRE_WEIGHT_UPDATE,
                after=WorkflowEvents.POST_WEIGHT_UPDATE,
            )
            def update_weights(self, meta):
                self._version = meta["version"]

        engine = DecoratedEngine()

        # Call update_weights - should trigger handler automatically
        engine.update_weights({"version": 6})

        # Handler should have processed the data
        assert len(engine.recompute_calls) == 1
        assert td["proximal_logprobs_t"][0, 0].item() == -1.0

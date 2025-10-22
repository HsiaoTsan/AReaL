"""Unit tests for areal.utils.model functions."""

import pytest
from dataclasses import replace

from areal.api.cli_args import (
    ClusterSpecConfig,
    GRPOConfig,
    InferenceEngineConfig,
    PPOActorConfig,
)
from areal.api.io_struct import AllocationMode, WeightUpdateMeta
from areal.utils.model import get_model_update_meta


@pytest.fixture
def base_grpo_config():
    """Create a minimal GRPOConfig for testing."""
    return GRPOConfig(
        experiment_name="test_experiment",
        trial_name="test_trial",
        cluster=ClusterSpecConfig(
            n_nodes=1,
            n_gpus_per_node=8,
            fileroot="/tmp/test_fileroot",
        ),
        allocation_mode="sglang.d4p1t1+d4p1t1",
        rollout=InferenceEngineConfig(),
        actor=PPOActorConfig(
            path="test/model",
        ),
    )


class TestGetModelUpdateMeta:
    """Test suite for get_model_update_meta function."""

    def test_disk_mode(self, base_grpo_config):
        """Test get_model_update_meta with weight_update_mode='disk'."""
        # Set weight_update_mode to 'disk'
        config = replace(
            base_grpo_config,
            actor=replace(base_grpo_config.actor, weight_update_mode="disk"),
        )

        result = get_model_update_meta(config)

        # Verify result is WeightUpdateMeta with type='disk'
        assert isinstance(result, WeightUpdateMeta)
        assert result.type == "disk"
        assert result.path is not None
        assert "test_experiment" in result.path
        assert "test_trial" in result.path
        assert "weight_update" in result.path

    def test_fsdp_xccl_mode(self, base_grpo_config):
        """Test get_model_update_meta with weight_update_mode='fsdp_xccl'."""
        # Set weight_update_mode to 'fsdp_xccl' (or any non-'disk' value)
        config = replace(
            base_grpo_config,
            actor=replace(base_grpo_config.actor, weight_update_mode="fsdp_xccl"),
        )

        result = get_model_update_meta(config)

        # Verify result is WeightUpdateMeta with type from platform
        assert isinstance(result, WeightUpdateMeta)
        # Platform-dependent: nccl/xccl/rccl on GPU, gloo on CPU/Windows
        assert result.type in ["nccl", "xccl", "rccl", "gloo"]
        assert result.alloc_mode is not None
        assert isinstance(result.alloc_mode, AllocationMode)

    def test_allocation_mode_parsing(self, base_grpo_config):
        """Test that allocation_mode is correctly parsed from string."""
        config = replace(
            base_grpo_config,
            allocation_mode="sglang.d2p1t1+d2p2t1",
            actor=replace(base_grpo_config.actor, weight_update_mode="fsdp_xccl"),
        )

        result = get_model_update_meta(config)

        # Verify allocation mode is parsed correctly
        assert result.alloc_mode is not None
        # Check that data_parallel_size matches the allocation mode (d2 = dp=2)
        assert result.alloc_mode.gen.data_parallel_size == 2
        assert result.alloc_mode.train.data_parallel_size == 2

    def test_config_paths_in_disk_mode(self, base_grpo_config):
        """Test that config paths are correctly used in disk mode."""
        custom_fileroot = "/custom/path/fileroot"
        custom_experiment = "my_experiment"
        custom_trial = "my_trial"

        config = replace(
            base_grpo_config,
            experiment_name=custom_experiment,
            trial_name=custom_trial,
            cluster=replace(base_grpo_config.cluster, fileroot=custom_fileroot),
            actor=replace(base_grpo_config.actor, weight_update_mode="disk"),
        )

        result = get_model_update_meta(config)

        # Verify custom paths are used
        assert result.type == "disk"
        assert custom_fileroot in result.path
        assert custom_experiment in result.path
        assert custom_trial in result.path

    def test_default_weight_update_mode(self, base_grpo_config):
        """Test with default weight_update_mode (should use fsdp_xccl)."""
        # Don't explicitly set weight_update_mode, use default
        result = get_model_update_meta(base_grpo_config)

        # Default should be fsdp_xccl (non-disk)
        assert isinstance(result, WeightUpdateMeta)
        # If default is disk, type will be 'disk', otherwise it's xccl-based
        assert result.type in ["disk", "nccl", "xccl", "rccl"]

    def test_type_annotation_compatibility(self, base_grpo_config):
        """Test that the function accepts GRPOConfig type correctly."""
        # This test verifies the type annotation change from untyped to GRPOConfig
        config = base_grpo_config

        # Should not raise TypeError due to type mismatch
        result = get_model_update_meta(config)
        assert isinstance(result, WeightUpdateMeta)

    def test_actor_config_attribute_access(self, base_grpo_config):
        """Test that config.actor.weight_update_mode is accessed correctly."""
        # This tests the change from config.weight_update_mode to config.actor.weight_update_mode
        config = replace(
            base_grpo_config,
            actor=replace(base_grpo_config.actor, weight_update_mode="disk"),
        )

        # Verify the attribute path is correct
        assert hasattr(config, "actor")
        assert hasattr(config.actor, "weight_update_mode")
        assert config.actor.weight_update_mode == "disk"

        result = get_model_update_meta(config)
        assert result.type == "disk"


class TestGetModelUpdateMetaEdgeCases:
    """Edge case tests for get_model_update_meta."""

    def test_with_lora_enabled(self, base_grpo_config):
        """Test get_model_update_meta when LoRA is enabled."""
        config = replace(
            base_grpo_config,
            actor=replace(
                base_grpo_config.actor,
                weight_update_mode="disk",
                use_lora=True,
            ),
        )

        result = get_model_update_meta(config)

        # LoRA flag should be passed through
        # Note: Current implementation doesn't use use_lora, but testing for future compatibility
        assert isinstance(result, WeightUpdateMeta)
        assert result.type == "disk"

    def test_various_allocation_modes(self, base_grpo_config):
        """Test with different allocation mode strings."""
        allocation_modes = [
            "sglang.d1p1t1+d1p1t1",
            "sglang.d2p2t1+d2p2t1",
            "sglang.d4p1t1+d4p1t1",
            "sglang.d8p1t1+d8p1t1",
        ]

        for alloc_mode in allocation_modes:
            config = replace(
                base_grpo_config,
                allocation_mode=alloc_mode,
                actor=replace(base_grpo_config.actor, weight_update_mode="fsdp_xccl"),
            )

            result = get_model_update_meta(config)

            # Should succeed without errors
            assert isinstance(result, WeightUpdateMeta)
            assert result.alloc_mode is not None

    def test_empty_experiment_name_disk_mode(self, base_grpo_config):
        """Test disk mode with minimal experiment name."""
        config = replace(
            base_grpo_config,
            experiment_name="e",
            trial_name="t",
            actor=replace(base_grpo_config.actor, weight_update_mode="disk"),
        )

        result = get_model_update_meta(config)

        # Should still create valid path
        assert result.type == "disk"
        assert result.path is not None
        assert len(result.path) > 0

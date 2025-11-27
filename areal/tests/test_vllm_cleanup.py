"""Unit tests for vLLM server cleanup functionality.

This module tests the signal handling and subprocess cleanup mechanisms
in the vLLM server wrapper to ensure proper resource cleanup on shutdown.
"""

import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

import psutil

from areal.launcher.vllm_server import terminate_process_tree, vLLMServerWrapper


class TestTerminateProcessTree(unittest.TestCase):
    """Test the terminate_process_tree helper function."""

    def test_terminate_single_process(self):
        """Test terminating a single process without children."""
        # Start a simple sleep process
        proc = subprocess.Popen(["sleep", "100"])
        pid = proc.pid

        # Verify process is running
        self.assertTrue(psutil.pid_exists(pid))

        # Terminate it
        terminate_process_tree(pid, timeout=2)

        # Wait and verify it's gone
        proc.wait(timeout=1)
        self.assertFalse(psutil.pid_exists(pid))

    def test_terminate_process_with_children(self):
        """Test terminating a process with child processes."""
        # Create a parent process that spawns children
        script = """
import subprocess
import time
# Spawn multiple child processes
children = []
for i in range(3):
    p = subprocess.Popen(['sleep', '100'])
    children.append(p)
# Keep parent alive
time.sleep(100)
"""
        proc = subprocess.Popen([sys.executable, "-c", script])
        parent_pid = proc.pid

        # Wait for children to spawn
        time.sleep(1)

        # Get child PIDs
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        child_pids = [c.pid for c in children]

        # Verify parent and children exist
        self.assertTrue(psutil.pid_exists(parent_pid))
        self.assertGreater(len(child_pids), 0)
        for child_pid in child_pids:
            self.assertTrue(psutil.pid_exists(child_pid))

        # Terminate the entire tree
        terminate_process_tree(parent_pid, timeout=2)

        # Wait and verify all are gone
        proc.wait(timeout=1)
        self.assertFalse(psutil.pid_exists(parent_pid))
        for child_pid in child_pids:
            self.assertFalse(psutil.pid_exists(child_pid))

    def test_terminate_already_dead_process(self):
        """Test that terminating an already-dead process doesn't raise errors."""
        # Start and immediately kill a process
        proc = subprocess.Popen(["sleep", "1"])
        pid = proc.pid
        proc.kill()
        proc.wait()

        # Verify it's dead
        self.assertFalse(psutil.pid_exists(pid))

        # This should not raise an exception
        terminate_process_tree(pid, timeout=1)

    def test_terminate_with_stubborn_process(self):
        """Test force-killing processes that don't respond to SIGTERM."""
        # Create a process that ignores SIGTERM
        script = """
import signal
import time
# Ignore SIGTERM
signal.signal(signal.SIGTERM, signal.SIG_IGN)
# Run for a long time
time.sleep(100)
"""
        proc = subprocess.Popen([sys.executable, "-c", script])
        pid = proc.pid

        # Verify it's running
        self.assertTrue(psutil.pid_exists(pid))

        # Terminate with short timeout (should force kill)
        terminate_process_tree(pid, timeout=1)

        # Wait and verify it's gone (killed by SIGKILL)
        proc.wait(timeout=1)
        self.assertFalse(psutil.pid_exists(pid))


class TestVLLMServerWrapperSignalHandling(unittest.TestCase):
    """Test signal handling in vLLMServerWrapper."""

    def setUp(self):
        """Set up mock objects for testing."""
        self.mock_config = Mock()
        self.mock_config.seed = 42
        self.mock_allocation_mode = Mock()
        self.mock_allocation_mode.gen_instance_size = 1
        self.mock_allocation_mode.gen.tp_size = 1
        self.mock_allocation_mode.gen.pp_size = 1
        self.mock_allocation_mode.gen.dp_size = 1

    @patch("areal.launcher.vllm_server.signal.signal")
    def test_signal_handlers_registered(self, mock_signal):
        """Test that signal handlers are registered during initialization."""
        _ = vLLMServerWrapper(
            experiment_name="test_exp",
            trial_name="test_trial",
            vllm_config=self.mock_config,
            allocation_mode=self.mock_allocation_mode,
            n_gpus_per_node=8,
        )

        # Verify SIGTERM and SIGINT handlers were registered
        self.assertEqual(mock_signal.call_count, 2)
        call_args_list = [call[0] for call in mock_signal.call_args_list]
        signals_registered = [args[0] for args in call_args_list]

        self.assertIn(signal.SIGTERM, signals_registered)
        self.assertIn(signal.SIGINT, signals_registered)

    def test_cleanup_all_servers_empty(self):
        """Test cleanup with no server processes."""
        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test_exp",
                trial_name="test_trial",
                vllm_config=self.mock_config,
                allocation_mode=self.mock_allocation_mode,
                n_gpus_per_node=8,
            )

            # Should not raise an exception
            wrapper._cleanup_all_servers()
            self.assertEqual(len(wrapper.server_processes), 0)

    @patch("areal.launcher.vllm_server.terminate_process_tree")
    def test_cleanup_all_servers_with_processes(self, mock_terminate):
        """Test cleanup actually terminates server processes."""
        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test_exp",
                trial_name="test_trial",
                vllm_config=self.mock_config,
                allocation_mode=self.mock_allocation_mode,
                n_gpus_per_node=8,
            )

            # Create mock processes
            mock_proc1 = Mock()
            mock_proc1.pid = 1234
            mock_proc1.poll.return_value = None  # Still running

            mock_proc2 = Mock()
            mock_proc2.pid = 5678
            mock_proc2.poll.return_value = None  # Still running

            wrapper.server_processes = [mock_proc1, mock_proc2]

            # Call cleanup
            wrapper._cleanup_all_servers()

            # Verify terminate_process_tree was called for each process
            self.assertEqual(mock_terminate.call_count, 2)
            mock_terminate.assert_any_call(1234, timeout=10)
            mock_terminate.assert_any_call(5678, timeout=10)

            # Verify processes list is cleared
            self.assertEqual(len(wrapper.server_processes), 0)

    @patch("areal.launcher.vllm_server.terminate_process_tree")
    def test_cleanup_skips_already_terminated(self, mock_terminate):
        """Test that cleanup skips processes that already terminated."""
        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test_exp",
                trial_name="test_trial",
                vllm_config=self.mock_config,
                allocation_mode=self.mock_allocation_mode,
                n_gpus_per_node=8,
            )

            # Create mix of running and dead processes
            mock_proc1 = Mock()
            mock_proc1.pid = 1234
            mock_proc1.poll.return_value = 0  # Already dead

            mock_proc2 = Mock()
            mock_proc2.pid = 5678
            mock_proc2.poll.return_value = None  # Still running

            wrapper.server_processes = [mock_proc1, mock_proc2]

            # Call cleanup
            wrapper._cleanup_all_servers()

            # Verify terminate was only called for the running process
            mock_terminate.assert_called_once_with(5678, timeout=10)

    @patch("areal.launcher.vllm_server.sys.exit")
    @patch("areal.launcher.vllm_server.terminate_process_tree")
    def test_handle_shutdown_signal_sigterm(self, mock_terminate, mock_exit):
        """Test handling SIGTERM signal."""
        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test_exp",
                trial_name="test_trial",
                vllm_config=self.mock_config,
                allocation_mode=self.mock_allocation_mode,
                n_gpus_per_node=8,
            )

            # Add a mock process
            mock_proc = Mock()
            mock_proc.pid = 9999
            mock_proc.poll.return_value = None
            wrapper.server_processes = [mock_proc]

            # Simulate receiving SIGTERM
            wrapper._handle_shutdown_signal(signal.SIGTERM, None)

            # Verify cleanup was called and exit was invoked
            mock_terminate.assert_called_once_with(9999, timeout=10)
            mock_exit.assert_called_once_with(0)
            self.assertTrue(wrapper._shutdown_requested)

    @patch("areal.launcher.vllm_server.sys.exit")
    @patch("areal.launcher.vllm_server.terminate_process_tree")
    def test_handle_shutdown_signal_sigint(self, mock_terminate, mock_exit):
        """Test handling SIGINT (Ctrl+C) signal."""
        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test_exp",
                trial_name="test_trial",
                vllm_config=self.mock_config,
                allocation_mode=self.mock_allocation_mode,
                n_gpus_per_node=8,
            )

            # Add a mock process
            mock_proc = Mock()
            mock_proc.pid = 9999
            mock_proc.poll.return_value = None
            wrapper.server_processes = [mock_proc]

            # Simulate receiving SIGINT
            wrapper._handle_shutdown_signal(signal.SIGINT, None)

            # Verify cleanup was called and exit was invoked
            mock_terminate.assert_called_once_with(9999, timeout=10)
            mock_exit.assert_called_once_with(0)
            self.assertTrue(wrapper._shutdown_requested)


class TestVLLMServerWrapperIntegration(unittest.TestCase):
    """Integration tests for the full vLLM server wrapper lifecycle."""

    @patch("areal.launcher.vllm_server.current_platform")
    @patch("areal.launcher.vllm_server.os.getenv")
    @patch("areal.launcher.vllm_server.gethostip")
    @patch("areal.launcher.vllm_server.find_free_ports")
    @patch("areal.launcher.vllm_server.vLLMConfig.build_cmd")
    @patch("areal.launcher.vllm_server.terminate_process_tree")
    def test_exception_in_run_triggers_cleanup(
        self,
        mock_terminate,
        mock_build_cmd,
        mock_find_ports,
        mock_gethostip,
        mock_getenv,
        mock_platform,
    ):
        """Test that exceptions during server startup trigger cleanup of partial processes."""
        # Mock environment setup
        mock_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"
        mock_getenv.return_value = None  # No device control env var set
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.side_effect = [(8000, 8001), (8002, 8003)]
        mock_build_cmd.return_value = ["vllm", "serve"]

        # Create wrapper config
        mock_config = Mock()
        mock_config.seed = 42
        mock_allocation = Mock()
        mock_allocation.gen_instance_size = 1
        mock_allocation.gen.tp_size = 1
        mock_allocation.gen.pp_size = 1
        mock_allocation.gen.dp_size = 1

        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test",
                trial_name="test",
                vllm_config=mock_config,
                allocation_mode=mock_allocation,
                n_gpus_per_node=2,  # Setup for 2 servers
            )

            # Mock the first server launch to succeed
            mock_proc1 = Mock()
            mock_proc1.pid = 1111
            mock_proc1.poll.return_value = None

            # Mock launch_one_server: first call succeeds, second fails
            with patch.object(
                wrapper,
                "launch_one_server",
                side_effect=[mock_proc1, RuntimeError("Launch failed")],
            ):
                # Call run() which should trigger cleanup on exception
                with self.assertRaises(RuntimeError):
                    wrapper.run()

            # Verify cleanup was called for the one process that was created
            mock_terminate.assert_called_once_with(1111, timeout=10)


class TestVLLMServerWrapperSemiIntegration(unittest.TestCase):
    """Semi-integration tests using real processes (CPU-only, no GPU required)."""

    @patch("areal.launcher.vllm_server.current_platform")
    @patch("areal.launcher.vllm_server.os.getenv")
    @patch("areal.launcher.vllm_server.gethostip")
    @patch("areal.launcher.vllm_server.find_free_ports")
    @patch("areal.launcher.vllm_server.vLLMConfig.build_cmd")
    @patch("areal.launcher.vllm_server.name_resolve.add_subentry")
    @patch("areal.launcher.vllm_server.wait_for_server")
    def test_real_process_cleanup_on_signal(
        self,
        mock_wait_for_server,
        mock_add_subentry,
        mock_build_cmd,
        mock_find_ports,
        mock_gethostip,
        mock_getenv,
        mock_platform,
    ):
        """Test cleanup with real subprocesses when SIGTERM is received."""
        # Mock environment setup
        mock_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"
        mock_getenv.return_value = None
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = (8000, 8001)
        mock_build_cmd.return_value = ["sleep", "100"]

        # Create wrapper
        mock_config = Mock()
        mock_config.seed = 42
        mock_allocation = Mock()
        mock_allocation.gen_instance_size = 1
        mock_allocation.gen.tp_size = 1
        mock_allocation.gen.pp_size = 1

        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test",
                trial_name="test",
                vllm_config=mock_config,
                allocation_mode=mock_allocation,
                n_gpus_per_node=1,
            )

            # Launch a real sleep process
            real_proc = subprocess.Popen(["sleep", "100"])
            wrapper.server_processes.append(real_proc)
            real_pid = real_proc.pid

            # Verify process is running
            self.assertTrue(psutil.pid_exists(real_pid))

            # Mock sys.exit to prevent test from exiting
            with patch("areal.launcher.vllm_server.sys.exit"):
                # Simulate receiving SIGTERM
                wrapper._handle_shutdown_signal(signal.SIGTERM, None)

            # Verify process was cleaned up
            self.assertFalse(psutil.pid_exists(real_pid))

    @patch("areal.launcher.vllm_server.current_platform")
    @patch("areal.launcher.vllm_server.os.getenv")
    @patch("areal.launcher.vllm_server.gethostip")
    @patch("areal.launcher.vllm_server.find_free_ports")
    @patch("areal.launcher.vllm_server.vLLMConfig.build_cmd")
    @patch("areal.launcher.vllm_server.name_resolve.add_subentry")
    @patch("areal.launcher.vllm_server.wait_for_server")
    def test_race_condition_multiple_signals(
        self,
        mock_wait_for_server,
        mock_add_subentry,
        mock_build_cmd,
        mock_find_ports,
        mock_gethostip,
        mock_getenv,
        mock_platform,
    ):
        """Test that multiple rapid signals don't cause race conditions."""
        # Mock environment setup
        mock_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"
        mock_getenv.return_value = None
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = (8000, 8001)
        mock_build_cmd.return_value = ["sleep", "100"]

        # Create wrapper
        mock_config = Mock()
        mock_config.seed = 42
        mock_allocation = Mock()
        mock_allocation.gen_instance_size = 1
        mock_allocation.gen.tp_size = 1
        mock_allocation.gen.pp_size = 1

        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test",
                trial_name="test",
                vllm_config=mock_config,
                allocation_mode=mock_allocation,
                n_gpus_per_node=1,
            )

            # Launch real sleep processes
            procs = []
            pids = []
            for _ in range(3):
                proc = subprocess.Popen(["sleep", "100"])
                procs.append(proc)
                pids.append(proc.pid)
                wrapper.server_processes.append(proc)

            # Verify all processes are running
            for pid in pids:
                self.assertTrue(psutil.pid_exists(pid))

            # Mock sys.exit to prevent test from exiting
            with patch("areal.launcher.vllm_server.sys.exit") as mock_exit:
                # Send first signal
                wrapper._handle_shutdown_signal(signal.SIGTERM, None)

                # Try to send a second signal immediately (should be ignored)
                wrapper._handle_shutdown_signal(signal.SIGTERM, None)

                # sys.exit should only be called once
                self.assertEqual(mock_exit.call_count, 1)

            # Verify all processes were cleaned up
            for pid in pids:
                # Wait a bit for cleanup to complete
                for _ in range(10):
                    if not psutil.pid_exists(pid):
                        break
                    time.sleep(0.1)
                self.assertFalse(psutil.pid_exists(pid))

    @patch("areal.launcher.vllm_server.current_platform")
    @patch("areal.launcher.vllm_server.os.getenv")
    @patch("areal.launcher.vllm_server.gethostip")
    @patch("areal.launcher.vllm_server.find_free_ports")
    @patch("areal.launcher.vllm_server.vLLMConfig.build_cmd")
    @patch("areal.launcher.vllm_server.name_resolve.add_subentry")
    @patch("areal.launcher.vllm_server.wait_for_server")
    def test_real_process_tree_cleanup(
        self,
        mock_wait_for_server,
        mock_add_subentry,
        mock_build_cmd,
        mock_find_ports,
        mock_gethostip,
        mock_getenv,
        mock_platform,
    ):
        """Test cleanup of real process trees (parent + children)."""
        # Mock environment setup
        mock_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"
        mock_getenv.return_value = None
        mock_gethostip.return_value = "127.0.0.1"
        mock_find_ports.return_value = (8000, 8001)
        mock_build_cmd.return_value = ["sleep", "100"]

        # Create wrapper
        mock_config = Mock()
        mock_config.seed = 42
        mock_allocation = Mock()
        mock_allocation.gen_instance_size = 1
        mock_allocation.gen.tp_size = 1
        mock_allocation.gen.pp_size = 1

        with patch("areal.launcher.vllm_server.signal.signal"):
            wrapper = vLLMServerWrapper(
                experiment_name="test",
                trial_name="test",
                vllm_config=mock_config,
                allocation_mode=mock_allocation,
                n_gpus_per_node=1,
            )

            # Launch a real process with children
            script = """
import subprocess
import time
children = []
for i in range(2):
    p = subprocess.Popen(['sleep', '100'])
    children.append(p)
time.sleep(100)
"""
            parent_proc = subprocess.Popen([sys.executable, "-c", script])
            wrapper.server_processes.append(parent_proc)
            parent_pid = parent_proc.pid

            # Wait for children to spawn
            time.sleep(0.5)

            # Get child PIDs
            parent_psutil = psutil.Process(parent_pid)
            children = parent_psutil.children(recursive=True)
            child_pids = [c.pid for c in children]

            # Verify parent and children exist
            self.assertTrue(psutil.pid_exists(parent_pid))
            self.assertGreater(len(child_pids), 0)

            # Mock sys.exit to prevent test from exiting
            with patch("areal.launcher.vllm_server.sys.exit"):
                # Trigger cleanup
                wrapper._handle_shutdown_signal(signal.SIGTERM, None)

            # Verify entire tree was cleaned up
            parent_proc.wait(timeout=2)
            self.assertFalse(psutil.pid_exists(parent_pid))
            for child_pid in child_pids:
                self.assertFalse(psutil.pid_exists(child_pid))


if __name__ == "__main__":
    unittest.main()

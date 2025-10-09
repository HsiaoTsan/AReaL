import asyncio
import itertools
import os
import queue
import threading
import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp
import torch
import torch.distributed as dist
import uvloop
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import RolloutStat
from areal.utils.data import concat_padded_tensors
from areal.utils.http import get_default_connector
from realhf.base import logging

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine

logger = logging.getLogger("areal.workflow_api")


RECOMPUTE_VERSION_KEY = "_recompute_version"


def _extract_version(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value)
        tensor = tensor.reshape(-1)
        if tensor.numel() == 0:
            return None
        return int(tensor[0].item())
    except Exception:
        try:
            return int(value)
        except Exception:
            return None


ROLLOUT_POLL_WAIT_TIME = 0.05


def _ensure_recompute_key(td: TensorDict) -> None:
    """Ensure the recompute tracking key exists with batch-aligned shape."""
    if not isinstance(td, TensorDict):
        return
    if RECOMPUTE_VERSION_KEY in td.keys():
        return
    if "versions" not in td.keys():
        return
    versions = td.get("versions")
    default_value = torch.full_like(versions[:, :1], -1, dtype=torch.int64)
    td.set(RECOMPUTE_VERSION_KEY, default_value)


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> TensorDict | None:
        """Run a single episode of the workflow.

        `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.
        """
        raise NotImplementedError()


class WorkflowExecutor:

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: "InferenceEngine",
    ):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.inference_engine = inference_engine

        qsize = config.queue_size or config.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[TensorDict] = []

        self.rollout_stat = RolloutStat()

        self.session = None
        # Count of dropped over-stale samples at output filtering
        self.drop_count = 0
        # Track last version purged from output_queue
        self._last_purged_ver = -1
        # Track recent effective samples returned to the trainer
        self._effective_history = deque(maxlen=16)
        self._effective_sum = 0.0

    def initialize(self):
        self.rollout_tasks: Dict[str, asyncio.Task] = {}
        self.rollout_thread = threading.Thread(
            target=self._rollout_thread, daemon=True
        )  # set daemon=True to automatically exit when error occurs
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()

    def get_capacity(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        with self.lock:
            max_concurrent_rollouts = max(
                1, self.config.max_concurrent_rollouts // world_size
            )
            capacity = max_concurrent_rollouts - len(self.rollout_tasks)
            # Staleness control
            version = self.inference_engine.get_version()
            ofp = self.config.max_head_offpolicyness
            if self._effective_history:
                effective_recent = self._effective_sum / len(self._effective_history)
            else:
                effective_recent = float(self.rollout_stat.accepted)
            sample_cnt = effective_recent + self.rollout_stat.running
            consumer_bs = max(1, self.config.consumer_batch_size // world_size)
            budget = (ofp + version + 1) * consumer_bs - sample_cnt
            budget_int = int(budget)
            capacity = min(capacity, max(0, budget_int))
        return capacity

    def _record_effective_samples(self, count: int) -> None:
        if count <= 0:
            return
        with self.lock:
            if len(self._effective_history) == self._effective_history.maxlen:
                self._effective_sum -= self._effective_history.popleft()
            self._effective_history.append(count)
            self._effective_sum += count

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        if self.session is None:
            # NOTE: Lazily initialize aiohttp.ClientSession since it needs to be initialized
            # inside asyncio loop in WorkflowExecutor
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.config.request_timeout,
                    sock_connect=self.config.request_timeout,
                    connect=self.config.request_timeout,
                ),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            )

        rollout_tasks = self.rollout_tasks
        rid = 0
        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create new rollout task
                self.lock.acquire()
                while (
                    capacity > 0
                    and not self.paused.is_set()
                    and self.input_queue.qsize() > 0
                ):
                    data, workflow = self.input_queue.get_nowait()
                    logger.debug(f"Get data from puller: {data}")
                    task = asyncio.create_task(
                        workflow.arun_episode(self.inference_engine, data),
                        name=str(rid),
                    )
                    rollout_tasks[str(rid)] = task
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )
                    capacity -= 1
                    rid += 1
                tasks = list(rollout_tasks.values())
                self.lock.release()

                # Wait for rollout completion
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                # Collect done results
                for task in done:
                    traj = await task
                    traj: TensorDict
                    task_rid = task.get_name()
                    with self.lock:
                        rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                            )
                    # Plan A: conditionally filter over-stale samples before enqueueing output
                    try:
                        drop = False
                        # high-water threshold and per-iteration drop budget
                        MAX_DROP_PER_LOOP = 64
                        qsize = self.output_queue.qsize()
                        qmax = self.output_queue.maxsize or 1
                        high_water = qsize >= int(0.8 * qmax)
                        # cache low-water: keep training flowing
                        cache_low_water = len(self.result_cache) < max(1, self.config.consumer_batch_size // 2)
                        # local drop counter
                        local_dropped = 0
                        try:
                            current_ver = self.inference_engine.get_version()
                            versions = traj.get("versions", None)
                            loss_mask = traj.get("loss_mask", None)
                            if versions is not None:
                                ver = versions[0].tolist()
                                valid_versions = [v for v in ver if v >= 0]
                                if valid_versions:
                                    max_version = max(valid_versions)
                                    if high_water and (not cache_low_water) and (current_ver - max_version) >= 2:
                                        drop = True
                        except Exception:
                            traceback.print_exc()
                        if drop:
                            with self.lock:
                                self.rollout_stat.accepted -= 1
                                self.drop_count += 1
                                dc = self.drop_count
                            logger.warning(
                                f"[OutputFilter] drop over-stale sample before enqueue; total_dropped={dc}"
                            )
                            continue
                        self.output_queue.put_nowait(traj)
                    except queue.Full:
                        raise RuntimeError(
                            "Output queue full. Please increase queue_size."
                        )

                await asyncio.sleep(1)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            with self.lock:
                for task in rollout_tasks.values():
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            await self.session.close()

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> None:
        try:
            if workflow is None:
                workflow = workflow_builder()
            self.input_queue.put_nowait((data, workflow))
        except queue.Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        tik = time.perf_counter()
        accepted = len(self.result_cache)
        timeout = timeout or float(7 * 24 * 3600)

        # Version-switch purge: only when version increases, drain queue once and classify
        try:
            current_ver = self.inference_engine.get_version()
            if current_ver > self._last_purged_ver:
                drained = 0
                dropped = 0
                kept = 0
                put_back_buf: List[TensorDict] = []

                while True:
                    try:
                        traj = self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained += 1
                    try:
                        versions = traj.get("versions", None)
                        loss_mask = traj.get("loss_mask", None)
                        if versions is None or loss_mask is None:
                            put_back_buf.append(traj)
                            continue

                        ver = versions[0].tolist()
                        lm_row = loss_mask[0]
                        lm = lm_row.tolist() if torch.is_tensor(lm_row) else list(lm_row)

                        output_positions = [idx for idx, m in enumerate(lm) if m]
                        output_versions = [ver[idx] for idx in output_positions if ver[idx] >= 0]
                        if not output_versions:
                            put_back_buf.append(traj)
                            continue

                        max_version = max(output_versions)
                        min_version = min(output_versions)

                        patched_ver = _extract_version(traj.get(RECOMPUTE_VERSION_KEY, None))
                        recomputed = patched_ver is not None and patched_ver >= 0

                        tail_staleness = current_ver - max_version
                        head_staleness = current_ver - min_version if recomputed else None
                        staleness = head_staleness if recomputed else tail_staleness

                        allow_staleness = 1
                        if recomputed and self.config.max_head_offpolicyness is not None:
                            allow_staleness = max(allow_staleness, int(self.config.max_head_offpolicyness))

                        if staleness > allow_staleness:
                            dropped += 1
                        else:
                            put_back_buf.append(traj)
                            kept += 1
                    except Exception:
                        put_back_buf.append(traj)
                        traceback.print_exc()

                for item in put_back_buf:
                    try:
                        _ensure_recompute_key(item)
                        self.output_queue.put_nowait(item)
                    except queue.Full:
                        _ensure_recompute_key(item)
                        self.result_cache.append(item)

                logger.info(
                    f"[QueuePurge] ver_switch to {current_ver}: "
                    f"drained={drained} dropped={dropped} kept={kept} "
                    f"cache_size={len(self.result_cache)}"
                )
                self._last_purged_ver = current_ver
        except Exception:
            traceback.print_exc()


        accepted = len(self.result_cache)

        while (
            accepted < count
            and not self.exiting.is_set()
            and time.perf_counter() - tik < timeout
        ):
            try:
                result = self.output_queue.get(timeout=ROLLOUT_POLL_WAIT_TIME)
                if result is not None and (
                    should_accept is None or should_accept(result)
                ):
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Accept rollout result. accepted/count = {accepted}/{count}"
                        )
                    result = result.clone()
                    _ensure_recompute_key(result)
                    self.result_cache.append(result)
                    accepted += 1
                else:
                    if self.config.enable_rollout_tracing:
                        logger.info(f"Rollout is rejected.")
                    with self.lock:
                        self.rollout_stat.accepted -= 1
            except queue.Empty:
                pass
        if self.exiting.is_set():
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, " f"only received {accepted}."
            )
        if self.config.enable_rollout_tracing:
            logger.info(
                f"Rollout results are ready! accepted/count = {accepted}/{count}"
            )
        # Non-blocking recompute across the whole cache: for any cached sample,
        # if there exist output tokens with version == current_version - 1,
        # recompute once (prefill on full output span) and patch exactly those
        # positions in proximal_logprobs_t.
        try:
            current_ver = self.inference_engine.get_version()
            total_candidates = 0
            total_patched = 0
            if hasattr(self.inference_engine, "recompute_output_logprobs_sync"):
                for idx, td in enumerate(self.result_cache):
                    try:
                        input_ids = td.get("input_ids", None)
                        versions = td.get("versions", None)
                        loss_mask = td.get("loss_mask", None)
                        prox = td.get("proximal_logprobs_t", None)
                        if (
                            input_ids is None
                            or versions is None
                            or loss_mask is None
                            or prox is None
                        ):
                            continue
                        ids = input_ids[0].tolist()
                        ver = versions[0].tolist()
                        lm = loss_mask[0].tolist()
                        attn_mask = td.get("attention_mask", None)
                        valid_len = len(ids)
                        try:
                            if attn_mask is not None:
                                mask_row = attn_mask[0]
                                if torch.is_tensor(mask_row):
                                    valid_len = min(valid_len, int(mask_row.sum().item()))
                                else:
                                    valid_len = min(valid_len, int(sum(mask_row)))
                        except Exception:
                            traceback.print_exc()
                        valid_len = min(valid_len, len(ver), len(lm))
                        if valid_len <= 0:
                            continue
                        if patched_ver is not None and patched_ver >= current_ver:
                            # 这条样本在 current_ver 已经重算/确认过了，直接跳过，避免重复重算
                            continue
                        lm_valid = lm[:valid_len]
                        output_positions = [idx for idx, mask in enumerate(lm_valid) if mask]
                        out_len = len(output_positions)
                        if out_len == 0:
                            continue
                        first_output_idx = output_positions[0]
                        start_index = max(0, first_output_idx - 1)
                        if first_output_idx > 0:
                            assert start_index == first_output_idx - 1, (
                                f'start_index({start_index}) != first_output_idx-1({first_output_idx - 1})'
                            )
                        else:
                            assert start_index == 0, 'start_index should be 0 when first_output_idx is 0'
                        need_positions = [
                            (pos_idx, seq_idx)
                            for pos_idx, seq_idx in enumerate(output_positions)
                            if ver[seq_idx] == current_ver - 1
                        ]
                        v_last = ver[output_positions[-1]]
                        if need_positions:
                            total_candidates += 1
                            try:
                                seg = [ver[pos] for pos in output_positions]
                                hist = {}
                                for v in seg:
                                    hist[v] = hist.get(v, 0) + 1
                                hist_items = sorted(hist.items())
                                logger.info(
                                    f"[Recompute] sample#{idx}: version_hist={dict(hist_items)}"
                                )
                                if v_last == 0:
                                    try:
                                        idx_last = output_positions[-1]
                                        val_last = ver[idx_last]
                                        logger.warning(
                                            f"[Recompute] sample#{idx}: v_last==0 sanity idx={idx_last} versions[idx]={val_last}"
                                        )
                                    except Exception:
                                        traceback.print_exc()
                            except Exception:
                                traceback.print_exc()
                        if not need_positions:
                            continue
                        latest_out_logp = self.inference_engine.recompute_output_logprobs_sync(
                            input_ids=ids,
                            start_index=start_index,
                        )
                        patched_here = 0
                        debug_diff = os.environ.get("AREAL_DEBUG_RECOMPUTE_DIFF")
                        diff_entries = [] if debug_diff else None
                        logprobs_field = td.get("logprobs", None) if debug_diff else None
                        max_required_offset = output_positions[-1] - start_index - 1
                        assert max_required_offset < len(latest_out_logp), (
                            f'recompute too short: need {max_required_offset}, got {len(latest_out_logp) - 1}'
                        )
                        for pos_idx, seq_idx in need_positions:
                            assert lm[seq_idx], f'patching a non-output position seq_idx={seq_idx}'
                            rel_offset = seq_idx - start_index - 1
                            if rel_offset < 0 or rel_offset >= len(latest_out_logp):
                                logger.warning(
                                    f"[Recompute] sample#{idx}: rel_offset={rel_offset} out_of_range for logprobs len={len(latest_out_logp)}"
                                )
                                continue
                            new_val = float(latest_out_logp[rel_offset])
                            old_val = float(prox[0, seq_idx].item())
                            prox[0, seq_idx] = new_val
                            patched_here += 1
                            if diff_entries is not None:
                                infer_val = None
                                try:
                                    if logprobs_field is not None:
                                        if torch.is_tensor(logprobs_field):
                                            infer_val = float(logprobs_field[0, seq_idx].item())
                                        else:
                                            infer_val = float(logprobs_field[0][seq_idx])
                                except Exception:
                                    infer_val = None
                                token_id = None
                                try:
                                    if seq_idx < len(ids):
                                        token_id = int(ids[seq_idx])
                                except Exception:
                                    token_id = None
                                diff_entries.append(
                                    {
                                        "seq_idx": int(seq_idx),
                                        "token_id": token_id,
                                        "old_prox": old_val,
                                        "new_prox": new_val,
                                        "infer_logp": infer_val,
                                        "delta_prox": new_val - old_val,
                                        "delta_infer": (None if infer_val is None else new_val - infer_val),
                                    }
                                )
                        if patched_here == 0:
                            continue
                        patched_value = torch.full_like(
                            versions[:, :1], int(current_ver), dtype=torch.int64
                        )
                        td.set(RECOMPUTE_VERSION_KEY, patched_value)
                        if diff_entries:
                            logger.info(
                                f"[RecomputeDiff] sample#{idx} entries={diff_entries}"
                            )
                        total_patched += patched_here
                        logger.info(
                            "[Recompute] sample=%d patched=%d start=%d old_tail=%s new_tail=%s token_ids=%s",
                            idx,
                            patched_here,
                            start_index,
                            [float(prox[0, pos]) for _, pos in need_positions[:5]],
                            latest_out_logp[:5],
                            ids[start_index + 1 : start_index + 6],
                        )
                    except Exception:
                        traceback.print_exc()
                        continue
        except Exception:
            traceback.print_exc()

        

        # After patching the cache, drop too-old samples in cache (v_last <= current-2), then return results
        try:
            current_ver = self.inference_engine.get_version()
            filtered: List[TensorDict] = []
            dropped_cache = 0
            for td in self.result_cache:
                try:
                    versions = td.get("versions", None)
                    loss_mask = td.get("loss_mask", None)
                    if versions is None or loss_mask is None:
                        filtered.append(td)
                        continue
                    ver = versions[0].tolist()
                    lm_row = loss_mask[0]
                    if torch.is_tensor(lm_row):
                        lm = lm_row.tolist()
                    else:
                        lm = list(lm_row)
                    output_positions = [idx for idx, mask in enumerate(lm) if mask]
                    output_versions = [ver[idx] for idx in output_positions if ver[idx] >= 0]
                    if not output_versions:
                        filtered.append(td)
                        continue
                    max_version = max(output_versions)
                    min_version = min(output_versions)
                    patched_ver = _extract_version(td.get(RECOMPUTE_VERSION_KEY, None))
                    recomputed = patched_ver is not None and patched_ver >= 0
                    tail_staleness = current_ver - max_version
                    head_staleness = current_ver - min_version if recomputed else None
                    staleness = head_staleness if recomputed else tail_staleness
                    allow_staleness = 1
                    if recomputed and self.config.max_head_offpolicyness is not None:
                        allow_staleness = max(
                            allow_staleness, int(self.config.max_head_offpolicyness)
                        )
                    if staleness > allow_staleness:
                        dropped_cache += 1
                    else:
                        filtered.append(td)
                except Exception:
                    filtered.append(td)
                    traceback.print_exc()
            self.result_cache = filtered
            cache_debug_entries: List[str] = []
            for idx, td in enumerate(self.result_cache):
                try:
                    versions = td.get("versions", None)
                    loss_mask = td.get("loss_mask", None)
                    prox = td.get("proximal_logprobs_t", None)
                    if versions is None or loss_mask is None:
                        cache_debug_entries.append(f"idx={idx}:missing_fields")
                        continue
                    ver = versions[0].tolist()
                    lm_row = loss_mask[0]
                    if torch.is_tensor(lm_row):
                        lm = lm_row.tolist()
                    else:
                        lm = list(lm_row)
                    output_positions = [idx_pos for idx_pos, mask in enumerate(lm) if mask]
                    output_versions = [ver[idx_pos] for idx_pos in output_positions if ver[idx_pos] >= 0]
                    if not output_versions:
                        cache_debug_entries.append(f"idx={idx}:no_valid_outputs")
                        continue
                    patched_ver = _extract_version(
                        td.get(RECOMPUTE_VERSION_KEY, None)
                    )
                    recomputed = patched_ver is not None and patched_ver >= 0
                    last_output_version = None
                    for pos in reversed(output_positions):
                        val = ver[pos]
                        if val >= 0:
                            last_output_version = val
                            break
                    max_version = max(output_versions)
                    min_version = min(output_versions)
                    tail_staleness = current_ver - max_version
                    head_staleness = current_ver - min_version if recomputed else None
                    staleness = head_staleness if recomputed else tail_staleness
                    allow_staleness = 1
                    if recomputed and self.config.max_head_offpolicyness is not None:
                        allow_staleness = max(
                            allow_staleness, int(self.config.max_head_offpolicyness)
                        )
                    hist: Dict[int, int] = {}
                    for v in output_versions:
                        hist[v] = hist.get(v, 0) + 1
                    hist_items = ", ".join(f"{k}:{hist[k]}" for k in sorted(hist))
                    prox_shape = tuple(prox.shape) if prox is not None else None
                    cache_debug_entries.append(
                        f"idx={idx}:last_out={last_output_version} first_out={min_version} max_out={max_version} patched={patched_ver} recomputed={recomputed} head_staleness={head_staleness} tail_staleness={tail_staleness} allow={allow_staleness} used_staleness={staleness} prox_shape={prox_shape} hist={{ {hist_items} }}"
                    )
                except Exception:
                    cache_debug_entries.append(f"idx={idx}:error")
                    traceback.print_exc()
            if cache_debug_entries:
                logger.info(
                    f"[CacheState] size={len(self.result_cache)} "
                    + "; ".join(cache_debug_entries)
                )
            if dropped_cache:
                logger.info(f"[CacheFilter] dropped_cache={dropped_cache} size={len(self.result_cache)}")
        except Exception:
            traceback.print_exc()

        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        for td in results:
            _ensure_recompute_key(td)
        self._record_effective_samples(len(results))
        return concat_padded_tensors(results)

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow, workflow_builder)
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ):
        if not hasattr(self, "data_generator"):
            self.data_generator = itertools.cycle(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.input_queue.qsize() + dataloader.batch_size
                < self.input_queue.maxsize
            ):
                data = next(self.data_generator)
                for item in data:
                    self.submit(
                        item,
                        workflow=workflow,
                        workflow_builder=workflow_builder,
                    )
            try:
                return self.wait(
                    dataloader.batch_size, timeout=1, should_accept=should_accept
                )
            except TimeoutError:
                pass

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

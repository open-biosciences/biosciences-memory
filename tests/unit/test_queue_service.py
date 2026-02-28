"""Unit tests for the queue service."""

import asyncio

import pytest

from biosciences_memory.services.queue_service import QueueService


@pytest.mark.unit
class TestQueueService:
    async def test_initial_queue_size(self):
        qs = QueueService()
        assert qs.get_queue_size("test") == 0

    async def test_worker_not_running_initially(self):
        qs = QueueService()
        assert qs.is_worker_running("test") is False

    async def test_add_episode_task_starts_worker(self):
        qs = QueueService()
        processed = []

        async def process():
            processed.append(True)

        await qs.add_episode_task("group1", process)
        await asyncio.sleep(0.1)  # Let worker process
        assert len(processed) == 1

    async def test_sequential_processing_within_group(self):
        qs = QueueService()
        order = []

        async def make_task(n):
            async def process():
                order.append(n)

            return process

        await qs.add_episode_task("g1", await make_task(1))
        await qs.add_episode_task("g1", await make_task(2))
        await qs.add_episode_task("g1", await make_task(3))
        await asyncio.sleep(0.2)
        assert order == [1, 2, 3]

    async def test_uninitialized_add_episode_raises(self):
        qs = QueueService()
        with pytest.raises(RuntimeError, match="not initialized"):
            await qs.add_episode(
                group_id="test",
                name="test",
                content="test",
                source_description="test",
                episode_type=None,
                entity_types=None,
                uuid=None,
            )

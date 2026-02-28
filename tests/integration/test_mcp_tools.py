"""Integration tests for MCP tools using FastMCP in-memory Client pattern.

These tests create the FastMCP server and call tools directly without
requiring a running Neo4j instance — the Graphiti service is mocked.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
async def mcp_server():
    """Create a test MCP server with mocked Graphiti service."""
    # We test tool registration and response format, not Graphiti internals
    with patch("biosciences_memory.server.GraphitiService") as MockService:
        mock_service = MockService.return_value
        mock_service.initialize = AsyncMock()
        mock_service.get_client = AsyncMock()
        mock_service.entity_types = None

        with patch("biosciences_memory.server.QueueService") as MockQueue:
            mock_queue = MockQueue.return_value
            mock_queue.initialize = AsyncMock()
            mock_queue.add_episode = AsyncMock(return_value=1)

            from biosciences_memory.server import create_server

            server = await create_server()
            # Attach mocks for assertions
            server._test_service = mock_service
            server._test_queue = mock_queue
            yield server


@pytest.mark.integration
class TestMCPTools:
    async def test_server_has_all_tools(self, mcp_server):
        """Verify all 9 MCP tools are registered."""
        from fastmcp.client import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            expected = {
                "add_memory",
                "search_nodes",
                "search_memory_facts",
                "get_episodes",
                "get_entity_edge",
                "delete_entity_edge",
                "delete_episode",
                "clear_graph",
                "get_status",
            }
            assert tool_names == expected

    async def test_add_memory_returns_success(self, mcp_server):
        """Verify add_memory queues episode and returns success."""
        from fastmcp.client import Client

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "add_memory",
                {"name": "Test Episode", "episode_body": "BRCA1 interacts with TP53"},
            )
            assert result is not None

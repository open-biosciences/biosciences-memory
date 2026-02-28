"""Unit tests for configuration schema."""

import pytest

from biosciences_memory.config.schema import (
    GraphitiConfig,
    Neo4jProviderConfig,
    ServerConfig,
)


@pytest.mark.unit
class TestConfig:
    def test_default_config(self, monkeypatch):
        monkeypatch.setenv("CONFIG_PATH", "/dev/null/nonexistent.yaml")
        config = GraphitiConfig()
        assert config.server.transport == "http"
        assert config.server.port == 8000
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4.1"
        assert config.database.provider == "neo4j"

    def test_server_defaults(self):
        server = ServerConfig()
        assert server.host == "0.0.0.0"
        assert server.port == 8000

    def test_neo4j_pool_defaults(self):
        neo4j = Neo4jProviderConfig()
        assert neo4j.max_connection_lifetime == 300
        assert neo4j.max_connection_pool_size == 50
        assert neo4j.connection_acquisition_timeout == 60.0

    def test_graphiti_group_id_default(self):
        config = GraphitiConfig()
        assert config.graphiti.group_id == "main"
        assert config.graphiti.user_id == "mcp_user"

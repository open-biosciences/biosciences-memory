"""Unit tests for factory classes."""

import pytest

from biosciences_memory.config.schema import DatabaseConfig, Neo4jProviderConfig
from biosciences_memory.services.factories import DatabaseDriverFactory


@pytest.mark.unit
class TestDatabaseDriverFactory:
    def test_neo4j_config_defaults(self, monkeypatch):
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        config = DatabaseConfig()
        result = DatabaseDriverFactory.create_config(config)
        assert result["uri"] == "bolt://localhost:7687"
        assert result["user"] == "neo4j"
        assert result["database"] == "neo4j"
        assert result["max_connection_lifetime"] == 300

    def test_neo4j_config_custom(self, monkeypatch):
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        neo4j = Neo4jProviderConfig(
            uri="neo4j+s://custom:7687",
            username="admin",
            password="secret",
            max_connection_pool_size=100,
        )
        config = DatabaseConfig(neo4j=neo4j)
        result = DatabaseDriverFactory.create_config(config)
        assert result["uri"] == "neo4j+s://custom:7687"
        assert result["user"] == "admin"
        assert result["max_connection_pool_size"] == 100

"""Factory classes for creating LLM, Embedder, and Database clients."""

import logging
import os

from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

from biosciences_memory.config.schema import DatabaseConfig, EmbedderConfig, LLMConfig

logger = logging.getLogger(__name__)


def _validate_api_key(provider_name: str, api_key: str | None) -> str:
    if not api_key:
        raise ValueError(
            f"{provider_name} API key is not configured. "
            "Please set the appropriate environment variable."
        )
    logger.info(f"Creating {provider_name} client")
    return api_key


class LLMClientFactory:
    """Factory for creating OpenAI LLM clients."""

    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        if not config.openai:
            raise ValueError("OpenAI provider configuration not found")

        api_key = _validate_api_key("OpenAI", config.openai.api_key)

        is_reasoning_model = (
            config.model.startswith("gpt-5")
            or config.model.startswith("o1")
            or config.model.startswith("o3")
        )
        small_model = "gpt-5-nano" if is_reasoning_model else "gpt-4.1-mini"

        llm_config = CoreLLMConfig(
            api_key=api_key,
            model=config.model,
            small_model=small_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        if is_reasoning_model:
            return OpenAIClient(config=llm_config, reasoning="minimal", verbosity="low")
        return OpenAIClient(config=llm_config, reasoning=None, verbosity=None)


class EmbedderFactory:
    """Factory for creating OpenAI Embedder clients."""

    @staticmethod
    def create(config: EmbedderConfig) -> EmbedderClient:
        if not config.openai:
            raise ValueError("OpenAI provider configuration not found")

        api_key = _validate_api_key("OpenAI Embedder", config.openai.api_key)

        from graphiti_core.embedder.openai import OpenAIEmbedderConfig

        embedder_config = OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model=config.model,
            embedding_dim=config.dimensions,
        )
        return OpenAIEmbedder(config=embedder_config)


class DatabaseDriverFactory:
    """Factory for creating Neo4j database configuration."""

    @staticmethod
    def create_config(config: DatabaseConfig) -> dict:
        neo4j_config = config.neo4j
        if not neo4j_config:
            from biosciences_memory.config.schema import Neo4jProviderConfig

            neo4j_config = Neo4jProviderConfig()

        uri = os.environ.get("NEO4J_URI", neo4j_config.uri)
        username = os.environ.get("NEO4J_USER", neo4j_config.username)
        password = os.environ.get("NEO4J_PASSWORD", neo4j_config.password)

        return {
            "uri": uri,
            "user": username,
            "password": password,
            "database": neo4j_config.database,
            "max_connection_lifetime": neo4j_config.max_connection_lifetime,
            "max_connection_pool_size": neo4j_config.max_connection_pool_size,
            "connection_acquisition_timeout": neo4j_config.connection_acquisition_timeout,
        }

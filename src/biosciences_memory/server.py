"""
Biosciences Memory MCP Server — Exposes Graphiti functionality through MCP.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel
from starlette.responses import JSONResponse

from biosciences_memory.config.schema import GraphitiConfig
from biosciences_memory.models.entity_types import ENTITY_TYPES
from biosciences_memory.models.response_types import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
    StatusResponse,
    SuccessResponse,
)
from biosciences_memory.services.factories import (
    DatabaseDriverFactory,
    EmbedderFactory,
    LLMClientFactory,
)
from biosciences_memory.services.queue_service import QueueService
from biosciences_memory.utils.formatting import format_fact_result

# Load .env file
mcp_server_dir = Path(__file__).parent.parent.parent
env_file = mcp_server_dir / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# Semaphore limit configuration
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)

# Configure specific loggers
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logging.getLogger('mcp.server.streamable_http_manager').setLevel(logging.WARNING)

# Enable OpenAI trace logging (configurable via environment)
openai_log_level = os.getenv('OPENAI_LOG_LEVEL', 'DEBUG')
httpx_log_level = os.getenv('HTTPX_LOG_LEVEL', 'INFO')
graphiti_log_level = os.getenv('GRAPHITI_LOG_LEVEL', 'DEBUG')

logging.getLogger('openai').setLevel(getattr(logging, openai_log_level))
logging.getLogger('httpx').setLevel(getattr(logging, httpx_log_level))
logging.getLogger('graphiti_core').setLevel(getattr(logging, graphiti_log_level))

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Custom Neo4j Driver with Connection Pool Configuration
# ------------------------------------------------------------------------------


class Neo4jDriverWithPoolConfig(Neo4jDriver):
    """Neo4j driver with configurable connection pool settings.

    This is a workaround for graphiti-core not exposing connection pool parameters.
    Required for Neo4j Aura and other cloud providers that terminate idle connections.

    See: https://github.com/donbr/graphiti-fastmcp/issues/17
    """

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'neo4j',
        max_connection_lifetime: int = 300,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: float = 60.0,
    ):
        # Skip parent __init__ to avoid creating driver with default settings
        # Call grandparent __init__ instead
        from graphiti_core.driver.driver import GraphDriver

        GraphDriver.__init__(self)

        # Create driver with pool configuration
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=connection_acquisition_timeout,
        )
        self._database = database
        self.aoss_client = None

        logger.info(
            f'Created Neo4j driver with pool config: '
            f'max_lifetime={max_connection_lifetime}s, '
            f'pool_size={max_connection_pool_size}, '
            f'acquisition_timeout={connection_acquisition_timeout}s'
        )


# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to
capture relationships between concepts, entities, and information. The system organizes data as episodes
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic,
queryable memory store that evolves with new information.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations.
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality.
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid
API keys are provided for any language model operations.
"""

# ------------------------------------------------------------------------------
# Service Definition
# ------------------------------------------------------------------------------


class GraphitiService:
    """Graphiti service using the unified configuration system."""

    def __init__(self, config: GraphitiConfig, semaphore_limit: int = 10):
        self.config = config
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.client: Graphiti | None = None
        self.entity_types = None

    async def initialize(self) -> None:
        """Initialize the Graphiti client with factory-created components."""
        try:
            llm_client = None
            embedder_client = None

            try:
                llm_client = LLMClientFactory.create(self.config.llm)
            except Exception as e:
                logger.warning(f'Failed to create LLM client: {e}')

            try:
                embedder_client = EmbedderFactory.create(self.config.embedder)
            except Exception as e:
                logger.warning(f'Failed to create embedder client: {e}')

            db_config = DatabaseDriverFactory.create_config(self.config.database)

            # Build custom entity types from config, or use biosciences defaults
            custom_types = None
            if self.config.graphiti.entity_types:
                custom_types = {}
                for entity_type in self.config.graphiti.entity_types:
                    entity_model = type(
                        entity_type.name,
                        (BaseModel,),
                        {'__doc__': entity_type.description},
                    )
                    custom_types[entity_type.name] = entity_model
            else:
                # Use biosciences domain entity types as default
                custom_types = ENTITY_TYPES
            self.entity_types = custom_types

            # Use custom Neo4j driver with connection pool configuration
            # This prevents "defunct connection" errors with Neo4j Aura
            neo4j_driver = Neo4jDriverWithPoolConfig(
                uri=db_config['uri'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config.get('database', 'neo4j'),
                max_connection_lifetime=db_config.get('max_connection_lifetime', 300),
                max_connection_pool_size=db_config.get('max_connection_pool_size', 50),
                connection_acquisition_timeout=db_config.get(
                    'connection_acquisition_timeout', 60.0
                ),
            )
            self.client = Graphiti(
                graph_driver=neo4j_driver,
                llm_client=llm_client,
                embedder=embedder_client,
                max_coroutines=self.semaphore_limit,
            )

            # Build indices - wrap in try/except to handle Neo4j 5.x race condition
            # with parallel IF NOT EXISTS index creation
            try:
                await self.client.build_indices_and_constraints()
            except Exception as idx_error:
                if 'EquivalentSchemaRuleAlreadyExists' in str(idx_error):
                    logger.warning(
                        'Index creation race condition detected (Neo4j 5.x issue). '
                        'Indexes likely already exist. Continuing...'
                    )
                else:
                    raise

            logger.info('Successfully initialized Graphiti client')

        except Exception as e:
            logger.error(f'Failed to initialize Graphiti client: {e}')
            raise

    async def get_client(self) -> Graphiti:
        if self.client is None:
            await self.initialize()
        if self.client is None:
            raise RuntimeError('Failed to initialize Graphiti client')
        return self.client


# ------------------------------------------------------------------------------
# Factory Entrypoint
# ------------------------------------------------------------------------------


async def create_server() -> FastMCP:
    """Factory function that creates and initializes the MCP server.

    This is the entrypoint for FastMCP Cloud and `fastmcp dev`.
    Configuration comes from environment variables and config files only.
    """
    logger.info('Initializing Biosciences Memory MCP server via factory pattern...')

    # 1. Load configuration
    factory_config = GraphitiConfig()

    # 2. Initialize Services
    factory_graphiti_service = GraphitiService(factory_config, SEMAPHORE_LIMIT)
    factory_queue_service = QueueService()

    # Await initialization of the Graphiti service
    await factory_graphiti_service.initialize()

    # Initialize queue with the Graphiti client
    client = await factory_graphiti_service.get_client()
    await factory_queue_service.initialize(client)

    logger.info('Graphiti services initialized successfully via factory')

    # 3. Create Server Instance
    server = FastMCP(
        'Biosciences Memory Server',
        instructions=GRAPHITI_MCP_INSTRUCTIONS,
    )

    # 4. Register Tools
    _register_tools(server, factory_config, factory_graphiti_service, factory_queue_service)

    # 5. Register Custom Routes
    @server.custom_route('/health', methods=['GET'])
    async def health_check(request):
        return JSONResponse({'status': 'healthy', 'service': 'biosciences-memory'})

    @server.custom_route('/status', methods=['GET'])
    async def status_check(request):
        return JSONResponse({'status': 'ok', 'service': 'biosciences-memory'})

    logger.info('FastMCP server created with factory pattern')
    return server


def _register_tools(
    server: FastMCP,
    cfg: GraphitiConfig,
    graphiti_svc: GraphitiService,
    queue_svc: QueueService,
) -> None:
    """Register all MCP tools using closure to capture service instances."""

    @server.tool()
    async def add_memory(
        name: str,
        episode_body: str,
        group_id: str | None = None,
        source: str = 'text',
        source_description: str = '',
        uuid: str | None = None,
    ) -> SuccessResponse | ErrorResponse:
        """Add an episode to memory. This is the primary way to add information to the graph.

        This function returns immediately and processes the episode addition in the background.
        Episodes for the same group_id are processed sequentially to avoid race conditions.

        Args:
            name (str): Name of the episode
            episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                               properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                               automatically processed to extract entities and relationships.
            group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                     or a generated one.
            source (str, optional): Source type, must be one of:
                                   - 'text': For plain text content (default)
                                   - 'json': For structured data
                                   - 'message': For conversation-style content
            source_description (str, optional): Description of the source
            uuid (str, optional): Optional UUID for the episode

        Examples:
            # Adding plain text content
            add_memory(
                name="Company News",
                episode_body="Acme Corp announced a new product line today.",
                source="text",
                source_description="news article",
                group_id="some_arbitrary_string"
            )

            # Adding structured JSON data
            # NOTE: episode_body should be a JSON string (standard JSON escaping)
            add_memory(
                name="Customer Profile",
                episode_body='{"company": {"name": "Acme Technologies"}, "products": [{"id": "P001", "name": "CloudSync"}, {"id": "P002", "name": "DataMiner"}]}',
                source="json",
                source_description="CRM data"
            )
        """
        try:
            effective_group_id = group_id or cfg.graphiti.group_id

            episode_type = EpisodeType.text
            if source:
                try:
                    episode_type = EpisodeType[source.lower()]
                except (KeyError, AttributeError):
                    logger.warning(f"Unknown source type '{source}', using 'text'")
                    episode_type = EpisodeType.text

            await queue_svc.add_episode(
                group_id=effective_group_id,
                name=name,
                content=episode_body,
                source_description=source_description,
                episode_type=episode_type,
                entity_types=graphiti_svc.entity_types,
                uuid=uuid or None,
            )

            return SuccessResponse(
                message=f"Episode '{name}' queued for processing in group '{effective_group_id}'"
            )
        except Exception as e:
            logger.error(f'Error queuing episode: {e}')
            return ErrorResponse(error=f'Error queuing episode: {str(e)}')

    @server.tool()
    async def search_nodes(
        query: str,
        group_ids: list[str] | None = None,
        max_nodes: int = 10,
        entity_types: list[str] | None = None,
    ) -> NodeSearchResponse | ErrorResponse:
        """Search for nodes in the graph memory.

        Args:
            query (str): The search query
            group_ids (list[str], optional): Optional list of group IDs to filter results
            max_nodes (int): Maximum number of nodes to return (default: 10)
            entity_types (list[str], optional): Optional list of entity type names to filter by
        """
        try:
            client = await graphiti_svc.get_client()
            effective_group_ids = (
                group_ids
                if group_ids is not None
                else [cfg.graphiti.group_id]
                if cfg.graphiti.group_id
                else []
            )

            search_filters = SearchFilters(node_labels=entity_types)
            from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

            results = await client.search_(
                query=query,
                config=NODE_HYBRID_SEARCH_RRF,
                group_ids=effective_group_ids,
                search_filter=search_filters,
            )

            nodes = results.nodes[:max_nodes] if results.nodes else []
            if not nodes:
                return NodeSearchResponse(message='No relevant nodes found', nodes=[])

            node_results = []
            for node in nodes:
                attrs = node.attributes if hasattr(node, 'attributes') else {}
                attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}

                node_results.append(
                    NodeResult(
                        uuid=node.uuid,
                        name=node.name,
                        labels=node.labels if node.labels else [],
                        created_at=node.created_at.isoformat() if node.created_at else None,
                        summary=node.summary,
                        group_id=node.group_id,
                        attributes=attrs,
                    )
                )

            return NodeSearchResponse(message='Nodes retrieved successfully', nodes=node_results)
        except Exception as e:
            logger.error(f'Error searching nodes: {e}')
            return ErrorResponse(error=f'Error searching nodes: {str(e)}')

    @server.tool()
    async def search_memory_facts(
        query: str,
        group_ids: list[str] | None = None,
        max_facts: int = 10,
        center_node_uuid: str | None = None,
    ) -> FactSearchResponse | ErrorResponse:
        """Search the graph memory for relevant facts.

        Args:
            query (str): The search query
            group_ids (list[str], optional): Optional list of group IDs to filter results
            max_facts (int): Maximum number of facts to return (default: 10)
            center_node_uuid (str, optional): Optional UUID of a node to center the search around
        """
        try:
            if max_facts <= 0:
                return ErrorResponse(error='max_facts must be a positive integer')

            client = await graphiti_svc.get_client()
            effective_group_ids = (
                group_ids
                if group_ids is not None
                else [cfg.graphiti.group_id]
                if cfg.graphiti.group_id
                else []
            )

            relevant_edges = await client.search(
                group_ids=effective_group_ids,
                query=query,
                num_results=max_facts,
                center_node_uuid=center_node_uuid,
            )

            if not relevant_edges:
                return FactSearchResponse(message='No relevant facts found', facts=[])

            facts = [format_fact_result(edge) for edge in relevant_edges]
            return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
        except Exception as e:
            logger.error(f'Error searching facts: {e}')
            return ErrorResponse(error=f'Error searching facts: {str(e)}')

    @server.tool()
    async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
        """Delete an entity edge from the graph memory.

        Args:
            uuid (str): UUID of the entity edge to delete
        """
        try:
            client = await graphiti_svc.get_client()
            entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
            await entity_edge.delete(client.driver)
            return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
        except Exception as e:
            logger.error(f'Error deleting entity edge: {e}')
            return ErrorResponse(error=f'Error deleting entity edge: {str(e)}')

    @server.tool()
    async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
        """Delete an episode from the graph memory.

        Args:
            uuid (str): UUID of the episode to delete
        """
        try:
            client = await graphiti_svc.get_client()
            episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
            await episodic_node.delete(client.driver)
            return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
        except Exception as e:
            logger.error(f'Error deleting episode: {e}')
            return ErrorResponse(error=f'Error deleting episode: {str(e)}')

    @server.tool()
    async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
        """Get an entity edge from the graph memory by its UUID.

        Args:
            uuid (str): UUID of the entity edge to retrieve
        """
        try:
            client = await graphiti_svc.get_client()
            entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
            return format_fact_result(entity_edge)
        except Exception as e:
            logger.error(f'Error getting entity edge: {e}')
            return ErrorResponse(error=f'Error getting entity edge: {str(e)}')

    @server.tool()
    async def get_episodes(
        group_ids: list[str] | None = None,
        max_episodes: int = 10,
    ) -> EpisodeSearchResponse | ErrorResponse:
        """Get episodes from the graph memory.

        Args:
            group_ids (list[str], optional): Optional list of group IDs to filter results
            max_episodes (int): Maximum number of episodes to return (default: 10)
        """
        try:
            client = await graphiti_svc.get_client()
            effective_group_ids = (
                group_ids
                if group_ids is not None
                else [cfg.graphiti.group_id]
                if cfg.graphiti.group_id
                else []
            )

            if effective_group_ids:
                episodes = await EpisodicNode.get_by_group_ids(
                    client.driver, effective_group_ids, limit=max_episodes
                )
            else:
                episodes = []

            if not episodes:
                return EpisodeSearchResponse(message='No episodes found', episodes=[])

            episode_results = []
            for episode in episodes:
                episode_dict = {
                    'uuid': episode.uuid,
                    'name': episode.name,
                    'content': episode.content,
                    'created_at': episode.created_at.isoformat() if episode.created_at else None,
                    'source': episode.source.value
                    if hasattr(episode.source, 'value')
                    else str(episode.source),
                    'source_description': episode.source_description,
                    'group_id': episode.group_id,
                }
                episode_results.append(episode_dict)

            return EpisodeSearchResponse(
                message='Episodes retrieved successfully', episodes=episode_results
            )
        except Exception as e:
            logger.error(f'Error getting episodes: {e}')
            return ErrorResponse(error=f'Error getting episodes: {str(e)}')

    @server.tool()
    async def clear_graph(group_ids: list[str] | None = None) -> SuccessResponse | ErrorResponse:
        """Clear all data from the graph for specified group IDs.

        Args:
            group_ids (list[str], optional): Optional list of group IDs to clear. If not provided, clears the default group.
        """
        try:
            client = await graphiti_svc.get_client()
            effective_group_ids = (
                group_ids or [cfg.graphiti.group_id] if cfg.graphiti.group_id else []
            )

            if not effective_group_ids:
                return ErrorResponse(error='No group IDs specified for clearing')

            await clear_data(client.driver, group_ids=effective_group_ids)
            return SuccessResponse(
                message=f'Graph data cleared for group IDs: {", ".join(effective_group_ids)}'
            )
        except Exception as e:
            logger.error(f'Error clearing graph: {e}')
            return ErrorResponse(error=f'Error clearing graph: {str(e)}')

    @server.tool()
    async def get_status() -> StatusResponse:
        """Get the status of the Graphiti MCP server and database connection.

        Returns information about server health and database connectivity.
        """
        try:
            client = await graphiti_svc.get_client()
            # Test database connection
            async with client.driver.session() as session:
                result = await session.run('MATCH (n) RETURN count(n) as count')
                if result:
                    _ = [record async for record in result]

            provider_name = cfg.database.provider
            return StatusResponse(
                status='ok',
                message=f'Graphiti MCP server is running and connected to {provider_name} database',
            )
        except Exception as e:
            logger.error(f'Error checking database connection: {e}')
            return StatusResponse(
                status='error',
                message=f'Graphiti MCP server is running but database connection failed: {str(e)}',
            )


# ------------------------------------------------------------------------------
# Main Execution Block (Local Dev)
# ------------------------------------------------------------------------------


async def run_local():
    """Run the server locally using the factory pattern."""
    import argparse

    server = await create_server()

    parser = argparse.ArgumentParser(description='Run Biosciences Memory MCP server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--transport', choices=['http', 'stdio'], default='http')
    args, _ = parser.parse_known_args()

    logger.info(f'Starting MCP server with {args.transport} transport on {args.host}:{args.port}')

    if args.transport == 'stdio':
        await server.run_stdio_async()
    else:
        for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.handlers.clear()
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            uvicorn_logger.addHandler(handler)
            uvicorn_logger.propagate = False

        await server.run_http_async(host=args.host, port=args.port)


if __name__ == '__main__':
    try:
        asyncio.run(run_local())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error(f'Fatal error: {str(e)}')
        sys.exit(1)

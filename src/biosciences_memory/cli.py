"""
Biosciences Memory MCP Server — CLI entry point with argparse and YAML config.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import fastmcp
from dotenv import load_dotenv
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from biosciences_memory.config.schema import GraphitiConfig
from biosciences_memory.server import (
    SEMAPHORE_LIMIT,
    GraphitiService,
    QueueService,
    _register_tools,
)

# Load .env file
mcp_server_dir = Path(__file__).parent.parent.parent
env_file = mcp_server_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def configure_uvicorn_logging():
    """Configure uvicorn loggers to match our format after they're created."""
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False


async def initialize_server():
    """Parse CLI arguments and initialize the Biosciences Memory server."""
    from starlette.responses import JSONResponse

    parser = argparse.ArgumentParser(
        description="Run the Biosciences Memory MCP server with YAML configuration support"
    )

    # Configuration file argument
    default_config = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to YAML configuration file (default: config/config.yaml)",
    )

    # Transport arguments
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        help="Transport to use: http (recommended, default) or stdio (standard I/O)",
    )
    parser.add_argument("--host", help="Host to bind the MCP server to")
    parser.add_argument("--port", type=int, help="Port to bind the MCP server to")

    # LLM configuration
    parser.add_argument("--model", help="Model name to use with the LLM client")

    # Graphiti-specific arguments
    parser.add_argument(
        "--group-id",
        help='Namespace for the graph. If not provided, uses config file or "main".',
    )
    parser.add_argument("--user-id", help="User ID for tracking operations")
    parser.add_argument(
        "--destroy-graph",
        action="store_true",
        help="Destroy all Graphiti graphs on startup",
    )

    args = parser.parse_args()

    # Set config path in environment for the settings to pick up
    if args.config:
        os.environ["CONFIG_PATH"] = str(args.config)

    # Load configuration with environment variables and YAML
    config = GraphitiConfig()

    # Apply CLI overrides
    config.apply_cli_overrides(args)

    if hasattr(args, "destroy_graph"):
        config.destroy_graph = args.destroy_graph

    # Log configuration details
    logger.info("Using configuration:")
    logger.info(f"  - LLM: {config.llm.provider} / {config.llm.model}")
    logger.info(f"  - Embedder: {config.embedder.provider} / {config.embedder.model}")
    logger.info(f"  - Database: {config.database.provider}")
    logger.info(f"  - Group ID: {config.graphiti.group_id}")
    logger.info(f"  - Transport: {config.server.transport}")

    # Handle graph destruction if requested
    if hasattr(config, "destroy_graph") and config.destroy_graph:
        logger.warning("Destroying all Graphiti graphs as requested...")
        temp_service = GraphitiService(config, SEMAPHORE_LIMIT)
        await temp_service.initialize()
        client = await temp_service.get_client()
        await clear_data(client.driver)
        logger.info("All graphs destroyed")

    # Initialize services
    graphiti_service = GraphitiService(config, SEMAPHORE_LIMIT)
    queue_service = QueueService()
    await graphiti_service.initialize()

    graphiti_client = await graphiti_service.get_client()
    await queue_service.initialize(graphiti_client)

    # Create MCP server
    from biosciences_memory.server import GRAPHITI_MCP_INSTRUCTIONS

    mcp = fastmcp.FastMCP(
        "Biosciences Memory Server",
        instructions=GRAPHITI_MCP_INSTRUCTIONS,
    )

    # Register tools and routes
    _register_tools(mcp, config, graphiti_service, queue_service)

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):
        return JSONResponse({"status": "healthy", "service": "biosciences-memory"})

    # Set MCP server settings
    if config.server.host:
        fastmcp.settings.host = config.server.host
    if config.server.port:
        fastmcp.settings.port = config.server.port

    return mcp, config.server


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    mcp, server_config = await initialize_server()

    logger.info(f"Starting MCP server with transport: {server_config.transport}")
    if server_config.transport == "stdio":
        await mcp.run_stdio_async()
    elif server_config.transport == "http":
        display_host = "localhost" if fastmcp.settings.host == "0.0.0.0" else fastmcp.settings.host
        logger.info(
            f"Running MCP server with HTTP transport on "
            f"{fastmcp.settings.host}:{fastmcp.settings.port}"
        )
        logger.info(f"  MCP Endpoint: http://{display_host}:{fastmcp.settings.port}/mcp/")
        configure_uvicorn_logging()
        await mcp.run_http_async(transport="http")
    else:
        raise ValueError(f"Unsupported transport: {server_config.transport}")


def main():
    """Main function to run the Biosciences Memory MCP server."""
    try:
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Error initializing server: {str(e)}")
        raise


if __name__ == "__main__":
    main()

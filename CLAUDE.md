# CLAUDE.md — biosciences-memory

## Purpose

Knowledge graph persistence layer using Graphiti, Neo4j, and Qdrant. This repo is owned by the **Memory Engineer** agent.

## MCP Server Connections

This repo's `.mcp.json` defines 5 MCP server connections for graph operations:

| Connection | Transport | Endpoint | Purpose |
|------------|-----------|----------|---------|
| `graphiti-aura` | stdio | `/home/donbr/graphiti-fastmcp/scripts/run_mcp_server.sh` | Graphiti FastMCP for Neo4j Aura |
| `neo4j-aura-management` | HTTP | `:8004` | Neo4j Aura instance management |
| `neo4j-aura-cypher` | HTTP | `:8003` | Direct Cypher queries on Aura |
| `graphiti-docker` | HTTP | `:8002` | Graphiti FastMCP for local Docker |
| `neo4j-docker-cypher` | HTTP | `:8005` | Direct Cypher on local Docker |

## Dual Environment

| Environment | Use Case | Config |
|-------------|----------|--------|
| **Neo4j Aura** (cloud) | Production graph database | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` |
| **Neo4j Docker** (local) | Development and testing | `docker-compose.yml` |

## Environment Variables

Required (documented in `.env.example`):

```bash
# Neo4j Aura (cloud)
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

# Neo4j Aura API
NEO4J_AURA_CLIENT_ID=...
NEO4J_AURA_CLIENT_SECRET=...

# External APIs
BIOGRID_API_KEY=...
NCBI_API_KEY=...

# LLM
OPENAI_API_KEY=...

# FastMCP Cloud
FASTMCP_CLOUD_ENDPOINT=...
```

## Namespace Policies

- Graph entities use canonical CURIEs from ADR-001 §5 (e.g., `HGNC:1100`, `UniProtKB:P38398`)
- Namespace isolation between research projects via graph labels
- Entity deduplication uses cross-reference matching

## Dependencies

- **Upstream**: `biosciences-architecture` (schema definitions)
- **Downstream**: `biosciences-research` (graph persistence), `biosciences-deepagents` (PERSIST phase)

## Development Commands

```bash
uv sync                          # Install dependencies

# Verify MCP connections
# (start relevant MCP servers first)
```

## Conventions

- Python >=3.11, uv, hatchling, ruff, pyright
- Pydantic v2 for all models
- httpx for async HTTP
- pytest with marker-based test organization

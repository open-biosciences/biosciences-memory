# biosciences-memory

Graphiti knowledge graph memory layer for the [Open Biosciences](https://github.com/open-biosciences) platform. Curated migration of graphiti-fastmcp into an OpenAI + Neo4j only package with biosciences domain entity types.

## Architecture

Factory-pattern FastMCP server with:

- **9 MCP tools**: `add_memory`, `search_nodes`, `search_memory_facts`, `get_episodes`, `get_entity_edge`, `delete_entity_edge`, `delete_episode`, `clear_graph`, `get_status`
- **Async queue-based episode processing** — sequential per `group_id`, parallel across groups
- **Pydantic-Settings config** with YAML + env var expansion
- **Custom Neo4j driver** for Aura connection pool management
- **Both factory (FastMCP Cloud) and CLI entry points**
- **14 entity types** — 9 generic + 5 biosciences: Gene, Protein, Drug, Disease, Pathway
- **5 edge types** with `edge_type_map` for domain relationships

## Installation

```bash
uv sync                    # Install dependencies
uv sync --extra dev        # Install with dev dependencies
```

## Running the Server

```bash
# Factory mode (FastMCP Cloud compatible)
uv run python -m biosciences_memory.server

# CLI mode with YAML config
uv run python -m biosciences_memory.cli --config config/config.yaml

# CLI with overrides
uv run python -m biosciences_memory.cli --model gpt-4.1 --group-id my-project
```

## Running Tests

```bash
uv run pytest -m unit -v           # Unit tests (no external deps)
uv run pytest -m integration -v    # Integration tests
uv run pytest -v                   # All tests
```

## Configuration

The `config.yaml` file supports `${VAR:default}` env var expansion. Configuration priority is:

**CLI > env vars > YAML > defaults**

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

## MCP Server Connections

Five MCP servers are configured in `.mcp.json`:

| Connection | Transport | Status | Purpose |
|------------|-----------|--------|---------|
| `graphiti-aura` | stdio | Read-only (write-frozen) | Graphiti FastMCP for Neo4j Aura cloud |
| `neo4j-aura-management` | HTTP `:8004` | Read-only (write-frozen) | Neo4j Aura instance management |
| `neo4j-aura-cypher` | HTTP `:8003` | Read-only (write-frozen) | Direct Cypher queries on Aura |
| `graphiti-docker` | HTTP `:8002` | **Active** | Graphiti FastMCP for local Docker Neo4j |
| `neo4j-docker-cypher` | HTTP `:8005` | **Active** | Direct Cypher on local Docker Neo4j |

### Aura Write-Freeze Policy

The free-tier Neo4j Aura instance is at capacity (~5k nodes). This is a known, accepted state.

- **Reads from Aura are fine** — searches, queries, and analytics all work
- **No new writes to Aura** — use `graphiti-docker` for all new memory episodes
- All new data goes to the local Docker Neo4j instance

## Design References

- **Graphiti Custom Entity and Edge Types**: https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types — Pattern for defining domain-specific Pydantic entity and edge types with edge_type_map for relationship constraints
- **Graphiti Graph Namespacing**: https://help.getzep.com/graphiti/core-concepts/graph-namespacing — group_id-based namespace isolation for multi-tenant knowledge graphs
- **Graphiti Communities**: https://help.getzep.com/graphiti/core-concepts/communities — Leiden algorithm community detection for entity clustering
- **FastMCP Testing Patterns**: https://gofastmcp.com/patterns/testing — In-memory Client(server) pattern for pytest-asyncio MCP tool testing
- **FastMCP Test Organization**: https://gofastmcp.com/development/tests — Test structure and async fixture patterns
- **Neo4j Python Driver Pool Config**: https://github.com/neo4j/neo4j-python-driver/issues/316 — Connection pool tuning for Aura cloud idle timeouts

## Agent Ownership

Maintained by the **Memory Engineer** (Agent 4). See [AGENTS.md](https://github.com/open-biosciences/biosciences-program/blob/main/AGENTS.md) for full team definitions.

## Dependencies

| Direction | Repository | Relationship |
|-----------|------------|--------------|
| Upstream | [biosciences-architecture](https://github.com/open-biosciences/biosciences-architecture) | ADRs and schemas |
| Downstream | [biosciences-research](https://github.com/open-biosciences/biosciences-research) | Persists research results to knowledge graph |
| Downstream | [biosciences-deepagents](https://github.com/open-biosciences/biosciences-deepagents) | PERSIST phase writes to knowledge graph |

## Migration Waves

| Wave | Status | Scope |
|------|--------|-------|
| 2 — Platform | Complete | .mcp.json, .env.example, 5 MCP connections |
| 4 — Validation | Complete | graphiti-fastmcp code migrated to biosciences_memory package |

## License

MIT

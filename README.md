# biosciences-memory

Graphiti + Neo4j knowledge graph persistence layer for the [Open Biosciences](https://github.com/open-biosciences) platform. Manages entity resolution, graph schemas, and namespace policies across cloud and local Neo4j environments.

## Status

**Wave 2 (Platform) complete.** Configuration files (`.mcp.json`, `.env.example`) are in place and the 5 MCP server connections are operational. Full graphiti-fastmcp server code migration is scheduled for Wave 4.

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

### Dual Environment

| Environment | Use Case | Active Connections |
|-------------|----------|--------------------|
| Neo4j Aura (cloud) | Historical read access | graphiti-aura, neo4j-aura-management, neo4j-aura-cypher |
| Neo4j Docker (local) | All new writes, active development | graphiti-docker, neo4j-docker-cypher |

## Environment Setup

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

Required variables:

```bash
# Neo4j Aura (cloud — for reads)
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_AURA_CLIENT_ID=...
NEO4J_AURA_CLIENT_SECRET=...

# LLM
OPENAI_API_KEY=...

# External APIs
BIOGRID_API_KEY=...
NCBI_API_KEY=...
```

## What Wave 4 Will Add

Wave 4 will migrate the `graphiti-fastmcp` predecessor service into this repo as the `biosciences_memory` Python package:

- **GraphitiFastMCP server** — FastMCP Cloud-compatible entrypoint
- **Queue service** — sequential per-group episode processing to prevent race conditions
- **Config schema** — multi-source config: env vars, YAML, CLI, defaults
- **Entity models** — domain-specific Pydantic v2 models for biosciences entities
- **Test suite** — unit, integration, and e2e tests (pytest marker-based)

Before migration, the predecessor at `graphiti-fastmcp` needs alignment to hatchling build backend, ruff config, and platform pytest marker conventions.

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
| 2 — Platform | ✅ Complete | `.mcp.json`, `.env.example`, 5 MCP connections |
| 4 — Validation | ⬜ Not Started | graphiti-fastmcp code → `biosciences_memory` package |

## License

MIT

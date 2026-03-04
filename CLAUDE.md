# CLAUDE.md — biosciences-memory

## Purpose
Knowledge graph persistence layer using Graphiti and Neo4j. This repository is owned by the **Memory Engineer (Agent 4)** and serves as the long-term memory layer for the Open Biosciences platform.

## Local Environment (Docker-First)
The project is configured to run primarily in a containerized environment to ensure consistency and avoid conflicts with local Neo4j installations.

### Port Mappings (Host Access)
| Service | Protocol | Host URL | Container Port | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Memory MCP** | HTTP | `http://localhost:8007` | 8003 | Graphiti Logic & MCP Tools |
| **Cypher MCP** | HTTP | `http://localhost:8008` | 8005 | Direct Neo4j Query Tool |
| **Neo4j Bolt** | Bolt | `bolt://localhost:7688` | 7687 | Database Binary Connection |
| **Neo4j UI** | HTTP | `http://localhost:7475` | 7474 | Database Browser UI |

---

## Development Commands

### Orchestration
```bash
uv sync                          # Install dependencies
docker compose up -d             # Start full stack (Neo4j + 2 MCPs)
docker compose logs -f           # Follow logs
docker compose down              # Stop containers (preserves data)
docker compose down -v           # Stop and WIPE data

```

### Manual/Local Run

Uses host `.env` settings. Always use the module flag to ensure correct relative imports.

```bash
uv run python -m biosciences_memory.server

```

---

## Environment Variables

The project uses a `.env` file for local development. Note that `docker-compose.yml` overrides some of these for internal container networking.

```bash
# Neo4j Connection (Host-side for scripts/tests)
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=demodemo

# OpenAI (Required for extraction)
OPENAI_API_KEY=sk-...

# Application Defaults
GRAPHITI_GROUP_ID=main
SEMAPHORE_LIMIT=10

```

---

## MCP Client Configuration (`.mcp.json`)

Ensure your local MCP client (e.g., Claude Desktop) is aligned with the Docker host ports:

```json
{
  "mcpServers": {
    "biosciences-neo4j-cypher": {
      "type": "http",
      "url": "http://localhost:8008/mcp/"
    },
    "biosciences-graphiti": {
      "type": "http",
      "url": "http://localhost:8007/mcp/"
    }
  }
}

```

---

## Core Architecture Patterns

| Component | Location | Pattern |
| --- | --- | --- |
| **Server** | `src/biosciences_memory/server.py` | FastMCP Factory Pattern |
| **Queueing** | `src/biosciences_memory/services/queue_service.py` | Sequential per-group processing |
| **Config** | `src/biosciences_memory/config/schema.py` | Pydantic-Settings with YAML expansion |
| **Models** | `src/biosciences_memory/models/entity_types.py` | Domain-specific entity/edge definitions |

---

## Conventions & Maintenance

* **Python**: >=3.11, uv, hatchling, ruff, pyright.
* **Database**: Neo4j 5.x with **APOC** (required for Graphiti).
* **Namespacing**: Entities use canonical CURIEs (e.g., `HGNC:1100`). Use `group_id` for project/research isolation.
* **Testing**: Use `pytest -m unit` for logic and `pytest -m integration` for Neo4j-dependent tests.
* **Migration**: This project has been migrated from an external script model to a self-contained package.
* **Port Note**: Port 8005 is often pre-allocated on local machines; host mapping is moved to **8008** for Cypher to avoid collisions.

**MCP Tools Exposed**: `add_memory`, `search_nodes`, `search_memory_facts`, `get_episodes`, `get_entity_edge`, `delete_entity_edge`, `delete_episode`, `clear_graph`, `get_status`.

# biosciences-memory

Graphiti knowledge graph memory layer for the **Open Biosciences** platform.

---

## Architecture

Factory-pattern **FastMCP** server featuring:

* **9 MCP Tools:** `add_memory`, `search_nodes`, `search_memory_facts`, `get_episodes`, `get_entity_edge`, `delete_entity_edge`, `delete_episode`, `clear_graph`, `get_status`
* **Async Processing:** Queue-based episode processing—sequential per `group_id`, parallel across groups.
* **Config:** Pydantic-Settings configuration with YAML + environment variable expansion.
* **Domain Model:** 14 entity types (9 generic + 5 biosciences: `Gene`, `Protein`, `Drug`, `Disease`, `Pathway`).
* **Relationships:** 5 edge types with `edge_type_map` for domain relationships.

---

## Installation

```bash
uv sync                    # Install dependencies
uv sync --extra dev        # Install with dev dependencies

```

---

## Configuration

The application uses a layered configuration approach: **CLI > Environment Variables > YAML > Defaults**.

### 1. Environment Variables (.env)

Copy `.env.example` to `.env`. This file is used for local `uv run` commands and provides values for Docker injection.

```env
# Neo4j Connection (Host-side mapping for local dev)
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=demodemo

# OpenAI
OPENAI_API_KEY=sk-your-key-here

```

### 2. YAML Configuration (config/config.yaml)

The `config.yaml` file supports `${VAR:default}` expansion.

* **Local Dev:** Defaults to `bolt://localhost:7688`.
* **Docker:** The `NEO4J_URI` is overridden in `docker-compose.yml` to `bolt://neo4j:7687` for internal networking.

---

## Local Docker Environment

The easiest way to run the full stack is via **Docker Compose**. This starts Neo4j, the Cypher query tool, and the Biosciences Memory server.

```bash
docker compose up -d

```

### Exposed Services (Host Access)

| Service | Host URL | Internal Port | Description |
| --- | --- | --- | --- |
| **Neo4j Browser** | `http://localhost:7475` | 7474 | Database UI |
| **Neo4j Bolt** | `bolt://localhost:7688` | 7687 | Database Binary Protocol |
| **Memory MCP** | `http://localhost:8007` | 8003 | Graphiti Logic Server |
| **Cypher MCP** | `http://localhost:8008` | 8005 | Direct Neo4j Query Tool |

---

## MCP Client Configuration

Update your `.mcp.json` (e.g., for Claude Desktop) to connect to the local Docker services. Note that the Cypher tool uses port `8008` on the host to avoid local allocation conflicts.

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

## Running the Server (Manual/Local)

If you prefer to run the server outside of Docker while still using the Docker-managed Neo4j instance:

```bash
# Ensure .env is configured with NEO4J_URI=bolt://localhost:7688
uv run python -m biosciences_memory.server

```

## Running Tests

```bash
uv run pytest -m unit -v           # Unit tests (no external deps)
uv run pytest -m integration -v    # Integration tests (requires Neo4j)

```

---

## Design References

* **Graphiti Custom Entity Types:** [Custom Types Documentation](https://www.google.com/search?q=https://github.com/getgraphiti/graphiti)
* **Graphiti Namespacing:** [Namespacing Documentation](https://www.google.com/search?q=https://github.com/getgraphiti/graphiti)
* **FastMCP Testing:** [Testing Patterns](https://github.com/modelcontextprotocol/python-sdk)

## Agent Ownership

Maintained by the **Memory Engineer (Agent 4)**. See `AGENTS.md` for full team definitions.

## License

MIT

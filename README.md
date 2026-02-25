# biosciences-memory

Graphiti + Neo4j knowledge graph persistence layer for the [Open Biosciences](https://github.com/open-biosciences) platform. Manages entity resolution, graph schemas, and namespace policies across cloud and local Neo4j environments.

## Status

**Pending Wave 2 (Platform) migration.** Some configuration files (`.mcp.json`, `.env.example`) are already in place. Full content is being migrated from the predecessor `lifesciences-research` repository.

## What's Coming

After migration, this repository will contain:

- **Graph schema design** for biosciences domain entities and relationships
- **Entity resolution** logic for deduplication and linking
- **Namespace policies** governing data isolation and access patterns
- **Dual environment support**
  - Neo4j Aura (cloud production)
  - Neo4j Docker (local development)
- **5 MCP server connections**: graphiti-aura, neo4j-aura-management, neo4j-aura-cypher, graphiti-docker, neo4j-docker-cypher

## Agent Ownership

Maintained by the **Memory Engineer** agent (Agent 4). See [AGENTS.md](../biosciences-program/AGENTS.md) for full team definitions.

## Dependencies

| Direction | Repository | Relationship |
|-----------|------------|--------------|
| Upstream | biosciences-architecture | Schemas and ADRs |
| Downstream | biosciences-research | Persists research results to knowledge graph |
| Downstream | biosciences-deepagents | PERSIST phase writes to knowledge graph |

## Related Repositories

- [biosciences-architecture](https://github.com/open-biosciences/biosciences-architecture) -- ADRs and schemas
- [biosciences-research](https://github.com/open-biosciences/biosciences-research) -- Research workflows
- [biosciences-deepagents](https://github.com/open-biosciences/biosciences-deepagents) -- LangGraph multi-agent system

## License

MIT

---
description: Run verification sequence to confirm biosciences-memory stack is working
allowed-tools:
  - mcp__biosciences-graphiti__get_status
  - mcp__biosciences-graphiti__search_nodes
  - mcp__biosciences-neo4j-cypher__read_neo4j_cypher
  - Bash(docker ps:*)
---

Run verification sequence to confirm the biosciences-memory Docker stack is working.

## Instructions

Execute these checks in order. Report results but do NOT attempt fixes without user approval.

### Check 1: Docker Containers

```bash
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}" | grep biosciences
```

**Expected**: 3 containers running with these port mappings:
- `biosciences-neo4j` — 7475 (browser UI), 7688 (Bolt)
- `biosciences-memory-mcp` — 8007
- `biosciences-cypher-mcp` — 8008

### Check 2: Graphiti MCP Server Health

```python
mcp__biosciences-graphiti__get_status()
```

**Expected**: Status OK with database connection confirmed.

### Check 3: Cypher Query — Namespace Listing

```python
mcp__biosciences-neo4j-cypher__read_neo4j_cypher(
    query="MATCH (e:Episodic) RETURN DISTINCT e.group_id AS group_id, count(*) AS episodes ORDER BY episodes DESC"
)
```

**Expected**: List of group_ids with episode counts, or empty results for fresh environment.

### Check 4: Semantic Search Test

```python
mcp__biosciences-graphiti__search_nodes(
    query="test",
    group_ids=["dev_main"],
    max_nodes=3
)
```

**Expected**: Returns entity nodes if data exists, or empty results for fresh environment. Confirms embedding/search pipeline is functional.

## After Running

Report your findings to the user:
- ✅ All checks passed — biosciences-memory stack is healthy
- ⚠️ Some checks failed — describe what you found, ASK before fixing
- ❌ Connection errors — report error, do NOT attempt destructive fixes
- ✅ Empty query results are normal for a fresh environment

## Troubleshooting

**Containers not running**:
```bash
docker compose up -d
```

**biosciences-graphiti MCP not available**:
- Ensure `.mcp.json` has `biosciences-graphiti` configured at `http://localhost:8007/mcp/`
- Check container: `docker compose logs biosciences-memory-mcp`

**Cypher query fails**:
- Check Neo4j container: `docker compose logs biosciences-neo4j`
- Verify Bolt connection: `bolt://localhost:7688`
- Check Cypher MCP: `docker compose logs biosciences-cypher-mcp`

## Related Commands

- `/graphiti-health` — Quick health check for this stack
- `/graphiti-docker-stats` — Detailed namespace analytics

For full workspace + Aura verification, use the global `/graphiti-verify` skill.

## Remember

- Empty results usually mean no data yet, NOT a broken system
- NEVER run `clear_graph` or delete commands without explicit user request
- When in doubt, ASK THE USER

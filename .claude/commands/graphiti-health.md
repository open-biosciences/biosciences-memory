---
description: Display real-time health status for the biosciences-memory Docker stack
allowed-tools:
  - mcp__biosciences-graphiti__get_status
  - mcp__biosciences-neo4j-cypher__read_neo4j_cypher
  - Bash(docker ps:*)
---

Display real-time health status for the biosciences-memory Docker stack (Neo4j + 2 MCP servers).

## Instructions

Execute these health checks to provide a stack status report. Report results with color-coded indicators.

### Check 1: Docker Containers

```bash
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}" | grep biosciences
```

**Expected**: 3 containers running:
- `biosciences-neo4j` — ports 7475 (UI), 7688 (Bolt)
- `biosciences-memory-mcp` — port 8007
- `biosciences-cypher-mcp` — port 8008

**Health Indicators**:
- ✅ All 3 containers running
- ⚠️ Some containers running (partial stack)
- ❌ No containers found — start with `docker compose up -d`

### Check 2: Graphiti MCP Server Health

```python
mcp__biosciences-graphiti__get_status()
```

**Expected Response**:
```json
{
  "status": "ok",
  "message": "Graphiti MCP server is running and connected to neo4j database"
}
```

**Health Indicators**:
- ✅ `status == "ok"` — Server healthy and database connected
- ❌ Any other status or error — Server unavailable or database connection failed

### Check 3: Neo4j Data Overview

```python
mcp__biosciences-neo4j-cypher__read_neo4j_cypher(
    query="MATCH (e:Episodic) RETURN count(DISTINCT e.group_id) AS namespaces, count(*) AS episodes"
)
```

**Health Indicators**:
- ✅ Query succeeds — Neo4j accessible via Cypher MCP
- ✅ Zero namespaces/episodes — normal for fresh environment
- ❌ Query fails — check Neo4j container and Cypher MCP connection

## Output Format

```
biosciences-memory Stack Health
Generated: <timestamp>

Overall Status: ✅ HEALTHY | ⚠️ DEGRADED | ❌ UNHEALTHY

=== Docker Containers ===
biosciences-neo4j:       ✅ Running (ports 7475, 7688)
biosciences-memory-mcp:  ✅ Running (port 8007)
biosciences-cypher-mcp:  ✅ Running (port 8008)

=== Graphiti MCP (biosciences-graphiti) ===
Status: ✅ Running
Database: ✅ Connected

=== Neo4j Data ===
Namespaces: <count>
Episodes: <count>

=== Summary ===
All biosciences-memory services operational.
```

## Troubleshooting

**Containers not running**:
```bash
docker compose up -d
docker compose logs -f  # check for startup errors
```

**biosciences-graphiti MCP not available**:
- **Symptom**: Tool `mcp__biosciences-graphiti__get_status` not found
- **Solution**: Ensure `.mcp.json` has `biosciences-graphiti` configured at `http://localhost:8007/mcp/`

**Cypher MCP connection failed**:
- **Symptom**: `mcp__biosciences-neo4j-cypher__read_neo4j_cypher` fails
- **Solution**: Check `biosciences-cypher-mcp` container logs: `docker compose logs biosciences-cypher-mcp`
- Verify Neo4j is accessible at `bolt://localhost:7688`

## Related Commands

- `/graphiti-docker-stats` — Detailed namespace analytics for this stack
- `/graphiti-verify` — Comprehensive verification for this stack

For full workspace + Aura coverage, use the global `/graphiti-health` skill.

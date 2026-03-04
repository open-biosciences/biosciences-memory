---
description: Generate statistics for biosciences-memory Neo4j namespace management
allowed-tools:
  - mcp__biosciences-neo4j-cypher__read_neo4j_cypher
  - mcp__mcp-server-time__get_current_time
---

Generate statistics for biosciences-memory Neo4j namespace management and cleanup.

## Instructions

Execute these queries sequentially to build a development environment report. Report results to the user in a formatted summary.

**Purpose**: Namespace cleanup, development progress tracking, and quick sanity checks.

### Query 1: Namespace Overview

```python
mcp__biosciences-neo4j-cypher__read_neo4j_cypher(
    query="""
    MATCH (e:Episodic)
    RETURN e.group_id AS namespace,
           count(*) AS episodes,
           min(e.created_at) AS first_episode,
           max(e.created_at) AS last_episode
    ORDER BY episodes DESC
    """
)
```

**Purpose**: Shows all namespaces with episode counts and identifies cleanup candidates.

### Query 2: Development Namespace Breakdown

```python
mcp__biosciences-neo4j-cypher__read_neo4j_cypher(
    query="""
    MATCH (e:Episodic)
    WHERE e.group_id STARTS WITH 'dev_'
       OR e.group_id STARTS WITH 'test_'
       OR e.group_id STARTS WITH 'experimental_'
       OR e.group_id STARTS WITH 'scratch_'
       OR e.group_id STARTS WITH 'demo_'
    RETURN e.group_id AS namespace,
           count(*) AS episodes,
           max(e.created_at) AS last_activity
    ORDER BY last_activity DESC
    """
)
```

**Purpose**: Identifies development namespaces with activity timestamps for cleanup decisions.

### Query 3: Total Graph Size

```python
mcp__biosciences-neo4j-cypher__read_neo4j_cypher(
    query="""
    MATCH (n:Entity)
    WITH count(n) AS entity_count
    MATCH ()-[r]->()
    RETURN entity_count,
           count(r) AS relationship_count
    """
)
```

**Purpose**: Simple counts for development environment capacity check.

### Query 4: Recent Activity (Last 7 Days)

```python
mcp__biosciences-neo4j-cypher__read_neo4j_cypher(
    query="""
    MATCH (e:Episodic)
    WHERE e.created_at > datetime() - duration('P7D')
    RETURN e.group_id AS namespace,
           count(*) AS recent_episodes,
           max(e.created_at) AS last_episode
    ORDER BY recent_episodes DESC
    """
)
```

**Purpose**: Shows active development areas in the last week.

### Query 5: Cleanup Candidates

```python
mcp__biosciences-neo4j-cypher__read_neo4j_cypher(
    query="""
    MATCH (e:Episodic)
    WHERE (e.group_id STARTS WITH 'test_' OR e.group_id STARTS WITH 'scratch_')
      AND e.created_at < datetime() - duration('P7D')
    RETURN e.group_id AS namespace,
           count(*) AS episodes,
           max(e.created_at) AS last_activity,
           duration.between(max(e.created_at), datetime()).days AS days_inactive
    ORDER BY days_inactive DESC
    """
)
```

**Purpose**: Identifies stale test/scratch namespaces older than 7 days for cleanup.

## Output Format

Format the results as a development environment report with these sections:

### 1. Namespace Summary

- Total namespaces in development environment
- Breakdown by prefix (dev_*, test_*, experimental_*, scratch_*, demo_*)
- Highlight empty namespaces

### 2. Graph Size

- Total entities
- Total relationships

### 3. Recent Activity

- Active namespaces (last 7 days)
- Number of episodes added per namespace
- Current development focus areas

### 4. Cleanup Recommendations

- Stale test/scratch namespaces (>7 days old)
- Empty namespaces
- Suggested cleanup commands

## Troubleshooting

**Empty results**:
- **Cause**: No data in development environment (expected for fresh instances)
- **Solution**: This is normal - create test data with dev_*, test_*, or experimental_* namespaces

**MCP server not available**:
- **Symptom**: Tool `mcp__biosciences-neo4j-cypher__read_neo4j_cypher` not found
- **Solution**: Start biosciences-memory Docker stack:
  ```bash
  docker compose up -d
  ```
- **Verify**: Check containers are running: `docker ps | grep biosciences`

**Connection errors**:
- **Symptom**: "Failed to connect" errors
- **Solutions**:
  - Verify Docker containers are running: `docker ps | grep biosciences`
  - Check Neo4j is accessible at `bolt://localhost:7688`
  - Verify MCP server containers: `docker ps | grep -E "(biosciences-neo4j|biosciences-cypher-mcp|biosciences-memory-mcp)"`
  - Check logs: `docker compose logs biosciences-cypher-mcp`

## Related Commands

- `/graphiti-health` - System health for biosciences-memory stack
- `/graphiti-verify` - Comprehensive environment validation

## Notes

- This command targets the **biosciences-memory Docker stack** (Neo4j at bolt://localhost:7688)
- For workspace-wide Docker stats, use the global `/graphiti-docker-stats` skill
- Optimized for frequent use during active development
- All queries are read-only (no writes)

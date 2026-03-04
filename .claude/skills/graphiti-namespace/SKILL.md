---
name: graphiti-namespace
description: Namespace validation and guidance for Graphiti knowledge graph operations. Use when adding episodes to Graphiti, creating new namespaces, or before any write operation to Neo4j. Triggers on "add episode", "add memory", "create namespace", "which namespace", or when using graphiti-aura/graphiti-docker/biosciences-graphiti MCP tools.
---

# Graphiti Namespace Guidance

Ensures correct namespace selection, environment targeting, and policy compliance before Graphiti write operations.

## Quick Decision Tree

```
Is this production-worthy, persistent knowledge?
├── YES → Use Aura (graphiti-aura MCP) — write-frozen, confirm with user
│   └── Does namespace exist in registry?
│       ├── YES → Check category and write approval
│       └── NO → Ask user before creating
│
└── NO → Which scope?
    ├── Repo-isolated dev/test → Use biosciences-graphiti MCP (localhost:8007, Neo4j :7688)
    │   └── Use prefix: dev_*, test_*, experimental_*, scratch_*, demo_*
    │
    └── Cross-repo shared dev → Use graphiti-docker MCP (localhost:8002, Neo4j :7687)
        └── Use prefix: dev_*, test_*, experimental_*, scratch_*, demo_*
```

## Before Any Write Operation

1. **Verify environment**: Run `/verify`
2. **Check registry**: Read `docs/NAMESPACE_REGISTRY.yaml`
3. **Validate namespace**: See references/validation.md

## Namespace Categories

| Category | Environment | Write Approval | Action |
|----------|-------------|----------------|--------|
| Essential | Aura (write-frozen) | Required | Keep |
| Reference | graphiti-docker or biosciences-graphiti | Optional | Migrate |
| Experimental | graphiti-docker or biosciences-graphiti | None | Review |
| Archive | N/A | None | Export & delete |
| Prohibited | Never | Blocked | Delete |

## Anti-Patterns

**Version namespaces are forbidden.** See references/anti-patterns.md for details.

```
# WRONG
don_branson_resume_v1 → v2 → v3

# RIGHT
don_branson_resume  # Graphiti tracks versions via valid_at/invalid_at
```

## Key Files

| File | Purpose |
|------|---------|
| `docs/NAMESPACE_REGISTRY.yaml` | All 22 namespaces with policies |
| `docs/NAMESPACE_POLICY.md` | Full policy documentation |
| `reference/GRAPHITI_PATTERNS_AND_RECIPES.md` | Pattern 6: Knowledge Evolution |

## When to Ask User

Always ask before:
- Creating new production namespace
- Writing to Essential namespace (first time in session)
- Any deletion operation

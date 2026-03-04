# Namespace Validation

## Validation Steps

### Step 1: Environment Check

```bash
/verify  # Confirms active environment
```

Expected output shows either:
- `bolt://localhost:7687` → Docker (safe for dev_* namespaces)
- `neo4j+s://*.databases.neo4j.io` → Aura (production only)

### Step 2: Registry Lookup

Read `docs/NAMESPACE_REGISTRY.yaml` to find namespace:

```yaml
namespaces:
  graphiti_meta_knowledge:
    category: essential
    write_approval: required
```

### Step 3: Category Validation

| Category | Allowed in Aura | Allowed in Docker |
|----------|-----------------|-------------------|
| essential | Yes (with approval) | No |
| reference | Yes | Yes (preferred) |
| experimental | No | Yes |
| archive | Export only | No |
| prohibited | Never | Never |

### Step 4: Prefix Validation (Docker only)

Valid prefixes for Docker namespaces:
- `dev_*` - Active development
- `test_*` - Automated testing
- `experimental_*` - Research
- `scratch_*` - Throwaway (delete immediately)
- `demo_*` - Demonstrations
- `ref_*` - Reference data migrated from production

## Prohibited Namespace Names

Never use:
- `main`, `default`, `temp`, `test` (no prefix)
- `development`, `staging`, `production`

## Error Handling

If namespace validation fails:
1. Inform user of the policy violation
2. Suggest correct namespace or prefix
3. Do not proceed with write operation

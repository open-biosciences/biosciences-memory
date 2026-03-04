# Namespace Anti-Patterns

## Version Namespace Naming

**Severity:** High
**Detection:** Namespace ends with `_v1`, `_v2`, `_v3`, etc.

### The Problem

```
# WRONG: Creates separate namespaces for each version
don_branson_resume_v1
don_branson_resume_v2
don_branson_resume_v3
```

### Why It's Wrong

1. **Graphiti already handles versioning** - Facts have `valid_at` and `invalid_at` timestamps
2. **Capacity waste** - Each version duplicates entities (~700+ per resume)
3. **Lost temporal relationships** - Version history is disconnected
4. **Namespace proliferation** - 22 namespaces found when 10 expected

### The Correct Pattern

```
# RIGHT: Single namespace with temporal evolution
don_branson_resume
```

When you add updated information to the same namespace:
- Old facts get `invalid_at` timestamp
- New facts get current `valid_at` timestamp
- History preserved automatically

### Remediation

1. Export versioned namespace:
   ```bash
   uv run scripts/export_graph.py --group-id my_data_v3 --output backup.json
   ```

2. Create unversioned namespace and import
3. Archive versioned namespace (90-day retention)
4. Delete from production after verification

### Related Issues

- AGE-63: Document Version Namespace Anti-Pattern
- AGE-64: Consolidate Resume Namespaces

---

## Generic Namespace Names

**Severity:** High
**Detection:** Matches prohibited list

### Prohibited Names

| Name | Problem | Use Instead |
|------|---------|-------------|
| `main` | Ambiguous | Domain-specific name |
| `default` | Conflicts | Specific purpose |
| `temp` | Unclear lifecycle | `scratch_*` |
| `test` | No prefix | `test_*` |
| `development` | Too generic | `dev_*` |
| `staging` | No staging env | `dev_*` |
| `production` | Redundant | Domain name |

### Evidence

89 episodes were polluted into unclear namespaces by Claude Desktop, requiring manual cleanup.

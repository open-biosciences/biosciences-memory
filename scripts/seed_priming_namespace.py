"""Seed the biosciences-graphiti instance with the priming namespace.

Reads priming episodes from graphiti-docker (shared dev Neo4j via SSE)
and writes them to biosciences-graphiti (repo-local via MCP SDK).

Usage:
    uv run python scripts/seed_priming_namespace.py
    uv run python scripts/seed_priming_namespace.py --dry-run
"""

import argparse
import asyncio
import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SOURCE_CYPHER_URL = "http://localhost:8005/mcp/"  # neo4j-cypher-local (SSE)
TARGET_GRAPHITI_URL = "http://localhost:8007/mcp"  # biosciences-graphiti (Streamable HTTP)
GROUP_ID = "open-biosciences-migration-2026-priming"


def fetch_priming_episodes() -> list[dict]:
    """Fetch priming episodes from Docker Neo4j via the Cypher MCP (SSE)."""
    query = """
    MATCH (e:Episodic)
    WHERE e.group_id = $group_id
    RETURN e.name AS name,
           e.content AS content,
           e.source AS source,
           e.source_description AS source_description
    ORDER BY e.created_at
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "read_neo4j_cypher",
            "arguments": {"query": query, "params": {"group_id": GROUP_ID}},
        },
    }
    resp = httpx.post(
        SOURCE_CYPHER_URL,
        json=payload,
        headers={"Accept": "application/json, text/event-stream"},
        timeout=30,
    )
    resp.raise_for_status()

    # Parse SSE response — look for "event: message" data line
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            content = data.get("result", {}).get("content", [])
            for block in content:
                if block.get("type") == "text":
                    return json.loads(block["text"])
    return []


async def write_episodes(episodes: list[dict]) -> None:
    """Write episodes to biosciences-graphiti via MCP Streamable HTTP."""
    async with streamable_http_client(TARGET_GRAPHITI_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for i, ep in enumerate(episodes, 1):
                name = ep.get("name", "unnamed")
                print(f"  [{i}/{len(episodes)}] {name}...", end=" ", flush=True)
                result = await session.call_tool(
                    "add_memory",
                    arguments={
                        "name": name,
                        "episode_body": ep.get("content", ""),
                        "group_id": GROUP_ID,
                        "source": ep.get("source", "text"),
                        "source_description": ep.get("source_description", ""),
                    },
                )
                if result.isError:
                    print(f"FAILED: {result.content}")
                else:
                    print("OK")


async def check_target_health() -> bool:
    """Verify biosciences-graphiti is reachable."""
    try:
        async with streamable_http_client(TARGET_GRAPHITI_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_status", arguments={})
                return not result.isError
    except Exception as e:
        print(f"  Connection error: {e}")
        return False


async def async_main(dry_run: bool) -> None:
    print(f"Source: {SOURCE_CYPHER_URL} (Docker Neo4j Cypher)")
    print(f"Target: {TARGET_GRAPHITI_URL} (biosciences-graphiti)")
    print(f"Group:  {GROUP_ID}")
    print()

    # Check target
    print("Checking target...", end=" ", flush=True)
    if not await check_target_health():
        print("\nERROR: Cannot reach biosciences-graphiti.")
        print("  Start it with: docker compose up -d")
        raise SystemExit(1)
    print("OK")

    # Fetch source episodes
    print("Fetching priming episodes from source...", end=" ", flush=True)
    episodes = fetch_priming_episodes()
    if not episodes:
        print("\nERROR: No priming episodes found.")
        print(f"  Expected group_id: {GROUP_ID}")
        raise SystemExit(1)
    print(f"found {len(episodes)}\n")

    for i, ep in enumerate(episodes, 1):
        name = ep.get("name", "unnamed")
        content_len = len(ep.get("content", ""))
        print(f"  {i}. {name} ({content_len:,} chars)")

    if dry_run:
        print("\n--dry-run: No data written.")
        return

    # Write to target
    print(f"\nWriting {len(episodes)} episodes to biosciences-graphiti...")
    await write_episodes(episodes)
    print(f"\nDone. {len(episodes)} episodes written to {GROUP_ID}.")


def main():
    parser = argparse.ArgumentParser(description="Seed biosciences-graphiti priming namespace")
    parser.add_argument("--dry-run", action="store_true", help="Show episodes without writing")
    args = parser.parse_args()
    asyncio.run(async_main(args.dry_run))


if __name__ == "__main__":
    main()

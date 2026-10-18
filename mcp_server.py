#!/usr/bin/env python3
"""MCP server that exposes the fraud-graph Cypher tools over stdio using FastMCP.

Read tools are tagged {"read"} and write tools are tagged {"write"}, so a
client can mount a read-only view (e.g. a general discovery agent) while
reserving record_case for privileged workflows.

Usage (stdio transport):
    ./.venv/bin/python mcp_server.py

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "fraud-graph": {
          "command": "/abs/path/to/.venv/bin/python",
          "args": ["/abs/path/to/mcp_server.py"],
          "env": {
            "FALKORDB_HOST": "localhost",
            "FALKORDB_PORT": "6379"
          }
        }
      }
    }
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402 — must come after sys.path fix

from mcp.server.fastmcp import FastMCP

# Two-layer write authorization:
#   MCP_WRITE_ENABLED=1  — operational kill-switch; default off (read-only service)
#   MCP_WRITE_API_KEY    — token callers must supply as caller_api_key to record_case.
#                          If unset, any caller that passes the kill-switch check may write.
#                          Set it to distinguish privileged investigators from read-only users.
_WRITE_ENABLED = os.getenv("MCP_WRITE_ENABLED", "0") == "1"
_WRITE_API_KEY = os.getenv("MCP_WRITE_API_KEY", "")

_VALID_EVIDENCE_LABELS = {"Account", "Device", "IP"}

mcp = FastMCP("fraud-graph")


# ---- Read-only tools ----

@mcp.tool(tags={"read"})
def get_account(account_id: str) -> list:
    """Look up an account by id."""
    return agent.get_account(account_id)


@mcp.tool(tags={"read"})
def shared_device_accounts(account_id: str) -> list:
    """Accounts that share at least one device fingerprint with this account."""
    return agent.shared_device_accounts(account_id)


@mcp.tool(tags={"read"})
def shared_ip_accounts(account_id: str) -> list:
    """Accounts that have logged in from the same IP address as this account."""
    return agent.shared_ip_accounts(account_id)


@mcp.tool(tags={"read"})
def outgoing_flows(account_id: str, max_hops: int = 3) -> list:
    """Money that moved out of this account, following transfers up to max_hops (1-4)."""
    return agent.outgoing_flows(account_id, max_hops)


@mcp.tool(tags={"read"})
def money_cycles(account_id: str) -> list:
    """Transfer loops that start and end at this account (round-tripping money)."""
    return agent.money_cycles(account_id)


@mcp.tool(tags={"read"})
def connection_path(account_a: str, account_b: str) -> list:
    """Shortest chain of money transfers from account_a to account_b (follows transfer direction)."""
    return agent.connection_path(account_a, account_b)


@mcp.tool(tags={"read"})
def top_risk_accounts() -> list:
    """Run FalkorDB's native PageRank algorithm over the transfer graph.

    Returns the 10 accounts most structurally central to money flows.
    High PageRank indicates an account is a hub that receives transfers from
    many other high-ranked accounts — a structural signature of layering rings.
    """
    return agent.top_risk_accounts()


@mcp.tool(tags={"read"})
def risk_score(account_id: str) -> dict:
    """Compute a weighted fraud-risk score (0-100) from graph signals.

    Returns score, verdict_hint (LOW/MEDIUM/HIGH), per-signal breakdown, and
    scoring rules so the verdict can be fully explained.
    """
    return agent.risk_score(account_id)


@mcp.tool(tags={"read"})
def prior_cases(account_id: str) -> list:
    """Earlier investigation cases recorded against this account."""
    return agent.prior_cases(account_id)


@mcp.tool(tags={"read"})
def incoming_flows(account_id: str, max_hops: int = 3) -> list:
    """Money that moved into this account, following transfers up to max_hops (1-4)."""
    return agent.incoming_flows(account_id, max_hops)


@mcp.tool(tags={"read"})
def propagation_risk(account_id: str, n_simulations: int = 40) -> dict:
    """Monte Carlo estimate of fraud-spread probability to each transfer target.

    For each account that received money from account_id, runs n_simulations
    (clamped 10-100) Bernoulli trials combining:
      - transfer frequency (how many transfers link the pair)
      - shared device (same hardware fingerprint)
      - shared IP (same login IP)
    Returns infection_probability per neighbor, sorted highest-first.
    """
    return agent.propagation_risk(account_id, n_simulations)


@mcp.tool(tags={"read"})
def max_flow_risk(source_id: str, sink_id: str) -> dict:
    """Edmonds-Karp maximum money flow from source_id to sink_id.

    Uses transfer amounts as edge capacities over the full transfer subgraph
    (up to 500 edges). Returns the theoretical upper bound on funds that can
    move between the two accounts, the number of augmenting paths found, and
    the edge count analyzed.
    """
    return agent.max_flow_risk(source_id, sink_id)


# ---- Write tool ----

@mcp.tool(tags={"write"})
def record_case(account_id: str, verdict: str, summary: str,
                evidence_ids: list[str], override_reason: str = "",
                caller_api_key: str = "") -> dict:
    """Save the verdict to the graph as a Case linked to the account and evidence ids.

    verdict must be LOW, MEDIUM, or HIGH.
    evidence_ids must be Account, Device, or IP node ids only.
    override_reason is required when verdict differs from risk_score's verdict_hint.
    caller_api_key must match MCP_WRITE_API_KEY when that variable is set.

    Idempotent within a one-minute window: a second call for the same account
    returns the existing case_id rather than creating a duplicate.

    Requires MCP_WRITE_ENABLED=1 in the server environment.
    """
    if not _WRITE_ENABLED:
        return {
            "error": "write operations disabled — set MCP_WRITE_ENABLED=1 to allow record_case"
        }
    # Identity check: if an API key is configured, the caller must present it.
    if _WRITE_API_KEY and caller_api_key != _WRITE_API_KEY:
        return {
            "error": "unauthorized — valid caller_api_key required for write operations"
        }
    # Server-side evidence schema validation: only Account/Device/IP node types accepted.
    for eid in evidence_ids:
        if not isinstance(eid, str) or not eid.strip():
            return {"error": f"invalid evidence id: {eid!r}"}
    # Label check delegated to agent.record_case which queries the live graph.
    return agent.record_case(account_id, verdict, summary, evidence_ids, override_reason)


if __name__ == "__main__":
    mcp.run()

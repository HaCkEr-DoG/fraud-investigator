"""Fraud-investigation agent. A local model (via Ollama) reasons over a FalkorDB graph through Cypher-backed tools.

Usage:  python agent.py ACC-RING-01
Needs:  Ollama running locally with a tool-calling model, a running FalkorDB, and seed.py run once.
No API key is required.
"""
from __future__ import annotations

import collections
import copy
import json
import os
import random as _random
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

from falkordb import FalkorDB

SCORING_POLICY_VERSION = "2.0"  # bump whenever weights or thresholds change
_MAX_CYCLE_DEPTH = 8            # default max hops for money_cycles (clamped 3–10)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GRAPH = FalkorDB(
    host=os.getenv("FALKORDB_HOST", "localhost"),
    port=int(os.getenv("FALKORDB_PORT", "6379")),
).select_graph(os.getenv("FALKOR_GRAPH", "fraud"))

_VALID_EVIDENCE_LABELS = {"Account", "Device", "IP"}
_CASE_CONSTRAINT_READY = False
_MIN_EVIDENCE = 1
# Row-limit constants for traversal queries.  When len(results) == limit the
# traversal was likely cut short; risk_score exposes traversal_truncated=True.
_TRAVERSAL_LIMITS = {"shared_device": 20, "shared_ip": 20, "outgoing_flows": 25, "cycles": 10}


def run_read(query: str, params: dict | None = None, timeout: int | None = 5000) -> list[dict]:
    """Read-only Cypher — uses ro_query() so the engine rejects any mutating clause."""
    res = GRAPH.ro_query(query, params or {}, timeout=timeout)
    cols = [h[1] for h in res.header]
    return [dict(zip(cols, row)) for row in res.result_set]


def run_write(query: str, params: dict | None = None, timeout: int | None = 5000) -> list[dict]:
    """Mutable Cypher — only record_case and seed helpers call this."""
    res = GRAPH.query(query, params or {}, timeout=timeout)
    cols = [h[1] for h in res.header]
    return [dict(zip(cols, row)) for row in res.result_set]


def ensure_case_uniqueness_constraint(wait_seconds: float = 10.0) -> bool:
    """Ensure Case.session_key has an active unique constraint before accepting writes.

    FalkorDB creates constraints asynchronously. Fail closed if the constraint cannot
    be confirmed; MERGE alone is not treated as a substitute for uniqueness enforcement.
    """
    global _CASE_CONSTRAINT_READY
    if _CASE_CONSTRAINT_READY:
        return True

    def is_target_constraint(item: dict) -> bool:
        label = item.get("label", item.get("entity_label", ""))
        properties = item.get("properties", [])
        ctype = str(item.get("constraint_type", item.get("constraintType", item.get("type", "")))).lower()
        entity_type = str(item.get("entity_type", item.get("entityType", "node"))).lower()
        return (
            label == "Case"
            and "session_key" in properties
            and "unique" in ctype
            and entity_type in ("node", "nodes", "entitytype.node")
        )

    constraints = GRAPH.list_constraints()
    target = next((item for item in constraints if is_target_constraint(item)), None)
    if target is None:
        GRAPH.create_node_unique_constraint("Case", "session_key")

    deadline = time.monotonic() + wait_seconds
    last_status = "not listed"
    while time.monotonic() <= deadline:
        constraints = GRAPH.list_constraints()
        target = next((item for item in constraints if is_target_constraint(item)), None)
        if target:
            last_status = str(target.get("status", "unknown")).lower()
            if any(word in last_status for word in ("active", "enabled", "operational", "ready", "complete")):
                _CASE_CONSTRAINT_READY = True
                return True
            if any(word in last_status for word in ("failed", "error", "rejected")):
                break
        time.sleep(0.1)
    raise RuntimeError(f"Unique Case.session_key constraint is not active (status: {last_status})")


# ---- Tools the agent can call. Each one is a single Cypher query. ----

def get_account(account_id: str) -> list[dict]:
    return run_read(
        "MATCH (a:Account {id: $id}) RETURN a.id AS id, a.name AS name, a.created_at AS created_at",
        {"id": account_id},
    )


def shared_device_accounts(account_id: str) -> list[dict]:
    # LIMIT 20 is a deliberate super-node guard: high-degree device nodes can fan out exponentially.
    return run_read(
        """
        MATCH (a:Account {id: $id})-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(o:Account)
        WHERE o.id <> $id
        RETURN o.id AS account, collect(DISTINCT d.id) AS shared_devices
        ORDER BY size(shared_devices) DESC LIMIT 20
        """,
        {"id": account_id},
    )


def shared_ip_accounts(account_id: str) -> list[dict]:
    # LIMIT 20: same super-node guard as shared_device_accounts.
    return run_read(
        """
        MATCH (a:Account {id: $id})-[:LOGGED_FROM]->(i:IP)<-[:LOGGED_FROM]-(o:Account)
        WHERE o.id <> $id
        RETURN o.id AS account, collect(DISTINCT i.id) AS shared_ips
        ORDER BY size(shared_ips) DESC LIMIT 20
        """,
        {"id": account_id},
    )


def outgoing_flows(account_id: str, max_hops: int = 3) -> list[dict]:
    # hops is interpolated, so cap it to a bounded int before string-formatting.
    # LIMIT 25 caps the working-set matrix at ~25 path rows regardless of graph density.
    hops = min(max(int(max_hops), 1), 4)
    return run_read(
        f"""
        MATCH p = (a:Account {{id: $id}})-[:TRANSFERRED*1..{hops}]->(b:Account)
        WHERE b.id <> $id
        RETURN b.id AS account, length(p) AS hops,
               [n IN nodes(p) | n.id] AS path,
               [r IN relationships(p) | r.amount] AS amounts,
               [r IN relationships(p) | r.txn_id] AS txn_ids,
               [r IN relationships(p) | r.ts] AS timestamps
        LIMIT 25
        """,
        {"id": account_id},
    )


def incoming_flows(account_id: str, max_hops: int = 1) -> list[dict]:
    """Return transfer paths flowing into the target account, bounded to four hops."""
    hops = min(max(int(max_hops), 1), 4)
    return run_read(
        f"""
        MATCH p = (source:Account)-[:TRANSFERRED*1..{hops}]->(target:Account {{id: $id}})
        WHERE source.id <> $id
        RETURN source.id AS account, length(p) AS hops,
               [n IN nodes(p) | n.id] AS path,
               [r IN relationships(p) | r.amount] AS amounts,
               [r IN relationships(p) | r.txn_id] AS txn_ids,
               [r IN relationships(p) | r.ts] AS timestamps
        LIMIT 25
        """,
        {"id": account_id},
    )


def propagation_risk(account_id: str, n_simulations: int = 40) -> dict:
    """Monte Carlo simulation of fraud propagation probability to each direct neighbor.

    For each outgoing transfer target, estimates the probability it would be drawn into
    the same fraud scheme based on transfer frequency, shared devices, and shared IPs.
    Inspired by the WannaCry infection-spread demo from the FalkorDB cybersecurity workshop.

    n_simulations: number of Monte Carlo trials per neighbor (10–100, default 40).
    """
    n_sims = min(max(int(n_simulations), 10), 100)

    transfer_rows = run_read(
        """
        MATCH (a:Account {id: $id})-[r:TRANSFERRED]->(b:Account)
        WHERE b.id <> $id
        WITH b.id AS neighbor, count(r) AS transfer_count, sum(r.amount) AS total_volume
        RETURN neighbor, transfer_count, total_volume
        ORDER BY transfer_count DESC
        LIMIT 20
        """,
        {"id": account_id},
    )
    if not transfer_rows:
        return {"source": account_id, "neighbors": [], "simulations": n_sims}

    neighbor_ids = [r["neighbor"] for r in transfer_rows]

    device_rows = run_read(
        """
        MATCH (a:Account {id: $id})-[:USED_DEVICE]->(d)<-[:USED_DEVICE]-(b:Account)
        WHERE b.id IN $neighbors
        RETURN b.id AS neighbor, count(DISTINCT d) AS shared_devices
        """,
        {"id": account_id, "neighbors": neighbor_ids},
    )
    ip_rows = run_read(
        """
        MATCH (a:Account {id: $id})-[:LOGGED_FROM]->(i)<-[:LOGGED_FROM]-(b:Account)
        WHERE b.id IN $neighbors
        RETURN b.id AS neighbor, count(DISTINCT i) AS shared_ips
        """,
        {"id": account_id, "neighbors": neighbor_ids},
    )

    device_map = {r["neighbor"]: int(r.get("shared_devices") or 0) for r in device_rows}
    ip_map = {r["neighbor"]: int(r.get("shared_ips") or 0) for r in ip_rows}

    rng = _random.Random()
    results = []
    for row in transfer_rows:
        nbr = row["neighbor"]
        tc = max(int(row.get("transfer_count") or 1), 1)
        tv = float(row.get("total_volume") or 0)
        has_device = device_map.get(nbr, 0) > 0
        has_ip = ip_map.get(nbr, 0) > 0

        # Each additional transfer adds 10 pp base probability (capped at 0.70).
        p_transfer = min(0.20 + (tc - 1) * 0.10, 0.70)
        p_device = 0.35 if has_device else 0.0   # shared device is a strong ring signal
        p_ip = 0.15 if has_ip else 0.0
        # Combine independent channels: P(any channel propagates).
        p_base = round(min(1.0 - (1.0 - p_transfer) * (1.0 - p_device) * (1.0 - p_ip), 0.98), 4)

        infected = sum(1 for _ in range(n_sims) if rng.random() < p_base)
        results.append({
            "neighbor": nbr,
            "infection_probability": round(infected / n_sims, 4),
            "simulations": n_sims,
            "factors": {
                "transfer_count": tc,
                "total_volume": tv,
                "shared_device": has_device,
                "shared_ip": has_ip,
                "base_probability": p_base,
            },
        })

    results.sort(key=lambda x: x["infection_probability"], reverse=True)
    return {"source": account_id, "neighbors": results, "simulations": n_sims}


def max_flow_risk(source_id: str, sink_id: str) -> dict:
    """Maximum-flow analysis of the money transfer network (Edmonds-Karp algorithm).

    Treats each transfer edge's amount as its capacity and computes the theoretical upper
    bound on total funds that could flow from source_id to sink_id through the network.
    Inspired by the data-exfiltration max-flow demo from the FalkorDB cybersecurity workshop.

    Returns max_flow (total currency units), augmenting_paths found, and edges_analyzed.
    """
    edges_raw = run_read(
        """
        MATCH (a:Account)-[r:TRANSFERRED]->(b:Account)
        RETURN a.id AS src, b.id AS dst, coalesce(r.amount, 0) AS capacity
        LIMIT 500
        """,
    )
    if not edges_raw:
        return {"source": source_id, "sink": sink_id, "max_flow": 0.0,
                "edges_analyzed": 0, "augmenting_paths": 0, "method": "edmonds_karp"}

    # Build residual-capacity map and adjacency dict for O(1) neighbor lookup.
    cap: dict[tuple[str, str], float] = {}
    adj: dict[str, set[str]] = {}
    for e in edges_raw:
        u, v, c = e["src"], e["dst"], float(e.get("capacity") or 0)
        cap[(u, v)] = cap.get((u, v), 0.0) + c
        if (v, u) not in cap:
            cap[(v, u)] = 0.0
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)  # reverse edges needed for residual graph

    def _bfs() -> dict[str, str | None] | None:
        parent: dict[str, str | None] = {source_id: None}
        q: collections.deque[str] = collections.deque([source_id])
        while q:
            u = q.popleft()
            for v in adj.get(u, set()):
                if v not in parent and cap.get((u, v), 0.0) > 0:
                    parent[v] = u
                    if v == sink_id:
                        return parent
                    q.append(v)
        return None

    max_flow_val = 0.0
    iterations = 0
    max_iter = 1000  # safety cap so dense graphs can't loop indefinitely
    while iterations < max_iter:
        parent = _bfs()
        if parent is None:
            break
        # Find bottleneck capacity along this augmenting path.
        path_flow = float("inf")
        v = sink_id
        while v != source_id:
            u = parent[v]
            path_flow = min(path_flow, cap[(u, v)])
            v = u
        # Update residual capacities (forward − bottleneck, reverse + bottleneck).
        v = sink_id
        while v != source_id:
            u = parent[v]
            cap[(u, v)] -= path_flow
            cap[(v, u)] = cap.get((v, u), 0.0) + path_flow
            v = u
        max_flow_val += path_flow
        iterations += 1

    return {
        "source": source_id,
        "sink": sink_id,
        "max_flow": round(max_flow_val, 2),
        "edges_analyzed": len(edges_raw),
        "augmenting_paths": iterations,
        "method": "edmonds_karp",
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def money_cycles(account_id: str, max_depth: int | None = None) -> list[dict]:
    """Find structural cycles and mark only monotonically timed ones as confirmed.

    max_depth controls cycle length (3–10 hops, default _MAX_CYCLE_DEPTH=8).
    Increase to find larger rings that the default depth would miss.
    """
    # depth is interpolated into Cypher, so clamp it to [3, 10] before string-formatting.
    depth = min(max(int(max_depth or _MAX_CYCLE_DEPTH), 3), 10)
    rows = run_read(
        f"""
        MATCH p = (a:Account {{id: $id}})-[:TRANSFERRED*2..{depth}]->(a)
        RETURN [n IN nodes(p) | n.id] AS cycle,
               [r IN relationships(p) | r.amount] AS amounts,
               [r IN relationships(p) | r.txn_id] AS txn_ids,
               [r IN relationships(p) | r.ts] AS timestamps
        LIMIT 10
        """,
        {"id": account_id},
    )
    results = []
    for row in rows:
        ts_raw = row.get("timestamps") or []
        parsed = [_parse_timestamp(v) for v in ts_raw]
        chronological = bool(parsed) and all(v is not None for v in parsed)
        if chronological:
            chronological = all(parsed[i] <= parsed[i + 1] for i in range(len(parsed) - 1))
        results.append({
            "cycle": row["cycle"],
            "amounts": row["amounts"],
            "txn_ids": row.get("txn_ids"),
            "timestamps": ts_raw,
            "cycle_type": "confirmed_chronological" if chronological else "structural_only",
        })
    return results


def connection_path(account_a: str, account_b: str) -> list[dict]:
    return run_read(
        """
        MATCH (a:Account {id: $a}), (b:Account {id: $b})
        WITH shortestPath((a)-[:TRANSFERRED*..8]->(b)) AS p
        RETURN [n IN nodes(p) | coalesce(n.id, n.name)] AS nodes,
               [r IN relationships(p) | type(r)] AS relationships
        """,
        {"a": account_a, "b": account_b},
    )


def top_risk_accounts(anchor_id: str | None = None) -> list[dict]:
    """Run FalkorDB's native PageRank algorithm over the TRANSFERRED graph.

    PageRank score reflects how many high-value counterparties send money to an account.
    A high-score account that receives many transfers from other high-score accounts is
    structurally central to the transfer network — a key signal for layering rings.

    anchor_id: when provided, results are filtered to the 2-hop TRANSFERRED neighborhood
               of that account (ranked by global PageRank). This gives a subgraph-scoped
               view so the rank reflects position among connected peers, not the whole graph.
    """
    if anchor_id:
        # Find the 2-hop neighborhood (cheap traversal; LIMIT 200 bounds the subgraph).
        nbr_rows = run_read(
            """
            MATCH (a:Account {id: $id})-[:TRANSFERRED*1..2]-(n:Account)
            RETURN DISTINCT n.id AS nid
            LIMIT 200
            """,
            {"id": anchor_id},
        )
        nbr_ids = [r["nid"] for r in nbr_rows] + [anchor_id]
        # Run global PageRank then filter results to neighborhood members only.
        return run_read(
            """
            CALL algo.pageRank('Account', 'TRANSFERRED')
            YIELD node, score
            WITH node, score
            WHERE node.id IN $ids
            RETURN node.id AS account, score
            ORDER BY score DESC
            LIMIT 10
            """,
            {"ids": nbr_ids},
        )
    return run_read(
        """
        CALL algo.pageRank('Account', 'TRANSFERRED')
        YIELD node, score
        RETURN node.id AS account, score
        ORDER BY score DESC
        LIMIT 10
        """
    )


def risk_score(account_id: str) -> dict:
    """Compute a weighted fraud-risk score from multiple graph signals.

    Each signal maps to a score component so the model can cite the exact
    rule that drove the verdict (satisfies explainability requirement).

    Signals:
      shared_device  — capped at 3 peers (42 pts max) to prevent saturation
      shared_ip      — capped at 2 peers (10 pts max)
      cycle          — 18 pts each (confirmed_chronological only)
      outgoing_path  — 2 pts each, capped at 5 paths (10 pts max)
      prior_case     — 12 pts each
      centrality     — native PageRank: 20 pts if top-3, 10 pts if top-10 hub
    """
    devices = shared_device_accounts(account_id)
    ips = shared_ip_accounts(account_id)
    cycles = money_cycles(account_id)
    flows = outgoing_flows(account_id, 4)
    cases = prior_cases(account_id)

    confirmed_cycles = [c for c in cycles if c.get("cycle_type") == "confirmed_chronological"]
    structural_cycles = [c for c in cycles if c.get("cycle_type") != "confirmed_chronological"]

    device_pts = min(len(devices), 3) * 14
    ip_pts = min(len(ips), 2) * 5
    cycle_pts = len(confirmed_cycles) * 18
    flow_pts = min(len(flows), 5) * 2
    prior_pts = len(cases) * 12

    # Detect traversal truncation: if any query returned exactly its hard LIMIT the
    # graph may have more connections than the score reflects.
    traversal_truncated = (
        len(devices) >= _TRAVERSAL_LIMITS["shared_device"]
        or len(ips) >= _TRAVERSAL_LIMITS["shared_ip"]
        or len(cycles) >= _TRAVERSAL_LIMITS["cycles"]
        or len(flows) >= _TRAVERSAL_LIMITS["outgoing_flows"]
    )

    # Native FalkorDB PageRank centrality signal.
    centrality_pts = 0
    centrality_rank = None
    centrality_score = None
    centrality_error = None
    score_status = "complete"
    investigation_policy: dict = {}
    try:
        # Neighborhood-scoped ranking: ranks anchor among its connected peers.
        ranked = top_risk_accounts(anchor_id=account_id)
        id_list = [r["account"] for r in ranked]
        score_map = {r["account"]: r["score"] for r in ranked}
        if account_id in id_list:
            rank = id_list.index(account_id) + 1  # 1-based
            centrality_rank = rank
            centrality_score = float(score_map[account_id]) if score_map[account_id] is not None else None
            centrality_pts = 20 if rank <= 3 else 10
            if rank <= 3:
                investigation_policy = {
                    "recommended_next": "inspect_incoming_counterparties",
                    "reason": f"top-{rank} PageRank hub — identify who funds this account",
                }
            else:
                investigation_policy = {
                    "recommended_next": "inspect_shared_device_cluster",
                    "reason": f"rank {rank} — elevated centrality; focus on device/IP ring membership",
                }
        else:
            investigation_policy = {
                "recommended_next": "inspect_shared_device_cluster",
                "reason": "not in top-10 PageRank — structural links more likely via device/IP",
            }
    except Exception as exc:
        centrality_error = str(exc)
        score_status = "degraded"
        investigation_policy = {
            "recommended_next": "proceed_without_centrality",
            "reason": "PageRank unavailable — score excludes centrality; treat verdict conservatively",
        }

    total = min(
        device_pts + ip_pts + cycle_pts + flow_pts + prior_pts + centrality_pts,
        100,
    )
    verdict_hint = "HIGH" if total >= 50 else "MEDIUM" if total >= 20 else "LOW"

    result: dict = {
        "score": total,
        "score_status": score_status,
        "verdict_hint": verdict_hint,
        "traversal_truncated": traversal_truncated,
        "investigation_policy": investigation_policy,
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "breakdown": {
            "shared_device_peers": len(devices),
            "shared_device_pts": device_pts,
            "shared_ip_peers": len(ips),
            "shared_ip_pts": ip_pts,
            "money_cycles_total": len(cycles),
            "money_cycles_confirmed": len(confirmed_cycles),
            "structural_only_cycles": len(structural_cycles),
            "cycle_pts": cycle_pts,
            "confirmed_cycle_txn_ids": [c["txn_ids"] for c in confirmed_cycles if c.get("txn_ids")],
            "outgoing_paths": len(flows),
            "flow_pts": flow_pts,
            "flow_paths": [
                {"path": f["path"], "hops": f["hops"], "amounts": f["amounts"]}
                for f in flows[:5]
            ],
            "prior_cases": len(cases),
            "prior_pts": prior_pts,
            "pagerank_rank": centrality_rank,
            "pagerank_score": centrality_score,
            "centrality_pts": centrality_pts,
        },
        "evidence_details": {
            "shared_device_evidence": [
                {"account": r.get("account"), "shared_devices": r.get("shared_devices", [])}
                for r in devices
            ],
            "shared_ip_evidence": [
                {"account": r.get("account"), "shared_ips": r.get("shared_ips", [])}
                for r in ips
            ],
            "confirmed_cycles": [
                {k: c.get(k) for k in ("cycle", "txn_ids", "timestamps", "amounts")}
                for c in confirmed_cycles
            ],
            "structural_only_cycles": [
                {k: c.get(k) for k in ("cycle", "txn_ids", "timestamps", "amounts")}
                for c in structural_cycles
            ],
            "outgoing_paths": [
                {k: r.get(k) for k in ("account", "hops", "path", "txn_ids", "timestamps", "amounts")}
                for r in flows
            ],
            "prior_case_ids": [r.get("case_id") for r in cases if r.get("case_id") is not None],
            "pagerank": {"rank": centrality_rank, "score": centrality_score, "error": centrality_error},
        },
        "scoring_rules": {
            "shared_device": "14 pts each (cap 3 peers = 42 pts max)",
            "shared_ip": "5 pts each (cap 2 peers = 10 pts max)",
            "cycle": "18 pts each (confirmed_chronological only)",
            "outgoing_path": "2 pts each (cap 5 = 10 pts max)",
            "prior_case": "12 pts each",
            "centrality_top3": "20 pts (PageRank rank 1-3)",
            "centrality_top10": "10 pts (PageRank rank 4-10)",
            "max": 100,
            "thresholds": {"HIGH": ">=50", "MEDIUM": ">=20", "LOW": "<20"},
        },
    }
    if centrality_error:
        result["centrality_error"] = centrality_error
    return result


def prior_cases(account_id: str) -> list[dict]:
    return run_read(
        """
        MATCH (c:Case)-[:INVESTIGATES]->(a:Account {id: $id})
        RETURN c.id AS case_id, c.verdict AS verdict, c.summary AS summary, c.created_at AS created_at
        ORDER BY c.created_at DESC
        """,
        {"id": account_id},
    )


def _validate_evidence_ids(evidence_ids: list[str]) -> tuple[list[str] | None, dict | None]:
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return None, {"error": "evidence_ids must be a non-empty list of Account, Device, or IP ids"}
    if any(not isinstance(v, str) or not v.strip() for v in evidence_ids):
        return None, {"error": "every evidence id must be a non-empty string"}
    ids = list(dict.fromkeys(v.strip() for v in evidence_ids))
    rows = run_read(
        "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, labels(n) AS labels",
        {"ids": ids},
    )
    found: dict[str, list[list[str]]] = {}
    for row in rows:
        found.setdefault(row.get("id"), []).append(row.get("labels") or [])
    for eid in ids:
        matches = found.get(eid, [])
        if not matches:
            return None, {"error": f"evidence id {eid!r} does not exist"}
        if len(matches) != 1:
            return None, {"error": f"evidence id {eid!r} is ambiguous across multiple nodes"}
        if not _VALID_EVIDENCE_LABELS.intersection(matches[0]):
            return None, {"error": f"evidence id {eid!r} has unsupported labels {matches[0]}"}
    return ids, None


def record_case(
    account_id: str,
    verdict: str,
    summary: str,
    evidence_ids: list[str],
    override_reason: str | None = None,
    investigation_id: str | None = None,
) -> dict:
    """Validate decision/evidence and persist a Case node and citations in one Cypher query."""
    if verdict not in ("LOW", "MEDIUM", "HIGH"):
        return {"error": "verdict must be LOW, MEDIUM or HIGH"}
    if not get_account(account_id):
        return {"error": f"unknown account {account_id}"}
    if not isinstance(investigation_id, str) or not investigation_id.strip():
        return {"error": "investigation_id must be a non-empty string"}
    session_key = investigation_id.strip()

    # Fast retry path: return the existing case without re-scoring.
    existing = run_read(
        """
        MATCH (c:Case {session_key: $sk})-[:INVESTIGATES]->(a:Account)
        RETURN c.id AS case_id, coalesce(c.account_id, a.id) AS account_id LIMIT 1
        """,
        {"sk": session_key},
    )
    if existing:
        if existing[0].get("account_id") != account_id:
            return {"error": "investigation_id is already associated with a different account"}
        existing_id = existing[0]["case_id"]
        linked = run_read(
            "MATCH (c:Case {id: $cid})-[:CITES]->(n) RETURN count(DISTINCT n) AS cnt",
            {"cid": existing_id},
        )
        return {
            "case_id": existing_id,
            "cited_evidence": linked[0]["cnt"] if linked else 0,
            "investigation_id": session_key,
            "deduplicated": True,
            "note": "existing case returned for this investigation_id",
        }

    ids, validation_error = _validate_evidence_ids(evidence_ids)
    if validation_error:
        return validation_error
    assert ids is not None

    computed = risk_score(account_id)
    computed_hint = computed["verdict_hint"]
    mismatch = verdict != computed_hint
    override_is_valid = isinstance(override_reason, str) and bool(override_reason.strip())

    if mismatch and not override_is_valid:
        return {
            "error": (
                f"verdict {verdict!r} differs from computed hint {computed_hint!r}. "
                "Provide override_reason to confirm this is intentional."
            ),
            "computed_score": computed["score"],
            "computed_hint": computed_hint,
        }
    if computed.get("score_status") == "degraded" and not override_is_valid:
        return {
            "error": "risk score is degraded; an explicit override_reason is required to record this case",
            "score_status": "degraded",
            "centrality_error": computed.get("centrality_error"),
        }

    try:
        ensure_case_uniqueness_constraint()
    except Exception as exc:
        return {"error": f"case uniqueness constraint unavailable; refusing write: {type(exc).__name__}: {exc}"}

    now = datetime.now(timezone.utc)
    case_id = f"CASE-{now:%Y%m%d%H%M%S%f}"
    verdict_vs_computed = "match" if not mismatch else f"override:{computed_hint}:{override_reason}"

    breakdown_json = json.dumps(computed["breakdown"], sort_keys=True, separators=(",", ":"))
    scoring_rules_json = json.dumps(computed["scoring_rules"], sort_keys=True, separators=(",", ":"))
    evidence_details_json = json.dumps(computed["evidence_details"], sort_keys=True, separators=(",", ":"))
    investigation_policy_json = json.dumps(computed["investigation_policy"], sort_keys=True, separators=(",", ":"))

    cycle_txns = computed["breakdown"].get("confirmed_cycle_txn_ids", [])
    confirmed_cycle_txns_json = json.dumps(cycle_txns)
    flow_paths_json = json.dumps(computed["breakdown"].get("flow_paths", []))

    result = run_write(
        """
        MATCH (a:Account {id: $account_id})
        MATCH (n)
        WHERE n.id IN $evidence
          AND any(label IN labels(n) WHERE label IN ['Account', 'Device', 'IP'])
        WITH a, collect(DISTINCT n) AS evidence_nodes
        WHERE size(evidence_nodes) = size($evidence) AND size(evidence_nodes) > 0
        MERGE (c:Case {session_key: $sk})
        ON CREATE SET c.id = $cid,
                      c.account_id = $account_id,
                      c.verdict = $verdict,
                      c.summary = $summary,
                      c.created_at = $ts,
                      c.score = $score,
                      c.score_status = $score_status,
                      c.scoring_policy = $policy,
                      c.breakdown_json = $breakdown_json,
                      c.scoring_rules_json = $scoring_rules_json,
                      c.evidence_details_json = $evidence_details_json,
                      c.investigation_policy_json = $investigation_policy_json,
                      c.pagerank_score = $pagerank_score,
                      c.pagerank_rank = $pagerank_rank,
                      c.computed_verdict_hint = $computed_hint,
                      c.verdict_vs_computed = $vvc,
                      c.verdict_hint = $verdict_hint,
                      c.override_reason = $override_reason,
                      c.centrality_error = $centrality_error,
                      c.confirmed_cycle_txns_json = $cycle_txns_json,
                      c.flow_paths_json = $flow_paths_json
        WITH c, a, evidence_nodes
        WHERE c.account_id = a.id
        MERGE (c)-[:INVESTIGATES]->(a)
        WITH c, evidence_nodes
        UNWIND evidence_nodes AS en
        MERGE (c)-[:CITES]->(en)
        WITH c
        MATCH (c)-[:CITES]->(linked)
        RETURN c.id AS case_id, c.score AS score,
               count(DISTINCT linked) AS linked_count,
               c.scoring_policy AS scoring_policy,
               c.computed_verdict_hint AS computed_verdict_hint
        """,
        {
            "account_id": account_id, "sk": session_key, "cid": case_id,
            "verdict": verdict, "summary": summary, "ts": now.isoformat(),
            "score": computed["score"], "score_status": computed.get("score_status", "complete"),
            "policy": SCORING_POLICY_VERSION,
            "breakdown_json": breakdown_json,
            "scoring_rules_json": scoring_rules_json,
            "evidence_details_json": evidence_details_json,
            "investigation_policy_json": investigation_policy_json,
            "pagerank_score": computed["breakdown"].get("pagerank_score"),
            "pagerank_rank": computed["breakdown"].get("pagerank_rank"),
            "computed_hint": computed_hint,
            "vvc": verdict_vs_computed,
            "verdict_hint": computed["verdict_hint"],
            "override_reason": override_reason,
            "centrality_error": computed.get("centrality_error"),
            "cycle_txns_json": confirmed_cycle_txns_json,
            "flow_paths_json": flow_paths_json,
            "evidence": ids,
        },
    )

    if not result:
        claimed = run_read(
            "MATCH (c:Case {session_key: $sk}) RETURN c.account_id AS account_id LIMIT 1",
            {"sk": session_key},
        )
        if claimed and claimed[0].get("account_id") not in (None, account_id):
            return {"error": "investigation_id is already bound to a different account"}
        return {"error": "case write returned no result; evidence did not match exactly — check evidence IDs"}

    record = result[0]
    returned_case_id = record.get("case_id")
    ret = {
        "case_id": returned_case_id,
        "cited_evidence": record.get("linked_count", 0),
        "score": record.get("score", computed["score"]),
        "score_status": computed.get("score_status", "complete"),
        "policy": SCORING_POLICY_VERSION,       # kept for backward compatibility
        "scoring_policy": SCORING_POLICY_VERSION,
        "investigation_id": session_key,
        "deduplicated": returned_case_id != case_id,
    }
    if mismatch:
        ret["verdict_warning"] = (
            f"authorized override: submitted {verdict!r}, computed hint {computed_hint!r}"
        )
    return ret


TOOL_FUNCS = {
    "get_account": get_account,
    "shared_device_accounts": shared_device_accounts,
    "shared_ip_accounts": shared_ip_accounts,
    "outgoing_flows": outgoing_flows,
    "incoming_flows": incoming_flows,
    "money_cycles": money_cycles,
    "connection_path": connection_path,
    "top_risk_accounts": top_risk_accounts,
    "risk_score": risk_score,
    "prior_cases": prior_cases,
    "propagation_risk": propagation_risk,
    "max_flow_risk": max_flow_risk,
    "record_case": record_case,
}

TOOLS = [
    {"name": "get_account", "description": "Look up an account by id.",
     "input_schema": {"type": "object", "properties": {"account_id": {"type": "string"}},
                      "required": ["account_id"]}},
    {"name": "shared_device_accounts",
     "description": "Accounts that share at least one device fingerprint with this account.",
     "input_schema": {"type": "object", "properties": {"account_id": {"type": "string"}},
                      "required": ["account_id"]}},
    {"name": "shared_ip_accounts",
     "description": "Accounts that have logged in from the same IP address as this account.",
     "input_schema": {"type": "object", "properties": {"account_id": {"type": "string"}},
                      "required": ["account_id"]}},
    {"name": "outgoing_flows",
     "description": "Money paths flowing out of this account up to max_hops (1-4), "
                    "including transaction IDs and timestamps.",
     "input_schema": {"type": "object", "properties": {
         "account_id": {"type": "string"},
         "max_hops": {"type": "integer", "minimum": 1, "maximum": 4}},
         "required": ["account_id"]}},
    {"name": "incoming_flows",
     "description": "Transfer paths flowing INTO this account — call when PageRank identifies "
                    "this account as a top-three hub to identify who funds it.",
     "input_schema": {"type": "object", "properties": {
         "account_id": {"type": "string"},
         "max_hops": {"type": "integer", "minimum": 1, "maximum": 4}},
         "required": ["account_id"]}},
    {"name": "money_cycles",
     "description": "Transfer loops that start and end at this account (round-tripping money). "
                    "max_depth controls cycle length (3–10 hops, default 8) — increase to find larger rings.",
     "input_schema": {"type": "object", "properties": {
         "account_id": {"type": "string"},
         "max_depth": {"type": "integer", "minimum": 3, "maximum": 10,
                       "description": "Maximum cycle length in hops (default 8)."}},
         "required": ["account_id"]}},
    {"name": "connection_path",
     "description": "Shortest chain of money transfers from account_a to account_b "
                    "(follows transfer direction). Use it to explain how money reached an account.",
     "input_schema": {"type": "object", "properties": {
         "account_a": {"type": "string"}, "account_b": {"type": "string"}},
         "required": ["account_a", "account_b"]}},
    {"name": "top_risk_accounts",
     "description": "Run FalkorDB's native PageRank algorithm over the transfer graph. "
                    "Returns the 10 accounts most structurally central to money flows. "
                    "Pass anchor_id to restrict results to the account's 2-hop neighborhood "
                    "(subgraph-scoped ranking — rank among connected peers, not the whole graph).",
     "input_schema": {"type": "object", "properties": {
         "anchor_id": {"type": "string",
                       "description": "When set, returns only accounts in the 2-hop TRANSFERRED "
                                      "neighborhood ranked by global PageRank score."}}}},
    {"name": "risk_score",
     "description": "Compute a weighted fraud-risk score (0-100) from graph signals: "
                    "shared devices, shared IPs, money cycles, outgoing paths, and prior cases. "
                    "Returns score, verdict_hint (LOW/MEDIUM/HIGH), traversal_truncated flag, "
                    "investigation_policy, and a per-signal breakdown with scoring_rules.",
     "input_schema": {"type": "object", "properties": {"account_id": {"type": "string"}},
                      "required": ["account_id"]}},
    {"name": "prior_cases",
     "description": "Earlier investigation cases recorded against this account.",
     "input_schema": {"type": "object", "properties": {"account_id": {"type": "string"}},
                      "required": ["account_id"]}},
    {"name": "propagation_risk",
     "description": "Monte Carlo simulation (default 40 trials) of fraud spread probability "
                    "to each direct transfer target. Returns infection_probability (0–1) per neighbor "
                    "based on transfer frequency, shared devices, and shared IPs. "
                    "Call after risk_score when you want to rank which neighbors are most likely to be "
                    "part of the same scheme.",
     "input_schema": {"type": "object", "properties": {
         "account_id": {"type": "string"},
         "n_simulations": {"type": "integer", "minimum": 10, "maximum": 100,
                           "description": "Number of Monte Carlo trials per neighbor (default 40)."}},
         "required": ["account_id"]}},
    {"name": "max_flow_risk",
     "description": "Maximum-flow analysis (Edmonds-Karp) treating transfer amounts as edge capacities. "
                    "Returns the theoretical upper bound on funds that could flow from source_id to sink_id. "
                    "Use to quantify the financial exposure of a path between two accounts.",
     "input_schema": {"type": "object", "properties": {
         "source_id": {"type": "string", "description": "Account where funds originate."},
         "sink_id": {"type": "string", "description": "Account that receives the funds."}},
         "required": ["source_id", "sink_id"]}},
    {"name": "record_case",
     "description": "Save the verdict to the graph as a Case linked to the account and every evidence id. "
                    "Call once, at the end. If the verdict differs from risk_score's verdict_hint, "
                    "you must supply override_reason. Pass the exact investigation_id from the user message.",
     "input_schema": {"type": "object", "properties": {
         "account_id": {"type": "string"},
         "verdict": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
         "summary": {"type": "string"},
         "evidence_ids": {"type": "array", "items": {"type": "string"},
                          "description": "Account, Device, or IP ids the verdict relies on"},
         "override_reason": {"type": "string",
                             "description": "Required when verdict differs from computed hint, "
                                            "or when score_status is degraded."},
         "investigation_id": {"type": "string",
                              "description": "Idempotency key for this investigation — "
                                             "use the exact value from the user message."}},
         "required": ["account_id", "verdict", "summary", "evidence_ids", "investigation_id"]}},
]

SYSTEM = """You are a fraud investigator. The data is a property graph:
Account, Device, IP and Case nodes, with USED_DEVICE, LOGGED_FROM, TRANSFERRED and
INVESTIGATES/CITES edges. Investigate the account the user names.

1. Check the account exists and look at its prior cases.
2. Call top_risk_accounts to see if the target is structurally central in the network (PageRank).
3. Call risk_score to get a weighted score, breakdown, and investigation_policy.
   - Follow the investigation_policy.recommended_next field to decide your next step:
     * inspect_incoming_counterparties → call incoming_flows to identify who funds this account
     * inspect_shared_device_cluster  → call shared_device_accounts and trace those accounts
     * proceed_without_centrality     → continue but note the score is degraded
4. Look for links: shared devices, shared IPs, outgoing money flows, money cycles.
   - money_cycles returns cycle_type: only "confirmed_chronological" cycles count toward fraud.
   - If score_status is "degraded", explain the missing signal; do not present the score as complete.
4a. Optional deeper analysis when the risk is HIGH or links to multiple accounts are found:
   - propagation_risk(account_id) — Monte Carlo estimate of fraud spread probability to each
     direct neighbor; helps rank which connected accounts to investigate next.
   - max_flow_risk(source_id, sink_id) — Edmonds-Karp max-flow on transfer capacities; quantifies
     the theoretical upper bound on funds moving between two accounts.
5. Follow any suspicious link with connection_path so you can explain it.
6. Choose LOW, MEDIUM or HIGH based on the risk_score verdict_hint and your findings.
   Cite only ids that appeared in tool results.
   If your verdict differs from verdict_hint, you MUST supply override_reason.
   If score_status is "degraded", you MUST supply override_reason.
7. Call record_case once, with the verdict, a short summary, the evidence ids,
   and the exact investigation_id from the user message.

Finish with a short write-up that states the risk score, names each signal that
contributed, and traces the chain from the start account to the evidence."""


# Ollama's native /api/chat expects OpenAI-style tool schemas.
OLLAMA_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


def chat(messages: list[dict]) -> dict:
    """One round trip to the local model. Returns the assistant message."""
    body = json.dumps({
        "model": MODEL, "messages": messages, "tools": OLLAMA_TOOLS,
        "stream": False, "options": {"temperature": 0, "num_ctx": 16384},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["message"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"Ollama {e.code}: {err_body}") from e


def _message_content(message: dict) -> str:
    content = message.get("content", "")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _summarize_dropped_messages(messages: list[dict]) -> str:
    """Retain compact, decision-relevant highlights from pruned tool results."""
    findings: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_name = message.get("tool_name", "tool")
        content = _message_content(message)
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            findings.append(f"{tool_name}: result was unavailable in compact form")
            continue
        if isinstance(payload, dict) and "score" in payload:
            findings.append(
                f"risk_score={payload.get('score')} verdict_hint={payload.get('verdict_hint')} "
                f"status={payload.get('score_status')} breakdown={payload.get('breakdown')}"
            )
        elif isinstance(payload, list) and payload:
            ids = []
            for item in payload[:5]:
                if isinstance(item, dict):
                    ids.extend(
                        str(item[k]) for k in ("account", "case_id", "cycle_type")
                        if item.get(k) is not None
                    )
            if ids:
                findings.append(f"{tool_name}: " + ", ".join(ids[:10]))
            else:
                findings.append(f"{tool_name}: {len(payload)} result(s)")
        elif isinstance(payload, dict) and payload.get("error"):
            findings.append(f"{tool_name} error: {payload['error']}")
    unique = list(dict.fromkeys(findings))
    return "\n".join(unique[-12:]) if unique else "No decision-relevant details could be retained from older results."


def _exchange_groups(messages: list[dict]) -> list[list[dict]]:
    """Group assistant tool-call messages with their tool results for safe pruning."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role in ("assistant", "user"):
            if current:
                groups.append(current)
            current = [message]
        else:
            if not current:
                current = [message]
            else:
                current.append(message)
    if current:
        groups.append(current)
    return groups


def _prune_messages(messages: list[dict], max_messages: int = 14) -> list[dict]:
    """Prune complete exchanges, preserve system/original user, and summarize removed findings."""
    if len(messages) <= max_messages:
        return messages
    system = messages[0]
    original_user = messages[1]
    groups = _exchange_groups(messages[2:])
    kept_groups: list[list[dict]] = []
    kept_count = 0
    budget = max(1, max_messages - 3)
    for group in reversed(groups):
        if kept_groups and kept_count + len(group) > budget:
            break
        if not kept_groups and len(group) > budget:
            break
        kept_groups.append(group)
        kept_count += len(group)
    kept_groups.reverse()
    kept_messages = [msg for group in kept_groups for msg in group]
    kept_ids = {id(msg) for msg in kept_messages}
    dropped = [msg for msg in messages[2:] if id(msg) not in kept_ids]
    summary = {"role": "user", "content": "[Pruned investigation summary]\n" + _summarize_dropped_messages(dropped)}
    return [system, original_user, summary] + kept_messages


def investigate(account_id: str, max_steps: int = 12) -> None:
    investigation_id = str(uuid.uuid4())
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"Investigate account {account_id}. "
                    f"Use this exact investigation_id when recording the case: {investigation_id}"},
    ]
    recorded = False
    reminded = False
    # Loop-lock guard: maps (tool_name, args_json) → cached result.
    seen_results: dict[tuple[str, str], Any] = {}
    last_msg: dict = {}

    for _step in range(max_steps):
        messages = _prune_messages(messages)
        msg = chat(messages)
        last_msg = msg
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            if recorded or reminded:
                break
            reminded = True
            messages.append({
                "role": "user",
                "content": (
                    "You have not called record_case yet. Call it now with your verdict, "
                    f"summary, evidence ids, and investigation_id={investigation_id}."
                ),
            })
            continue

        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            # Ollama sometimes serializes args as a JSON string rather than a dict.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    out: Any = {"error": "tool arguments were not valid JSON"}
                    messages.append({"role": "tool", "tool_name": name,
                                     "content": json.dumps(out)})
                    continue
            if not isinstance(args, dict):
                out = {"error": "tool arguments must be a JSON object"}
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(out)})
                continue

            if name == "record_case":
                # Orchestration owns the idempotency key; the model cannot alter it.
                args["investigation_id"] = investigation_id

            call_key = (name, json.dumps(args, sort_keys=True, separators=(",", ":")))
            if call_key in seen_results:
                print(f"!! loop-lock on {name} — returning cached result")
                out = copy.deepcopy(seen_results[call_key])
            else:
                print(f"-> {name}({json.dumps(args, sort_keys=True)})")
                try:
                    out = TOOL_FUNCS[name](**args)
                except (KeyError, ValueError, TypeError, RuntimeError, OSError) as exc:
                    # Expected failure modes: bad args, network errors, FalkorDB query errors.
                    # Re-raise programmer errors (AttributeError, NameError, etc.) so they
                    # surface immediately rather than being swallowed as tool errors.
                    out = {"error": f"{type(exc).__name__}: {exc}"}
                seen_results[call_key] = copy.deepcopy(out)

            if name == "record_case" and "case_id" in out:
                recorded = True
                print(f"   saved {out['case_id']}")
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(out, default=str)})

    print("\n=== Investigation ===")
    print(_message_content(last_msg).strip() or "(model returned no write-up)")


if __name__ == "__main__":
    investigate(sys.argv[1] if len(sys.argv) > 1 else "ACC-RING-01")

"""Test suite for the fraud-investigator agent.

Run with:  python -m pytest tests.py -v
         OR python -m unittest discover -v

Unit tests mock the FalkorDB connection and run without a live database.
Integration tests (marked with # INTEGRATION) require seed.py to have been run.
"""
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Stub FalkorDB before importing agent so the module-level GRAPH = FalkorDB(...)
# does not attempt a real TCP connection.
_mock_graph = MagicMock()
_mock_falkordb = MagicMock()
_mock_falkordb.return_value.select_graph.return_value = _mock_graph
sys.modules.setdefault("falkordb", types.ModuleType("falkordb"))
import falkordb as _falkordb_mod  # noqa: E402
_falkordb_mod.FalkorDB = _mock_falkordb

import agent  # noqa: E402 — imported after stub is in place


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INV_ID = "test-investigation-id-001"  # stable investigation_id for all unit tests


def _make_risk_score(score=90, hint="HIGH", status="complete", rank=1, pr_score=0.42):
    return {
        "score": score,
        "score_status": status,
        "verdict_hint": hint,
        "traversal_truncated": False,
        "scoring_policy_version": "2.0",
        "investigation_policy": {
            "recommended_next": "inspect_incoming_counterparties",
            "reason": "top-1 PageRank hub",
        },
        "breakdown": {
            "shared_device_peers": 2, "shared_device_pts": 28,
            "shared_ip_peers": 1, "shared_ip_pts": 5,
            "money_cycles_total": 1, "money_cycles_confirmed": 1,
            "structural_only_cycles": 0, "cycle_pts": 18,
            "confirmed_cycle_txn_ids": [],
            "outgoing_paths": 3, "flow_pts": 6,
            "flow_paths": [
                {"path": ["ACC-1", "ACC-2"], "hops": 1, "amounts": [500]},
                {"path": ["ACC-1", "ACC-2", "ACC-3"], "hops": 2, "amounts": [500, 480]},
                {"path": ["ACC-1", "ACC-3"], "hops": 1, "amounts": [300]},
            ],
            "prior_cases": 0, "prior_pts": 0,
            "pagerank_rank": rank, "pagerank_score": pr_score,
            "centrality_pts": 20,
        },
        "evidence_details": {
            "shared_device_evidence": [], "shared_ip_evidence": [],
            "confirmed_cycles": [], "structural_only_cycles": [],
            "outgoing_paths": [], "prior_case_ids": [],
            "pagerank": {"rank": rank, "score": pr_score, "error": None},
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


# ---------------------------------------------------------------------------
# Dimension 1 — Multi-Hop Graph Reasoning
# ---------------------------------------------------------------------------

class TestMoneyCycles(unittest.TestCase):

    def test_confirmed_chronological_cycle(self):
        """A cycle where timestamps are non-decreasing is confirmed_chronological."""
        row = {
            "cycle": ["A", "B", "C", "A"],
            "amounts": [100, 200, 300],
            "txn_ids": ["TX1", "TX2", "TX3"],
            "timestamps": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
        with patch.object(agent, "run_read", return_value=[row]):
            results = agent.money_cycles("A")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["cycle_type"], "confirmed_chronological")

    def test_structural_only_cycle_reversed_timestamps(self):
        """A cycle where money moves backward in time is structural_only."""
        row = {
            "cycle": ["A", "B", "C", "A"],
            "amounts": [100, 200, 300],
            "txn_ids": ["TX1", "TX2", "TX3"],
            "timestamps": ["2026-01-03", "2026-01-02", "2026-01-01"],
        }
        with patch.object(agent, "run_read", return_value=[row]):
            results = agent.money_cycles("A")
        self.assertEqual(results[0]["cycle_type"], "structural_only")

    def test_cycle_includes_txn_ids_and_timestamps(self):
        row = {
            "cycle": ["A", "B", "A"],
            "amounts": [500, 500],
            "txn_ids": ["TX-RING-0", "TX-RING-1"],
            "timestamps": ["2026-09-01", "2026-09-02"],
        }
        with patch.object(agent, "run_read", return_value=[row]):
            results = agent.money_cycles("A")
        self.assertIn("txn_ids", results[0])
        self.assertIn("timestamps", results[0])


# ---------------------------------------------------------------------------
# Dimension 2 — Graph Algorithms as Agent Tools
# ---------------------------------------------------------------------------

class TestRiskScorePageRank(unittest.TestCase):

    def _patch_all(self, pagerank_rows, devices=None, ips=None, cycles=None, flows=None, cases=None):
        devices = devices or []
        ips = ips or []
        cycles = cycles or []
        flows = flows or []
        cases = cases or []

        def fake_top_risk(anchor_id=None):
            return pagerank_rows

        def fake_shared_device(acct):
            return devices

        def fake_shared_ip(acct):
            return ips

        def fake_money_cycles(acct, max_depth=None):
            return cycles

        def fake_outgoing(acct, max_hops=3):
            return flows

        def fake_prior(acct):
            return cases

        return {
            "top_risk_accounts": fake_top_risk,
            "shared_device_accounts": fake_shared_device,
            "shared_ip_accounts": fake_shared_ip,
            "money_cycles": fake_money_cycles,
            "outgoing_flows": fake_outgoing,
            "prior_cases": fake_prior,
        }

    def test_top3_centrality_uses_numeric_score(self):
        """Numeric PageRank score is preserved in breakdown."""
        ranked = [{"account": "ACC-1", "score": 0.9}, {"account": "ACC-2", "score": 0.5}]
        patches = self._patch_all(ranked)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ACC-1")
        self.assertEqual(result["breakdown"]["pagerank_rank"], 1)
        self.assertAlmostEqual(result["breakdown"]["pagerank_score"], 0.9)
        self.assertEqual(result["breakdown"]["centrality_pts"], 20)
        self.assertEqual(result["score_status"], "complete")

    def test_top10_centrality(self):
        ranked = [{"account": f"ACC-{i}", "score": 1.0 / i} for i in range(1, 11)]
        ranked[7]["account"] = "TARGET"  # rank 8
        patches = self._patch_all(ranked)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("TARGET")
        self.assertEqual(result["breakdown"]["pagerank_rank"], 8)
        self.assertEqual(result["breakdown"]["centrality_pts"], 10)

    def test_outside_top10_no_centrality_pts(self):
        ranked = [{"account": f"ACC-{i}", "score": 1.0} for i in range(10)]
        patches = self._patch_all(ranked)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("OUTSIDER")
        self.assertIsNone(result["breakdown"]["pagerank_rank"])
        self.assertEqual(result["breakdown"]["centrality_pts"], 0)

    def test_pagerank_failure_sets_degraded_status(self):
        def boom(anchor_id=None):
            raise RuntimeError("algo.pageRank unavailable")

        patches = self._patch_all([])
        patches["top_risk_accounts"] = boom
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ANY")
        self.assertEqual(result["score_status"], "degraded")
        self.assertIn("centrality_error", result)
        self.assertIn("proceed_without_centrality",
                      result["investigation_policy"]["recommended_next"])

    def test_investigation_policy_top3_recommends_incoming(self):
        ranked = [{"account": "HUB", "score": 0.99}]
        patches = self._patch_all(ranked)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("HUB")
        self.assertIn("inspect_incoming", result["investigation_policy"]["recommended_next"])

    def test_investigation_policy_rank4_recommends_device(self):
        ranked = [{"account": f"TOP{i}", "score": 1.0} for i in range(3)]
        ranked.append({"account": "MID", "score": 0.5})
        patches = self._patch_all(ranked)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("MID")
        self.assertIn("device", result["investigation_policy"]["recommended_next"])

    def test_risk_score_has_evidence_details(self):
        """risk_score result includes evidence_details with per-tool breakdowns."""
        ranked = [{"account": "T", "score": 0.5}]
        patches = self._patch_all(ranked)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("T")
        self.assertIn("evidence_details", result)
        ed = result["evidence_details"]
        self.assertIn("shared_device_evidence", ed)
        self.assertIn("confirmed_cycles", ed)
        self.assertIn("pagerank", ed)

    def test_risk_score_has_scoring_policy_version(self):
        """scoring_policy_version field is present and matches module constant."""
        patches = self._patch_all([])
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ANY")
        self.assertEqual(result["scoring_policy_version"], agent.SCORING_POLICY_VERSION)

    def test_structural_only_cycles_counted_separately(self):
        """Structural-only cycles appear in breakdown but score zero points."""
        structural = [{"cycle": ["A", "B", "A"], "amounts": [100],
                       "txn_ids": ["TX1"], "timestamps": ["2026-01-02", "2026-01-01"],
                       "cycle_type": "structural_only"}]
        patches = self._patch_all([], cycles=structural)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("A")
        self.assertEqual(result["breakdown"]["structural_only_cycles"], 1)
        self.assertEqual(result["breakdown"]["money_cycles_confirmed"], 0)
        self.assertEqual(result["breakdown"]["cycle_pts"], 0)


# ---------------------------------------------------------------------------
# Dimension 3 — MCP Access Control
# ---------------------------------------------------------------------------

# Stub the 'mcp' package before mcp_server is imported so FastMCP doesn't fail.
_mcp_stub = types.ModuleType("mcp")
_mcp_server_stub = types.ModuleType("mcp.server")
_mcp_fastmcp_stub = types.ModuleType("mcp.server.fastmcp")


class _FakeFastMCP:
    def __init__(self, name):
        self.name = name

    def tool(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def run(self):
        pass


_mcp_fastmcp_stub.FastMCP = _FakeFastMCP
_mcp_stub.server = _mcp_server_stub
_mcp_server_stub.fastmcp = _mcp_fastmcp_stub
sys.modules.setdefault("mcp", _mcp_stub)
sys.modules.setdefault("mcp.server", _mcp_server_stub)
sys.modules.setdefault("mcp.server.fastmcp", _mcp_fastmcp_stub)

import mcp_server  # noqa: E402 — imported after stubs are in place


class TestMCPAccessControl(unittest.TestCase):

    def test_write_disabled_by_default(self):
        with patch.object(mcp_server, "_WRITE_ENABLED", False):
            result = mcp_server.record_case("ACC-1", "HIGH", "test", ["ACC-2"])
        self.assertIn("error", result)
        self.assertIn("disabled", result["error"])

    def test_write_enabled_no_key_allows_any_caller(self):
        fake_out = {"case_id": "CASE-1", "cited_evidence": 1, "score": 80,
                    "policy": "2.0", "score_status": "complete"}
        with patch.object(mcp_server, "_WRITE_ENABLED", True), \
             patch.object(mcp_server, "_WRITE_API_KEY", ""), \
             patch.object(agent, "record_case", return_value=fake_out):
            result = mcp_server.record_case("ACC-1", "HIGH", "test", ["ACC-1"])
        self.assertIn("case_id", result)

    def test_write_enabled_with_key_rejects_wrong_caller(self):
        with patch.object(mcp_server, "_WRITE_ENABLED", True), \
             patch.object(mcp_server, "_WRITE_API_KEY", "secret-key"):
            result = mcp_server.record_case(
                "ACC-1", "HIGH", "test", ["ACC-1"], caller_api_key="wrong-key"
            )
        self.assertIn("error", result)
        self.assertIn("unauthorized", result["error"])

    def test_write_enabled_with_key_accepts_correct_caller(self):
        fake_out = {"case_id": "CASE-1", "cited_evidence": 1, "score": 80,
                    "policy": "2.0", "score_status": "complete"}
        with patch.object(mcp_server, "_WRITE_ENABLED", True), \
             patch.object(mcp_server, "_WRITE_API_KEY", "secret-key"), \
             patch.object(agent, "record_case", return_value=fake_out):
            result = mcp_server.record_case(
                "ACC-1", "HIGH", "test", ["ACC-1"], caller_api_key="secret-key"
            )
        self.assertIn("case_id", result)

    def test_invalid_evidence_id_rejected_at_mcp_layer(self):
        with patch.object(mcp_server, "_WRITE_ENABLED", True), \
             patch.object(mcp_server, "_WRITE_API_KEY", ""):
            result = mcp_server.record_case("ACC-1", "HIGH", "test", ["", "  "])
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Dimension 4 — Actions Based on Graph State
# ---------------------------------------------------------------------------

class TestRecordCase(unittest.TestCase):

    def _patch_graph(self, account_exists=True, label_rows=None, existing_case=None,
                     write_result=None, risk_override=None):
        """Patch run_read/run_write/risk_score/ensure_case_uniqueness_constraint for record_case tests."""
        def fake_get_account(acct):
            return [{"id": acct}] if account_exists else []

        # New format: _validate_evidence_ids returns {"id": ..., "labels": [...]}
        label_rows = label_rows or [{"id": "ACC-1", "labels": ["Account"]}]
        existing_case = existing_case or []
        write_result = write_result or [{"linked_count": 1, "case_id": "CASE-TEST-001",
                                         "score": 90, "scoring_policy": "2.0",
                                         "computed_verdict_hint": "HIGH"}]

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return existing_case
            # _validate_evidence_ids uses: MATCH (n) WHERE n.id IN $ids RETURN n.id, labels(n)
            if "WHERE n.id IN" in query:
                return label_rows
            if "CITES" in query:
                return [{"cnt": 1}]
            return []

        risk = risk_override or _make_risk_score()

        patches = {
            "get_account": fake_get_account,
            "run_read": fake_run_read,
            "run_write": MagicMock(return_value=write_result),
            "risk_score": MagicMock(return_value=risk),
            "ensure_case_uniqueness_constraint": MagicMock(return_value=True),
        }
        return patches

    def test_verdict_match_succeeds(self):
        p = self._patch_graph()
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertIn("case_id", result)
        self.assertNotIn("error", result)

    def test_verdict_mismatch_without_reason_rejected(self):
        risk = _make_risk_score(hint="HIGH")
        p = self._patch_graph(risk_override=risk)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "LOW", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)
        self.assertIn("override_reason", result["error"])

    def test_verdict_mismatch_with_reason_accepted(self):
        risk = _make_risk_score(hint="HIGH")
        p = self._patch_graph(risk_override=risk)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "LOW", "test", ["ACC-1"],
                                       override_reason="Investigator reviewed manually",
                                       investigation_id=_INV_ID)
        self.assertIn("case_id", result)

    def test_empty_evidence_rejected(self):
        p = self._patch_graph()
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", [],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)

    def test_missing_evidence_id_rejected(self):
        p = self._patch_graph(label_rows=[])  # no nodes found
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["NONEXISTENT"],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)
        self.assertIn("does not exist", result["error"])

    def test_unsupported_label_rejected(self):
        p = self._patch_graph(label_rows=[{"id": "CASE-OLD", "labels": ["Case"]}])
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["CASE-OLD"],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)
        self.assertIn("unsupported", result["error"])

    def test_duplicate_evidence_ids_deduplicated(self):
        """Duplicate evidence IDs are collapsed; the write still succeeds."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        p = self._patch_graph(label_rows=label_rows)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test",
                                       ["ACC-1", "ACC-1", "ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertIn("case_id", result)

    def test_dedup_path_returns_actual_linked_count(self):
        # account_id must be included so the cross-account check passes
        existing = [{"case_id": "CASE-EXISTING", "account_id": "ACC-1"}]
        p = self._patch_graph(existing_case=existing)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertEqual(result["case_id"], "CASE-EXISTING")
        self.assertIn("cited_evidence", result)

    def test_breakdown_json_persisted(self):
        """run_write is called with a non-empty breakdown_json parameter."""
        p = self._patch_graph()
        with patch.multiple(agent, **p):
            agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                              investigation_id=_INV_ID)
        call_params = p["run_write"].call_args[0][1]
        breakdown = json.loads(call_params["breakdown_json"])
        self.assertIn("pagerank_rank", breakdown)
        self.assertIn("pagerank_score", breakdown)

    def test_score_status_persisted_on_degraded_with_override(self):
        """When score is degraded and override_reason is provided, score_status is written to DB."""
        risk = _make_risk_score(status="degraded")
        p = self._patch_graph(risk_override=risk)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       override_reason="degraded — accepted by analyst",
                                       investigation_id=_INV_ID)
        call_params = p["run_write"].call_args[0][1]
        self.assertEqual(call_params["score_status"], "degraded")
        self.assertEqual(result.get("score_status"), "degraded")

    def test_unknown_account_rejected(self):
        p = self._patch_graph(account_exists=False)
        with patch.multiple(agent, **p):
            result = agent.record_case("GHOST", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Dimension 5 — Explainability
# ---------------------------------------------------------------------------

class TestExplainability(unittest.TestCase):

    def test_risk_score_breakdown_has_pagerank_score(self):
        ranked = [{"account": "TARGET", "score": 0.75}]
        with patch.multiple(agent,
                            top_risk_accounts=lambda anchor_id=None: ranked,
                            shared_device_accounts=lambda a: [],
                            shared_ip_accounts=lambda a: [],
                            money_cycles=lambda a, max_depth=None: [],
                            outgoing_flows=lambda a, max_hops=3: [],
                            prior_cases=lambda a: []):
            result = agent.risk_score("TARGET")
        self.assertAlmostEqual(result["breakdown"]["pagerank_score"], 0.75)
        self.assertIn("pagerank_rank", result["breakdown"])

    def test_score_status_complete_when_pagerank_ok(self):
        ranked = [{"account": "X", "score": 0.1}]
        with patch.multiple(agent,
                            top_risk_accounts=lambda anchor_id=None: ranked,
                            shared_device_accounts=lambda a: [],
                            shared_ip_accounts=lambda a: [],
                            money_cycles=lambda a, max_depth=None: [],
                            outgoing_flows=lambda a, max_hops=3: [],
                            prior_cases=lambda a: []):
            result = agent.risk_score("X")
        self.assertEqual(result["score_status"], "complete")


# ---------------------------------------------------------------------------
# Bonus B2 — Loop resilience (loop-lock caching)
# ---------------------------------------------------------------------------

class TestLoopLock(unittest.TestCase):

    def test_loop_lock_calls_tool_only_once(self):
        """Duplicate tool call should use cached result; underlying function called once."""
        call_count = {"n": 0}
        fake_result = [{"id": "ACC-1"}]

        def fake_get_account(account_id):
            call_count["n"] += 1
            return fake_result

        first_call = {"function": {"name": "get_account", "arguments": {"account_id": "ACC-1"}}}
        step = [0]

        def fake_chat(messages):
            step[0] += 1
            if step[0] <= 2:
                return {"role": "assistant", "tool_calls": [first_call]}
            return {"role": "assistant", "content": "done", "tool_calls": []}

        original_funcs = dict(agent.TOOL_FUNCS)
        agent.TOOL_FUNCS["get_account"] = fake_get_account
        try:
            with patch.object(agent, "chat", side_effect=fake_chat), \
                 patch.object(agent, "_prune_messages", side_effect=lambda m: m):
                agent.investigate("ACC-1", max_steps=5)
        finally:
            agent.TOOL_FUNCS.update(original_funcs)

        self.assertEqual(call_count["n"], 1, "Tool should be called once; duplicate uses cache")


# ---------------------------------------------------------------------------
# Bonus B2 — _prune_messages preserves complete exchanges
# ---------------------------------------------------------------------------

class TestPruneMessages(unittest.TestCase):

    def _make_messages(self, n_exchanges):
        """system + user + n_exchanges × (assistant + tool) = 2 + 2n messages."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "investigate"},
        ]
        for i in range(n_exchanges):
            msgs.append({"role": "assistant", "tool_calls": [{"function": {"name": f"tool{i}"}}]})
            msgs.append({"role": "tool", "content": f"result{i}"})
        return msgs

    def test_short_list_unchanged(self):
        msgs = self._make_messages(5)  # 2 + 10 = 12 messages
        result = agent._prune_messages(msgs)
        self.assertEqual(len(result), len(msgs))

    def test_pruned_list_starts_with_assistant(self):
        """After pruning, the first non-fixed message must be an assistant turn."""
        msgs = self._make_messages(8)  # 2 + 16 = 18 > 14
        result = agent._prune_messages(msgs)
        # system + user + summary = 3 fixed
        if len(result) > 3:
            self.assertEqual(result[3]["role"], "assistant")

    def test_pruned_list_total_within_limit(self):
        msgs = self._make_messages(10)  # 22 messages
        result = agent._prune_messages(msgs)
        self.assertLessEqual(len(result), 15)  # generous upper bound


# ---------------------------------------------------------------------------
# Gap 1.1/1.3 — Traversal truncation detection
# ---------------------------------------------------------------------------

class TestTraversalTruncation(unittest.TestCase):

    def _patches_with_sizes(self, n_devices=0, n_ips=0, n_cycles=0, n_flows=0):
        """Return patch dict producing result sets of given sizes."""
        def fake_devices(a):
            return [{"account": f"DEV-ACC-{i}"} for i in range(n_devices)]

        def fake_ips(a):
            return [{"account": f"IP-ACC-{i}"} for i in range(n_ips)]

        def fake_cycles(a, max_depth=None):
            return [{"cycle": ["A", "B", "A"], "amounts": [100], "txn_ids": [f"TX-{i}"],
                     "timestamps": ["2026-01-01", "2026-01-02"],
                     "cycle_type": "confirmed_chronological"} for i in range(n_cycles)]

        def fake_flows(a, max_hops=3):
            return [{"account": f"FLOW-{i}", "hops": 1,
                     "path": ["SRC", f"FLOW-{i}"], "amounts": [100 + i]}
                    for i in range(n_flows)]

        return {
            "top_risk_accounts": lambda anchor_id=None: [],
            "shared_device_accounts": fake_devices,
            "shared_ip_accounts": fake_ips,
            "money_cycles": fake_cycles,
            "outgoing_flows": fake_flows,
            "prior_cases": lambda a: [],
        }

    def test_no_truncation_when_below_limits(self):
        patches = self._patches_with_sizes(n_devices=5, n_ips=3, n_cycles=2, n_flows=10)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ANY")
        self.assertFalse(result["traversal_truncated"])

    def test_truncation_flagged_when_devices_hit_limit(self):
        """Exactly _TRAVERSAL_LIMITS['shared_device']=20 results → truncated."""
        patches = self._patches_with_sizes(n_devices=agent._TRAVERSAL_LIMITS["shared_device"])
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ANY")
        self.assertTrue(result["traversal_truncated"])

    def test_truncation_flagged_when_cycles_hit_limit(self):
        patches = self._patches_with_sizes(n_cycles=agent._TRAVERSAL_LIMITS["cycles"])
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ANY")
        self.assertTrue(result["traversal_truncated"])

    def test_truncation_flagged_when_flows_hit_limit(self):
        patches = self._patches_with_sizes(n_flows=agent._TRAVERSAL_LIMITS["outgoing_flows"])
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ANY")
        self.assertTrue(result["traversal_truncated"])

    def test_confirmed_cycle_txn_ids_in_breakdown(self):
        """Confirmed cycle txn_ids are captured in the breakdown for provenance."""
        patches = self._patches_with_sizes(n_cycles=2)
        with patch.multiple(agent, **patches):
            result = agent.risk_score("ANY")
        txn_ids = result["breakdown"]["confirmed_cycle_txn_ids"]
        self.assertEqual(len(txn_ids), 2)
        self.assertIsInstance(txn_ids[0], list)


# ---------------------------------------------------------------------------
# Gap 4.3 — MERGE-based idempotent case creation
# ---------------------------------------------------------------------------

class TestMergeIdempotency(unittest.TestCase):

    def test_merge_used_not_create(self):
        """record_case write query uses MERGE (not CREATE) for atomic idempotency."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        write_result = [{"linked_count": 1, "case_id": "CASE-NEW",
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        risk = _make_risk_score()
        write_mock = MagicMock(return_value=write_result)

        with patch.multiple(agent,
                            get_account=lambda a: [{"id": a}],
                            run_read=fake_run_read,
                            run_write=write_mock,
                            risk_score=MagicMock(return_value=risk),
                            ensure_case_uniqueness_constraint=MagicMock(return_value=True)):
            agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                              investigation_id=_INV_ID)

        call_query = write_mock.call_args[0][0]
        self.assertIn("MERGE", call_query.upper())
        self.assertNotIn("CREATE (c:Case", call_query)

    def test_concurrent_retry_uses_returned_case_id(self):
        """When MERGE matches existing Case (concurrent retry), returned case_id is used."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        existing_id = "CASE-EXISTING-123"
        write_result = [{"linked_count": 2, "case_id": existing_id,
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []  # pre-check misses (simulates race)
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        risk = _make_risk_score()

        with patch.multiple(agent,
                            get_account=lambda a: [{"id": a}],
                            run_read=fake_run_read,
                            run_write=MagicMock(return_value=write_result),
                            risk_score=MagicMock(return_value=risk),
                            ensure_case_uniqueness_constraint=MagicMock(return_value=True)):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)

        self.assertEqual(result["case_id"], existing_id)
        self.assertEqual(result["cited_evidence"], 2)

    def test_cycle_txns_persisted_on_case(self):
        """confirmed_cycle_txns_json is passed to run_write when cycles exist."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        write_result = [{"linked_count": 1, "case_id": "CASE-NEW",
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]
        risk = _make_risk_score()
        risk["breakdown"]["confirmed_cycle_txn_ids"] = [["TX-A", "TX-B"]]

        write_mock = MagicMock(return_value=write_result)

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        with patch.multiple(agent,
                            get_account=lambda a: [{"id": a}],
                            run_read=fake_run_read,
                            run_write=write_mock,
                            risk_score=MagicMock(return_value=risk),
                            ensure_case_uniqueness_constraint=MagicMock(return_value=True)):
            agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                              investigation_id=_INV_ID)

        call_params = write_mock.call_args[0][1]
        txns = json.loads(call_params["cycle_txns_json"])
        self.assertEqual(txns, [["TX-A", "TX-B"]])

    def test_verdict_hint_first_class_property_on_case(self):
        """verdict_hint is passed as a named parameter so it's stored as a Case property."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        write_result = [{"linked_count": 1, "case_id": "CASE-NEW",
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]
        risk = _make_risk_score(hint="HIGH")
        write_mock = MagicMock(return_value=write_result)

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        with patch.multiple(agent,
                            get_account=lambda a: [{"id": a}],
                            run_read=fake_run_read,
                            run_write=write_mock,
                            risk_score=MagicMock(return_value=risk),
                            ensure_case_uniqueness_constraint=MagicMock(return_value=True)):
            agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                              investigation_id=_INV_ID)

        call_params = write_mock.call_args[0][1]
        self.assertEqual(call_params["verdict_hint"], "HIGH")


# ---------------------------------------------------------------------------
# Fix 1 — money_cycles max_depth parameter
# ---------------------------------------------------------------------------

class TestMoneyCyclesDepth(unittest.TestCase):

    def _captured_query(self, account_id, **kwargs):
        """Return the Cypher query string passed to run_read."""
        captured = {}

        def fake_run_read(query, params=None):
            captured["query"] = query
            return []

        with patch.object(agent, "run_read", side_effect=fake_run_read):
            agent.money_cycles(account_id, **kwargs)
        return captured.get("query", "")

    def test_default_depth_is_max_cycle_depth(self):
        """Default max_depth uses _MAX_CYCLE_DEPTH (8), not the old hard-coded 5."""
        query = self._captured_query("A")
        self.assertIn(str(agent._MAX_CYCLE_DEPTH), query)
        self.assertNotIn("*2..5", query)

    def test_custom_depth_reflected_in_query(self):
        query = self._captured_query("A", max_depth=4)
        self.assertIn("*2..4", query)

    def test_depth_clamped_to_safe_maximum(self):
        """max_depth=100 is clamped to 10 before Cypher interpolation."""
        query = self._captured_query("A", max_depth=100)
        self.assertIn("*2..10", query)
        self.assertNotIn("100", query)

    def test_depth_clamped_to_minimum(self):
        """max_depth=1 is clamped to 3 (minimum meaningful cycle length)."""
        query = self._captured_query("A", max_depth=1)
        self.assertIn("*2..3", query)


# ---------------------------------------------------------------------------
# Fix 2 — top_risk_accounts neighborhood filtering
# ---------------------------------------------------------------------------

class TestNeighborhoodCentrality(unittest.TestCase):

    def test_no_anchor_id_issues_single_query(self):
        """Without anchor_id, only one run_read call (no neighborhood query)."""
        call_count = {"n": 0}

        def fake_run_read(query, params=None):
            call_count["n"] += 1
            return []

        with patch.object(agent, "run_read", side_effect=fake_run_read):
            agent.top_risk_accounts()
        self.assertEqual(call_count["n"], 1)

    def test_anchor_id_issues_neighborhood_query_first(self):
        """With anchor_id, first run_read fetches neighborhood, second fetches PageRank."""
        queries = []

        def fake_run_read(query, params=None):
            queries.append(query)
            return []

        with patch.object(agent, "run_read", side_effect=fake_run_read):
            agent.top_risk_accounts(anchor_id="ACC-1")
        self.assertEqual(len(queries), 2)
        self.assertIn("TRANSFERRED*1..2", queries[0])   # neighborhood query
        self.assertIn("pageRank", queries[1])            # global PageRank query

    def test_anchor_id_filters_pagerank_to_neighborhood(self):
        """PageRank query with anchor_id includes $ids param from neighborhood."""
        pagerank_params = {}

        def fake_run_read(query, params=None):
            if "pageRank" in query:
                pagerank_params.update(params or {})
                return []
            return [{"nid": "NBR-1"}, {"nid": "NBR-2"}]

        with patch.object(agent, "run_read", side_effect=fake_run_read):
            agent.top_risk_accounts(anchor_id="ANCHOR")
        self.assertIn("ANCHOR", pagerank_params.get("ids", []))
        self.assertIn("NBR-1", pagerank_params.get("ids", []))

    def test_risk_score_passes_anchor_id_to_top_risk_accounts(self):
        """risk_score passes account_id as anchor_id to top_risk_accounts."""
        received = {}

        def fake_top_risk(anchor_id=None):
            received["anchor_id"] = anchor_id
            return []

        with patch.multiple(agent,
                            top_risk_accounts=fake_top_risk,
                            shared_device_accounts=lambda a: [],
                            shared_ip_accounts=lambda a: [],
                            money_cycles=lambda a, max_depth=None: [],
                            outgoing_flows=lambda a, max_hops=3: [],
                            prior_cases=lambda a: []):
            agent.risk_score("TARGET-123")
        self.assertEqual(received["anchor_id"], "TARGET-123")


# ---------------------------------------------------------------------------
# Fix 3 — flow_paths provenance in breakdown and Case
# ---------------------------------------------------------------------------

class TestFlowPathsProvenance(unittest.TestCase):

    def _patches_with_flows(self, flows):
        def fake_top_risk(anchor_id=None):
            return []

        return {
            "top_risk_accounts": fake_top_risk,
            "shared_device_accounts": lambda a: [],
            "shared_ip_accounts": lambda a: [],
            "money_cycles": lambda a, max_depth=None: [],
            "outgoing_flows": lambda a, max_hops=3: flows,
            "prior_cases": lambda a: [],
        }

    def test_flow_paths_in_risk_score_breakdown(self):
        """risk_score includes flow_paths with path/hops/amounts for each scored flow."""
        flows = [
            {"account": "B", "hops": 1, "path": ["A", "B"], "amounts": [100]},
            {"account": "C", "hops": 2, "path": ["A", "B", "C"], "amounts": [100, 90]},
        ]
        with patch.multiple(agent, **self._patches_with_flows(flows)):
            result = agent.risk_score("A")
        fp = result["breakdown"]["flow_paths"]
        self.assertEqual(len(fp), 2)
        self.assertIn("path", fp[0])
        self.assertIn("hops", fp[0])
        self.assertIn("amounts", fp[0])

    def test_flow_paths_capped_at_five(self):
        """Only up to 5 flow paths are stored, matching the scoring cap."""
        flows = [
            {"account": f"ACC-{i}", "hops": 1, "path": ["A", f"ACC-{i}"], "amounts": [i * 10]}
            for i in range(8)
        ]
        with patch.multiple(agent, **self._patches_with_flows(flows)):
            result = agent.risk_score("A")
        self.assertLessEqual(len(result["breakdown"]["flow_paths"]), 5)

    def test_flow_paths_json_persisted_on_case(self):
        """record_case passes flow_paths_json to run_write when flows exist."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        write_result = [{"linked_count": 1, "case_id": "CASE-NEW",
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]
        risk = _make_risk_score()
        write_mock = MagicMock(return_value=write_result)

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        with patch.multiple(agent,
                            get_account=lambda a: [{"id": a}],
                            run_read=fake_run_read,
                            run_write=write_mock,
                            risk_score=MagicMock(return_value=risk),
                            ensure_case_uniqueness_constraint=MagicMock(return_value=True)):
            agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                              investigation_id=_INV_ID)

        call_params = write_mock.call_args[0][1]
        self.assertIn("flow_paths_json", call_params)
        paths = json.loads(call_params["flow_paths_json"])
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 3)

    def test_flow_paths_json_empty_when_no_flows(self):
        """flow_paths_json is '[]' when there are no outgoing flows."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        write_result = [{"linked_count": 1, "case_id": "CASE-NEW",
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]
        risk = _make_risk_score()
        risk["breakdown"]["flow_paths"] = []
        write_mock = MagicMock(return_value=write_result)

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        with patch.multiple(agent,
                            get_account=lambda a: [{"id": a}],
                            run_read=fake_run_read,
                            run_write=write_mock,
                            risk_score=MagicMock(return_value=risk),
                            ensure_case_uniqueness_constraint=MagicMock(return_value=True)):
            agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                              investigation_id=_INV_ID)

        call_params = write_mock.call_args[0][1]
        self.assertEqual(call_params["flow_paths_json"], "[]")


# ---------------------------------------------------------------------------
# New — investigation_id idempotency key
# ---------------------------------------------------------------------------

class TestInvestigationId(unittest.TestCase):

    def _base_patches(self, existing_case=None):
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        write_result = [{"linked_count": 1, "case_id": "CASE-T",
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]
        existing_case = existing_case or []

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return existing_case
            if "WHERE n.id IN" in query:
                return label_rows
            if "CITES" in query:
                return [{"cnt": 1}]
            return []

        return {
            "get_account": lambda a: [{"id": a}],
            "run_read": fake_run_read,
            "run_write": MagicMock(return_value=write_result),
            "risk_score": MagicMock(return_value=_make_risk_score()),
            "ensure_case_uniqueness_constraint": MagicMock(return_value=True),
        }

    def test_investigation_id_required(self):
        """record_case rejects calls without investigation_id."""
        p = self._base_patches()
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"])
        self.assertIn("error", result)
        self.assertIn("investigation_id", result["error"])

    def test_investigation_id_used_as_session_key(self):
        """The investigation_id is passed to run_write as the session key (sk)."""
        p = self._base_patches()
        with patch.multiple(agent, **p):
            agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                              investigation_id="my-unique-id-xyz")
        call_params = p["run_write"].call_args[0][1]
        self.assertEqual(call_params["sk"], "my-unique-id-xyz")

    def test_investigation_id_cross_account_rejected(self):
        """Using an investigation_id already bound to a different account returns error."""
        existing = [{"case_id": "CASE-OLD", "account_id": "ACC-OTHER"}]
        p = self._base_patches(existing_case=existing)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)
        self.assertIn("different account", result["error"])

    def test_result_includes_investigation_id(self):
        """Successful record_case result echoes investigation_id back to the caller."""
        p = self._base_patches()
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertEqual(result.get("investigation_id"), _INV_ID)


# ---------------------------------------------------------------------------
# New — degraded score requires override_reason
# ---------------------------------------------------------------------------

class TestDegradedScorePolicy(unittest.TestCase):

    def _base_patches(self, risk):
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]
        write_result = [{"linked_count": 1, "case_id": "CASE-T",
                         "score": 90, "scoring_policy": "2.0",
                         "computed_verdict_hint": "HIGH"}]

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        return {
            "get_account": lambda a: [{"id": a}],
            "run_read": fake_run_read,
            "run_write": MagicMock(return_value=write_result),
            "risk_score": MagicMock(return_value=risk),
            "ensure_case_uniqueness_constraint": MagicMock(return_value=True),
        }

    def test_degraded_score_without_override_rejected(self):
        """A degraded score requires override_reason — rejected if missing."""
        risk = _make_risk_score(status="degraded")
        p = self._base_patches(risk)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)
        self.assertIn("degraded", result["error"])

    def test_degraded_score_with_override_accepted(self):
        """Degraded score proceeds when override_reason is supplied."""
        risk = _make_risk_score(status="degraded")
        p = self._base_patches(risk)
        with patch.multiple(agent, **p):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       override_reason="analyst accepts degraded centrality",
                                       investigation_id=_INV_ID)
        self.assertIn("case_id", result)
        self.assertNotIn("error", result)


# ---------------------------------------------------------------------------
# New — incoming_flows tool
# ---------------------------------------------------------------------------

class TestIncomingFlows(unittest.TestCase):

    def test_incoming_flows_returns_inbound_rows(self):
        """incoming_flows query fires and returns correctly structured rows."""
        fake_row = {
            "account": "SENDER", "hops": 1,
            "path": ["SENDER", "TARGET"], "amounts": [500],
            "txn_ids": ["TX-IN-1"], "timestamps": ["2026-06-01"],
        }
        with patch.object(agent, "run_read", return_value=[fake_row]):
            result = agent.incoming_flows("TARGET")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["account"], "SENDER")

    def test_incoming_flows_hops_clamped(self):
        """max_hops outside [1,4] is clamped before being interpolated into Cypher."""
        captured = {}

        def fake_run_read(query, params=None):
            captured["query"] = query
            return []

        with patch.object(agent, "run_read", side_effect=fake_run_read):
            agent.incoming_flows("TARGET", max_hops=99)
        self.assertIn("*1..4", captured["query"])
        self.assertNotIn("99", captured["query"])


# ---------------------------------------------------------------------------
# New — _validate_evidence_ids helper
# ---------------------------------------------------------------------------

class TestValidateEvidenceIds(unittest.TestCase):

    def test_returns_ids_on_valid_input(self):
        rows = [{"id": "ACC-1", "labels": ["Account"]},
                {"id": "DEV-1", "labels": ["Device"]}]
        with patch.object(agent, "run_read", return_value=rows):
            ids, err = agent._validate_evidence_ids(["ACC-1", "DEV-1"])
        self.assertIsNone(err)
        self.assertEqual(set(ids), {"ACC-1", "DEV-1"})

    def test_per_id_error_for_missing_node(self):
        with patch.object(agent, "run_read", return_value=[]):
            _, err = agent._validate_evidence_ids(["MISSING-ID"])
        self.assertIsNotNone(err)
        self.assertIn("MISSING-ID", err["error"])
        self.assertIn("does not exist", err["error"])

    def test_ambiguous_node_rejected(self):
        """Two nodes with the same id (impossible in practice but guarded against)."""
        rows = [
            {"id": "DUP-ID", "labels": ["Account"]},
            {"id": "DUP-ID", "labels": ["Device"]},
        ]
        with patch.object(agent, "run_read", return_value=rows):
            _, err = agent._validate_evidence_ids(["DUP-ID"])
        self.assertIsNotNone(err)
        self.assertIn("ambiguous", err["error"])


# ---------------------------------------------------------------------------
# New — ensure_case_uniqueness_constraint
# ---------------------------------------------------------------------------

class TestCaseUniquenessConstraint(unittest.TestCase):

    def setUp(self):
        import agent as _agent
        _agent._CASE_CONSTRAINT_READY = False

    def test_constraint_failure_prevents_record_case_write(self):
        """If ensure_case_uniqueness_constraint raises, record_case returns an error."""
        label_rows = [{"id": "ACC-1", "labels": ["Account"]}]

        def fake_run_read(query, params=None):
            if "session_key" in query:
                return []
            if "WHERE n.id IN" in query:
                return label_rows
            return []

        def boom():
            raise RuntimeError("constraint not active")

        with patch.multiple(agent,
                            get_account=lambda a: [{"id": a}],
                            run_read=fake_run_read,
                            run_write=MagicMock(),
                            risk_score=MagicMock(return_value=_make_risk_score()),
                            ensure_case_uniqueness_constraint=boom):
            result = agent.record_case("ACC-1", "HIGH", "test", ["ACC-1"],
                                       investigation_id=_INV_ID)
        self.assertIn("error", result)
        self.assertIn("constraint", result["error"])

    def test_constraint_already_ready_skips_check(self):
        """With _CASE_CONSTRAINT_READY=True, run_read is not called for constraint check."""
        import agent as _agent
        _agent._CASE_CONSTRAINT_READY = True
        try:
            result = _agent.ensure_case_uniqueness_constraint()
            self.assertTrue(result)
        finally:
            _agent._CASE_CONSTRAINT_READY = False


# ---------------------------------------------------------------------------
# New — semantic pruning helpers
# ---------------------------------------------------------------------------

class TestSemanticPruning(unittest.TestCase):

    def test_summarize_extracts_risk_score_fields(self):
        """_summarize_dropped_messages extracts score/verdict_hint from a risk_score result."""
        dropped = [
            {"role": "tool", "tool_name": "risk_score",
             "content": json.dumps({"score": 77, "verdict_hint": "HIGH",
                                    "score_status": "complete", "breakdown": {}})},
        ]
        summary = agent._summarize_dropped_messages(dropped)
        self.assertIn("77", summary)
        self.assertIn("HIGH", summary)

    def test_summarize_extracts_account_ids_from_list(self):
        """Tool results that are account lists have their ids extracted."""
        dropped = [
            {"role": "tool", "tool_name": "shared_device_accounts",
             "content": json.dumps([{"account": "ACC-RING-01"},
                                    {"account": "ACC-RING-02"}])},
        ]
        summary = agent._summarize_dropped_messages(dropped)
        self.assertIn("ACC-RING-01", summary)

    def test_exchange_groups_keeps_assistant_tool_pairs(self):
        """_exchange_groups groups each assistant message with its following tool messages."""
        msgs = [
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "content": "r3"},
        ]
        groups = agent._exchange_groups(msgs)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 3)  # assistant + 2 tools
        self.assertEqual(len(groups[1]), 2)  # assistant + 1 tool


# ---------------------------------------------------------------------------
# New — propagation_risk (Monte Carlo)
# ---------------------------------------------------------------------------

class TestPropagationRisk(unittest.TestCase):

    def _fake_run_read(self, transfer_rows, device_rows=None, ip_rows=None):
        device_rows = device_rows or []
        ip_rows = ip_rows or []

        def fake(query, params=None):
            if "USED_DEVICE" in query:
                return device_rows
            if "LOGGED_FROM" in query:
                return ip_rows
            return transfer_rows  # transfer query

        return fake

    def test_empty_neighbors_returns_empty_list(self):
        with patch.object(agent, "run_read", side_effect=self._fake_run_read([])):
            result = agent.propagation_risk("ISOLATED")
        self.assertEqual(result["neighbors"], [])
        self.assertEqual(result["source"], "ISOLATED")

    def test_probabilities_are_in_range(self):
        transfer_rows = [
            {"neighbor": "B", "transfer_count": 2, "total_volume": 1000},
            {"neighbor": "C", "transfer_count": 1, "total_volume": 200},
        ]
        with patch.object(agent, "run_read",
                          side_effect=self._fake_run_read(transfer_rows)):
            result = agent.propagation_risk("A", n_simulations=50)
        self.assertEqual(len(result["neighbors"]), 2)
        for nbr in result["neighbors"]:
            self.assertGreaterEqual(nbr["infection_probability"], 0.0)
            self.assertLessEqual(nbr["infection_probability"], 1.0)
            self.assertEqual(nbr["simulations"], 50)

    def test_shared_device_raises_base_probability(self):
        """Shared device should push base_probability above transfer-only baseline."""
        transfer_rows = [{"neighbor": "B", "transfer_count": 1, "total_volume": 500}]
        device_rows = [{"neighbor": "B", "shared_devices": 1}]

        # No infrastructure
        with patch.object(agent, "run_read",
                          side_effect=self._fake_run_read(transfer_rows)):
            r_no_device = agent.propagation_risk("A", n_simulations=10)

        # With shared device
        with patch.object(agent, "run_read",
                          side_effect=self._fake_run_read(transfer_rows, device_rows)):
            r_with_device = agent.propagation_risk("A", n_simulations=10)

        p_no = r_no_device["neighbors"][0]["factors"]["base_probability"]
        p_yes = r_with_device["neighbors"][0]["factors"]["base_probability"]
        self.assertGreater(p_yes, p_no)

    def test_sorted_by_probability_descending(self):
        transfer_rows = [
            {"neighbor": "LOW", "transfer_count": 1, "total_volume": 100},
            {"neighbor": "HIGH", "transfer_count": 5, "total_volume": 5000},
        ]
        device_rows = [{"neighbor": "HIGH", "shared_devices": 1}]
        with patch.object(agent, "run_read",
                          side_effect=self._fake_run_read(transfer_rows, device_rows)):
            result = agent.propagation_risk("A", n_simulations=40)
        probs = [n["infection_probability"] for n in result["neighbors"]]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_n_simulations_clamped(self):
        """n_simulations < 10 is raised to 10; > 100 is capped at 100."""
        transfer_rows = [{"neighbor": "B", "transfer_count": 1, "total_volume": 100}]
        with patch.object(agent, "run_read",
                          side_effect=self._fake_run_read(transfer_rows)):
            r_low = agent.propagation_risk("A", n_simulations=1)
            r_high = agent.propagation_risk("A", n_simulations=999)
        self.assertEqual(r_low["simulations"], 10)
        self.assertEqual(r_high["simulations"], 100)


# ---------------------------------------------------------------------------
# New — max_flow_risk (Edmonds-Karp)
# ---------------------------------------------------------------------------

class TestMaxFlowRisk(unittest.TestCase):

    def test_simple_two_path_graph(self):
        """Classic max-flow: two disjoint paths from A to D."""
        # A→B (10), A→C (5), B→D (8), C→D (7)
        # Augmenting paths: A→B→D bottleneck 8; A→C→D bottleneck 5 → total 13
        edges = [
            {"src": "A", "dst": "B", "capacity": 10},
            {"src": "A", "dst": "C", "capacity": 5},
            {"src": "B", "dst": "D", "capacity": 8},
            {"src": "C", "dst": "D", "capacity": 7},
        ]
        with patch.object(agent, "run_read", return_value=edges):
            result = agent.max_flow_risk("A", "D")
        self.assertEqual(result["max_flow"], 13.0)
        self.assertEqual(result["source"], "A")
        self.assertEqual(result["sink"], "D")
        self.assertGreater(result["augmenting_paths"], 0)
        self.assertEqual(result["method"], "edmonds_karp")

    def test_no_path_returns_zero(self):
        """When sink is unreachable from source, max flow is 0."""
        edges = [{"src": "X", "dst": "Y", "capacity": 100}]
        with patch.object(agent, "run_read", return_value=edges):
            result = agent.max_flow_risk("A", "Z")
        self.assertEqual(result["max_flow"], 0.0)
        self.assertEqual(result["augmenting_paths"], 0)

    def test_empty_graph_returns_zero(self):
        with patch.object(agent, "run_read", return_value=[]):
            result = agent.max_flow_risk("A", "B")
        self.assertEqual(result["max_flow"], 0.0)
        self.assertEqual(result["edges_analyzed"], 0)

    def test_single_edge_bottleneck(self):
        """Single A→B edge: max flow equals its capacity."""
        edges = [{"src": "A", "dst": "B", "capacity": 9500}]
        with patch.object(agent, "run_read", return_value=edges):
            result = agent.max_flow_risk("A", "B")
        self.assertEqual(result["max_flow"], 9500.0)
        self.assertEqual(result["augmenting_paths"], 1)

    def test_parallel_edges_summed(self):
        """Multiple transfer records on the same pair should sum as capacity."""
        edges = [
            {"src": "A", "dst": "B", "capacity": 3000},
            {"src": "A", "dst": "B", "capacity": 2000},
        ]
        with patch.object(agent, "run_read", return_value=edges):
            result = agent.max_flow_risk("A", "B")
        self.assertEqual(result["max_flow"], 5000.0)

    def test_edges_analyzed_reported(self):
        edges = [{"src": "A", "dst": "B", "capacity": 100} for _ in range(5)]
        with patch.object(agent, "run_read", return_value=edges):
            result = agent.max_flow_risk("A", "B")
        self.assertEqual(result["edges_analyzed"], 5)


if __name__ == "__main__":
    unittest.main()

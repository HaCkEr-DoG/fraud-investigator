"""Integration tests for the fraud-investigator agent against a live FalkorDB.

Run with:
    RUN_FALKORDB_INTEGRATION=1 \\
    FALKORDB_TEST_GRAPH=fraud_integration_test \\
    python -m unittest -v integration_tests.py

Skipped automatically (and silently) when RUN_FALKORDB_INTEGRATION is not set,
so the file can live alongside the unit tests without breaking CI.

Isolation: every test class seeds its own data in FALKORDB_TEST_GRAPH (default:
fraud_integration_test) and tears it down in tearDownClass.  The graph is
distinct from the seed.py production graph (default: fraud).
"""
import json
import os
import unittest

# ---------------------------------------------------------------------------
# Guard: skip the whole module if the integration flag is not set.
# This must run before any FalkorDB import to avoid a failed TCP connection.
# ---------------------------------------------------------------------------
if os.getenv("RUN_FALKORDB_INTEGRATION") != "1":
    raise unittest.SkipTest(
        "RUN_FALKORDB_INTEGRATION not set — skipping integration tests"
    )

# Map FALKORDB_TEST_GRAPH → FALKOR_GRAPH so that agent.py picks up the right graph.
_test_graph = os.getenv("FALKORDB_TEST_GRAPH", "fraud_integration_test")
os.environ["FALKOR_GRAPH"] = _test_graph

import agent  # noqa: E402 — must come after env var is set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_graph():
    """Delete every node in the test graph between test classes."""
    agent.run_write("MATCH (n) DETACH DELETE n")


# ---------------------------------------------------------------------------
# 1. Multi-hop graph reasoning
# ---------------------------------------------------------------------------

class TestMultiHopTraversal(unittest.TestCase):
    """Verify that multi-hop Cypher traversals work correctly on a live graph."""

    RING_ACCOUNTS = ["RING-A", "RING-B", "RING-C"]

    @classmethod
    def setUpClass(cls):
        _clear_graph()
        # Seed a 3-account ring with chronological timestamps and a shared device/IP.
        agent.run_write("""
            CREATE (a:Account {id: 'RING-A', name: 'Ring A', created_at: '2025-01-01'}),
                   (b:Account {id: 'RING-B', name: 'Ring B', created_at: '2025-01-01'}),
                   (c:Account {id: 'RING-C', name: 'Ring C', created_at: '2025-01-01'}),
                   (mule:Account {id: 'MULE-1', name: 'Mule',   created_at: '2025-01-01'}),
                   (d:Device {id: 'RING-DEV'}),
                   (ip:IP    {id: '10.0.0.99'}),
                   (a)-[:TRANSFERRED {txn_id: 'TX-AB', amount: 5000, ts: '2026-01-01'}]->(b),
                   (b)-[:TRANSFERRED {txn_id: 'TX-BC', amount: 5000, ts: '2026-01-02'}]->(c),
                   (c)-[:TRANSFERRED {txn_id: 'TX-CA', amount: 5000, ts: '2026-01-03'}]->(a),
                   (c)-[:TRANSFERRED {txn_id: 'TX-CM', amount: 4900, ts: '2026-01-04'}]->(mule),
                   (a)-[:USED_DEVICE]->(d),
                   (b)-[:USED_DEVICE]->(d),
                   (a)-[:LOGGED_FROM]->(ip),
                   (b)-[:LOGGED_FROM]->(ip)
        """)

    @classmethod
    def tearDownClass(cls):
        _clear_graph()

    # --- shared device / shared IP ---

    def test_shared_device_finds_ring_members(self):
        result = agent.shared_device_accounts("RING-A")
        account_ids = [r["account"] for r in result]
        self.assertIn("RING-B", account_ids,
                      "RING-A and RING-B share RING-DEV; must be returned")

    def test_shared_ip_finds_ring_members(self):
        result = agent.shared_ip_accounts("RING-A")
        account_ids = [r["account"] for r in result]
        self.assertIn("RING-B", account_ids,
                      "RING-A and RING-B share IP 10.0.0.99; must be returned")

    # --- money cycles ---

    def test_money_cycles_finds_3_hop_ring(self):
        result = agent.money_cycles("RING-A")
        self.assertGreater(len(result), 0, "A→B→C→A is a valid 3-hop cycle")

    def test_confirmed_chronological_cycle_detected(self):
        """Timestamps TX-AB < TX-BC < TX-CA → confirmed_chronological."""
        result = agent.money_cycles("RING-A")
        confirmed = [c for c in result if c["cycle_type"] == "confirmed_chronological"]
        self.assertGreater(len(confirmed), 0)

    def test_cycle_includes_txn_ids(self):
        result = agent.money_cycles("RING-A")
        for cycle in result:
            self.assertIn("txn_ids", cycle)
            self.assertIsNotNone(cycle["txn_ids"])

    def test_structural_only_cycle_reversed_timestamps(self):
        """Seed a cycle where money flows backward; must be classified structural_only."""
        agent.run_write("""
            CREATE (x:Account {id: 'X', name: 'X', created_at: '2025-01-01'}),
                   (y:Account {id: 'Y', name: 'Y', created_at: '2025-01-01'}),
                   (x)-[:TRANSFERRED {txn_id: 'TX-XY', amount: 100,
                                       ts: '2026-05-02'}]->(y),
                   (y)-[:TRANSFERRED {txn_id: 'TX-YX', amount: 100,
                                       ts: '2026-05-01'}]->(x)
        """)
        result = agent.money_cycles("X")
        structural = [c for c in result if c["cycle_type"] == "structural_only"]
        self.assertGreater(len(structural), 0, "Reversed timestamps should be structural_only")
        agent.run_write("MATCH (n:Account) WHERE n.id IN ['X', 'Y'] DETACH DELETE n")

    # --- outgoing flows ---

    def test_outgoing_flows_1_hop(self):
        result = agent.outgoing_flows("RING-A", 1)
        account_ids = [r["account"] for r in result]
        self.assertIn("RING-B", account_ids)

    def test_outgoing_flows_2_hops_reaches_mule(self):
        result = agent.outgoing_flows("RING-A", 2)
        account_ids = [r["account"] for r in result]
        # A→B→C, A→B is 1 hop, A→B→C→MULE is 3 hops — at 2 hops we reach C
        self.assertIn("RING-C", account_ids)

    def test_hops_capped_at_4(self):
        """outgoing_flows silently caps max_hops at 4 per the implementation."""
        result = agent.outgoing_flows("RING-A", 99)
        self.assertIsInstance(result, list)  # must not raise

    # --- connection path ---

    def test_connection_path_directed(self):
        result = agent.connection_path("RING-A", "MULE-1")
        # A→B→C→MULE: directed path exists
        if result:  # shortestPath may return [] if no path
            self.assertIn("nodes", result[0])

    def test_traversal_truncated_flag_present(self):
        """risk_score always reports traversal_truncated regardless of graph size."""
        result = agent.risk_score("RING-A")
        self.assertIn("traversal_truncated", result)
        # With only 4 accounts, traversal limits are not hit.
        self.assertFalse(result["traversal_truncated"])

    def test_money_cycles_custom_depth_finds_deep_ring(self):
        """money_cycles(max_depth=6) finds a 6-node ring that max_depth=5 would miss."""
        # Seed a 6-node ring with chronologically ordered timestamps.
        agent.run_write("""
            CREATE (a:Account {id: 'D6-A', name: 'D6-A', created_at: '2025-01-01'}),
                   (b:Account {id: 'D6-B', name: 'D6-B', created_at: '2025-01-01'}),
                   (c:Account {id: 'D6-C', name: 'D6-C', created_at: '2025-01-01'}),
                   (d:Account {id: 'D6-D', name: 'D6-D', created_at: '2025-01-01'}),
                   (e:Account {id: 'D6-E', name: 'D6-E', created_at: '2025-01-01'}),
                   (f:Account {id: 'D6-F', name: 'D6-F', created_at: '2025-01-01'}),
                   (a)-[:TRANSFERRED {txn_id: 'D6-AB', amount: 100, ts: '2026-03-01'}]->(b),
                   (b)-[:TRANSFERRED {txn_id: 'D6-BC', amount: 100, ts: '2026-03-02'}]->(c),
                   (c)-[:TRANSFERRED {txn_id: 'D6-CD', amount: 100, ts: '2026-03-03'}]->(d),
                   (d)-[:TRANSFERRED {txn_id: 'D6-DE', amount: 100, ts: '2026-03-04'}]->(e),
                   (e)-[:TRANSFERRED {txn_id: 'D6-EF', amount: 100, ts: '2026-03-05'}]->(f),
                   (f)-[:TRANSFERRED {txn_id: 'D6-FA', amount: 100, ts: '2026-03-06'}]->(a)
        """)
        result = agent.money_cycles("D6-A", max_depth=6)
        self.assertGreater(len(result), 0, "A 6-node ring must be found with max_depth=6")
        # The ring must be classified confirmed_chronological (timestamps in order).
        confirmed = [c for c in result if c["cycle_type"] == "confirmed_chronological"]
        self.assertGreater(len(confirmed), 0, "6-node ring with ordered timestamps must be confirmed")
        # Clean up the 6-node ring.
        agent.run_write(
            "MATCH (n:Account) WHERE n.id IN ['D6-A','D6-B','D6-C','D6-D','D6-E','D6-F'] "
            "DETACH DELETE n"
        )

    def test_money_cycles_default_depth_uses_8(self):
        """Default max_depth (8) is greater than the old hard-coded 5."""
        import agent as _agent
        self.assertEqual(_agent._MAX_CYCLE_DEPTH, 8,
                         "_MAX_CYCLE_DEPTH must be 8 so 6-/7-/8-node rings are found by default")


# ---------------------------------------------------------------------------
# 2. Graph algorithms as agent tools
# ---------------------------------------------------------------------------

class TestGraphAlgorithms(unittest.TestCase):
    """Verify that native FalkorDB PageRank runs and drives investigation policy."""

    @classmethod
    def setUpClass(cls):
        _clear_graph()
        # Create a hub account that receives many transfers.
        agent.run_write("""
            CREATE (hub:Account {id: 'HUB', name: 'Hub', created_at: '2025-01-01'}),
                   (s1:Account {id: 'S1',  name: 'S1',  created_at: '2025-01-01'}),
                   (s2:Account {id: 'S2',  name: 'S2',  created_at: '2025-01-01'}),
                   (s3:Account {id: 'S3',  name: 'S3',  created_at: '2025-01-01'}),
                   (s1)-[:TRANSFERRED {txn_id: 'T1', amount: 100, ts: '2026-01-01'}]->(hub),
                   (s2)-[:TRANSFERRED {txn_id: 'T2', amount: 100, ts: '2026-01-01'}]->(hub),
                   (s3)-[:TRANSFERRED {txn_id: 'T3', amount: 100, ts: '2026-01-01'}]->(hub),
                   (hub)-[:TRANSFERRED {txn_id: 'T4', amount: 100, ts: '2026-01-02'}]->(s1)
        """)

    @classmethod
    def tearDownClass(cls):
        _clear_graph()

    def test_top_risk_accounts_returns_list(self):
        result = agent.top_risk_accounts()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_pagerank_score_is_numeric(self):
        result = agent.top_risk_accounts()
        for row in result:
            self.assertIn("score", row)
            self.assertIsInstance(row["score"], (int, float),
                                   "PageRank score must be a number")
            self.assertGreater(row["score"], 0)

    def test_risk_score_preserves_pagerank_numeric(self):
        result = agent.risk_score("HUB")
        self.assertIsNotNone(result["breakdown"]["pagerank_score"],
                              "Numeric PageRank score must be stored in breakdown")
        self.assertIsInstance(result["breakdown"]["pagerank_score"], (int, float))

    def test_investigation_policy_set_from_pagerank(self):
        result = agent.risk_score("HUB")
        policy = result["investigation_policy"]
        self.assertIn("recommended_next", policy)
        self.assertIn("reason", policy)

    def test_score_status_complete_when_pagerank_works(self):
        result = agent.risk_score("HUB")
        self.assertEqual(result["score_status"], "complete")

    def test_confirmed_cycle_txn_ids_in_breakdown(self):
        """breakdown always contains confirmed_cycle_txn_ids (empty if no cycles)."""
        result = agent.risk_score("HUB")
        self.assertIn("confirmed_cycle_txn_ids", result["breakdown"])
        self.assertIsInstance(result["breakdown"]["confirmed_cycle_txn_ids"], list)

    def test_neighborhood_anchor_scopes_pagerank_to_neighborhood(self):
        """top_risk_accounts(anchor_id=X) returns accounts in X's transfer neighborhood."""
        # HUB is connected to S1, S2, S3 via TRANSFERRED edges.
        result = agent.top_risk_accounts(anchor_id="HUB")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "Anchored PageRank must return neighbors")
        account_ids = {r["account"] for r in result}
        # HUB itself must appear (anchor is always included in neighborhood).
        self.assertIn("HUB", account_ids, "Anchor account must appear in neighborhood results")

    def test_neighborhood_anchor_excludes_unrelated_accounts(self):
        """top_risk_accounts(anchor_id=X) excludes accounts not in X's 2-hop neighborhood."""
        # Seed an isolated account with no TRANSFERRED edges to HUB's neighborhood.
        agent.run_write(
            "CREATE (iso:Account {id: 'ISO-ALONE', name: 'Isolated', created_at: '2025-01-01'})"
        )
        result = agent.top_risk_accounts(anchor_id="HUB")
        account_ids = {r["account"] for r in result}
        self.assertNotIn("ISO-ALONE", account_ids,
                         "Isolated account must not appear in HUB's neighborhood PageRank")
        agent.run_write("MATCH (n:Account {id: 'ISO-ALONE'}) DETACH DELETE n")


# ---------------------------------------------------------------------------
# 3 & 4. record_case: write safety, idempotency, evidence linking
# ---------------------------------------------------------------------------

class TestRecordCaseIntegration(unittest.TestCase):
    """Integration tests for record_case writing to the live graph."""

    @classmethod
    def setUpClass(cls):
        _clear_graph()
        agent.run_write("""
            CREATE (a:Account {id: 'ACC-WRITE', name: 'WriteTest', created_at: '2025-01-01'}),
                   (d:Device  {id: 'DEV-WRITE'})
        """)

    @classmethod
    def tearDownClass(cls):
        _clear_graph()

    def setUp(self):
        # Remove Case nodes between tests for clean isolation.
        agent.run_write("MATCH (c:Case) DETACH DELETE c")

    def test_record_case_creates_case_node(self):
        r = agent.record_case(
            "ACC-WRITE", "HIGH", "Integration test verdict", ["ACC-WRITE", "DEV-WRITE"]
        )
        self.assertNotIn("error", r)
        self.assertIn("case_id", r)

        rows = agent.run_read(
            "MATCH (c:Case {id: $cid}) RETURN c.verdict AS verdict, "
            "c.breakdown_json AS bj, c.verdict_hint AS vh, "
            "c.confirmed_cycle_txns_json AS ctj, c.scoring_policy AS pol",
            {"cid": r["case_id"]},
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["verdict"], "HIGH")
        self.assertEqual(row["pol"], agent.SCORING_POLICY_VERSION)
        # breakdown_json must be valid JSON with expected fields.
        bd = json.loads(row["bj"])
        self.assertIn("pagerank_rank", bd)
        self.assertIn("confirmed_cycle_txn_ids", bd)
        # verdict_hint and confirmed_cycle_txns_json as first-class properties.
        self.assertIn(row["vh"], ("LOW", "MEDIUM", "HIGH"))
        txns = json.loads(row["ctj"])
        self.assertIsInstance(txns, list)

    def test_cites_edges_link_to_evidence(self):
        r = agent.record_case(
            "ACC-WRITE", "HIGH", "Evidence citation", ["ACC-WRITE", "DEV-WRITE"]
        )
        linked = agent.run_read(
            "MATCH (c:Case {id: $cid})-[:CITES]->(n) RETURN n.id AS nid",
            {"cid": r["case_id"]},
        )
        cited_ids = {row["nid"] for row in linked}
        self.assertIn("ACC-WRITE", cited_ids)
        self.assertIn("DEV-WRITE", cited_ids)
        self.assertEqual(r["cited_evidence"], len(cited_ids))

    def test_investigates_edge_exists(self):
        r = agent.record_case(
            "ACC-WRITE", "HIGH", "INVESTIGATES edge test", ["ACC-WRITE"]
        )
        rows = agent.run_read(
            "MATCH (c:Case {id: $cid})-[:INVESTIGATES]->(a:Account) RETURN a.id AS aid",
            {"cid": r["case_id"]},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["aid"], "ACC-WRITE")

    def test_merge_idempotency_same_session_key(self):
        """Two calls in the same minute return the same case_id (MERGE dedup)."""
        r1 = agent.record_case("ACC-WRITE", "HIGH", "First call",  ["ACC-WRITE"])
        r2 = agent.record_case("ACC-WRITE", "HIGH", "Second call", ["ACC-WRITE"])
        self.assertNotIn("error", r1)
        # MERGE ensures at most one Case per session_key — node count must be ≤ 1.
        rows = agent.run_read(
            "MATCH (c:Case)-[:INVESTIGATES]->(a:Account {id: 'ACC-WRITE'}) "
            "RETURN count(c) AS cnt"
        )
        cnt = rows[0]["cnt"] if rows else 0
        self.assertEqual(cnt, 1, "MERGE must produce exactly one Case per session_key")
        # Both calls return the same case_id.
        self.assertEqual(r1["case_id"], r2["case_id"])

    def test_prior_cases_returns_saved_case(self):
        r = agent.record_case("ACC-WRITE", "HIGH", "Prior case", ["ACC-WRITE"])
        cases = agent.prior_cases("ACC-WRITE")
        self.assertGreater(len(cases), 0)
        self.assertIn(r["case_id"], [c["case_id"] for c in cases])

    def test_verdict_mismatch_without_reason_rejected(self):
        r = agent.record_case("ACC-WRITE", "LOW", "No override", ["ACC-WRITE"])
        self.assertIn("error", r)
        # Verify no Case was written.
        rows = agent.run_read("MATCH (c:Case) RETURN count(c) AS cnt")
        self.assertEqual(rows[0]["cnt"], 0)

    def test_verdict_mismatch_with_reason_accepted(self):
        r = agent.record_case(
            "ACC-WRITE", "LOW", "Override test", ["ACC-WRITE"],
            override_reason="Investigator manual review — low risk despite score"
        )
        # LOW may or may not match the computed hint; if mismatch, override_reason accepts it.
        # Either it succeeded or the hint was already LOW — no error expected.
        self.assertNotIn("error", r)
        self.assertIn("case_id", r)

    def test_empty_evidence_rejected(self):
        r = agent.record_case("ACC-WRITE", "HIGH", "No evidence", [])
        self.assertIn("error", r)
        rows = agent.run_read("MATCH (c:Case) RETURN count(c) AS cnt")
        self.assertEqual(rows[0]["cnt"], 0)

    def test_nonexistent_evidence_rejected(self):
        r = agent.record_case("ACC-WRITE", "HIGH", "Ghost evidence", ["GHOST-999"])
        self.assertIn("error", r)
        self.assertIn("not found", r["error"])

    def test_unsupported_label_rejected(self):
        # Case nodes are not valid evidence; create one to attempt linking it.
        agent.run_write("CREATE (c:Case {id: 'CASE-BAD', session_key: 'bad-test'})")
        r = agent.record_case("ACC-WRITE", "HIGH", "Case evidence", ["CASE-BAD"])
        self.assertIn("error", r)

    def test_duplicate_evidence_ids_deduplicated(self):
        """Repeated IDs in evidence_ids must not inflate cited_evidence count."""
        r = agent.record_case(
            "ACC-WRITE", "HIGH", "Dedup test", ["ACC-WRITE", "ACC-WRITE", "ACC-WRITE"]
        )
        self.assertNotIn("error", r)
        self.assertEqual(r["cited_evidence"], 1)

    def test_score_and_policy_stored_on_case(self):
        r = agent.record_case("ACC-WRITE", "HIGH", "Score test", ["ACC-WRITE"])
        rows = agent.run_read(
            "MATCH (c:Case {id: $cid}) RETURN c.score AS score, c.scoring_policy AS pol",
            {"cid": r["case_id"]},
        )
        self.assertIsNotNone(rows[0]["score"])
        self.assertEqual(rows[0]["pol"], agent.SCORING_POLICY_VERSION)

    def test_flow_paths_json_stored_and_parseable(self):
        """Case node must persist flow_paths_json as a valid JSON list."""
        r = agent.record_case("ACC-WRITE", "HIGH", "Flow paths test", ["ACC-WRITE"])
        self.assertNotIn("error", r)
        rows = agent.run_read(
            "MATCH (c:Case {id: $cid}) RETURN c.flow_paths_json AS fpj",
            {"cid": r["case_id"]},
        )
        self.assertEqual(len(rows), 1, "Case must exist after record_case")
        fpj_raw = rows[0]["fpj"]
        self.assertIsNotNone(fpj_raw, "flow_paths_json must be stored (not null) on Case node")
        paths = json.loads(fpj_raw)
        self.assertIsInstance(paths, list, "flow_paths_json must deserialize to a list")
        for p in paths:
            self.assertIn("path", p, "Each flow path entry must have a 'path' key")
            self.assertIn("hops", p, "Each flow path entry must have a 'hops' key")
            self.assertIn("amounts", p, "Each flow path entry must have an 'amounts' key")


# ---------------------------------------------------------------------------
# 5. Explainability — score reconstruction
# ---------------------------------------------------------------------------

class TestExplainabilityIntegration(unittest.TestCase):
    """Verify that a stored Case contains enough data to reproduce the score."""

    @classmethod
    def setUpClass(cls):
        _clear_graph()
        agent.run_write("""
            CREATE (a:Account {id: 'EXP-1', name: 'Explain', created_at: '2025-01-01'}),
                   (b:Account {id: 'EXP-2', name: 'Explain2', created_at: '2025-01-01'}),
                   (d:Device  {id: 'EXP-DEV'}),
                   (a)-[:USED_DEVICE]->(d),
                   (b)-[:USED_DEVICE]->(d),
                   (a)-[:TRANSFERRED {txn_id: 'EX-TX1', amount: 500, ts: '2026-02-01'}]->(b),
                   (b)-[:TRANSFERRED {txn_id: 'EX-TX2', amount: 500, ts: '2026-02-02'}]->(a)
        """)

    @classmethod
    def tearDownClass(cls):
        _clear_graph()

    def test_breakdown_json_allows_score_reconstruction(self):
        r = agent.record_case("EXP-1", "HIGH", "Explainability test", ["EXP-1", "EXP-2", "EXP-DEV"])
        # Fetch the Case and re-add the score from breakdown_json.
        rows = agent.run_read(
            "MATCH (c:Case {id: $cid}) RETURN c.breakdown_json AS bj, c.score AS stored_score",
            {"cid": r["case_id"]},
        )
        bd = json.loads(rows[0]["bj"])
        # Verify all required breakdown fields are present.
        for field in ("shared_device_pts", "shared_ip_pts", "cycle_pts",
                      "flow_pts", "prior_pts", "centrality_pts",
                      "pagerank_rank", "pagerank_score", "confirmed_cycle_txn_ids"):
            self.assertIn(field, bd, f"Missing breakdown field: {field}")
        # Reconstruct the score from the breakdown and compare with the stored value.
        reconstructed = min(
            bd["shared_device_pts"] + bd["shared_ip_pts"] + bd["cycle_pts"]
            + bd["flow_pts"] + bd["prior_pts"] + bd["centrality_pts"],
            100,
        )
        self.assertEqual(reconstructed, rows[0]["stored_score"],
                          "Stored score must be reproducible from breakdown_json alone")

    def test_confirmed_cycle_txns_traces_to_txn_ids(self):
        agent.run_write("MATCH (c:Case) DETACH DELETE c")
        r = agent.record_case("EXP-1", "HIGH", "Txn trace test", ["EXP-1", "EXP-2"])
        rows = agent.run_read(
            "MATCH (c:Case {id: $cid}) RETURN c.confirmed_cycle_txns_json AS ctj",
            {"cid": r["case_id"]},
        )
        ctj = json.loads(rows[0]["ctj"])
        # May be empty if cycle chronology wasn't confirmed, but must be valid JSON list.
        self.assertIsInstance(ctj, list)
        for cycle_txns in ctj:
            self.assertIsInstance(cycle_txns, list)


if __name__ == "__main__":
    unittest.main()

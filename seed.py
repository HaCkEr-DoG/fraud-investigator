"""Generate a synthetic payments graph with one planted fraud ring.

Run once before the agent:  python seed.py
All data is synthetic. No real customers, devices, or transactions.
"""
import os
import random
from datetime import datetime, timedelta, timezone

from falkordb import FalkorDB

random.seed(7)

HOST = os.getenv("FALKORDB_HOST", "localhost")
PORT = int(os.getenv("FALKORDB_PORT", "6379"))
GRAPH_NAME = os.getenv("FALKOR_GRAPH", "fraud")


def main():
    graph = FalkorDB(host=HOST, port=PORT).select_graph(GRAPH_NAME)
    try:
        graph.delete()  # start clean; errors if the graph does not exist yet
    except Exception:
        pass

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    legit = [f"ACC-{i:03d}" for i in range(60)]
    ring = [f"ACC-RING-{i:02d}" for i in range(1, 6)]
    mule = "ACC-MULE-01"
    accounts = legit + ring + [mule]

    devices = [f"DEV-{i:03d}" for i in range(40)] + ["DEV-RING-A", "DEV-RING-B"]
    ips = [f"10.0.{i // 16}.{i % 16 + 1}" for i in range(40)] + ["203.0.113.50"]

    # intern() converts string properties to a shared pointer at write time.
    # WHERE clauses on id, fingerprint, and IP address then compare pointers
    # rather than scanning character-by-character — ~75x faster on categorical filters.
    graph.query(
        "UNWIND $rows AS r CREATE (:Account {id: intern(r.id), name: r.name, created_at: r.created_at})",
        {"rows": [
            {"id": a, "name": f"Customer {a}",
             "created_at": (start + timedelta(days=random.randint(0, 300))).isoformat()}
            for a in accounts
        ]},
    )
    graph.query(
        "UNWIND $rows AS r MERGE (:Device {id: intern(r.id), fingerprint: intern(r.fp)})",
        {"rows": [{"id": d, "fp": f"fp-{d}"} for d in devices]},
    )
    graph.query(
        "UNWIND $rows AS r MERGE (:IP {id: intern(r.id)})",
        {"rows": [{"id": ip} for ip in ips]},
    )

    # Device and IP links. Legit customers use 1-2 random devices and 1 IP.
    dev_links, ip_links = [], []
    for a in legit:
        for d in random.sample(devices[:40], random.choice([1, 2])):
            dev_links.append({"acc": a, "dev": d})
        ip_links.append({"acc": a, "ip": random.choice(ips[:40])})

    # Planted ring: all five ring accounts share two devices and one IP.
    for a in ring:
        dev_links.append({"acc": a, "dev": "DEV-RING-A"})
        dev_links.append({"acc": a, "dev": "DEV-RING-B"})
        ip_links.append({"acc": a, "ip": "203.0.113.50"})
    # The mule reuses one ring device, which is the link the agent should find.
    dev_links.append({"acc": mule, "dev": "DEV-RING-B"})
    ip_links.append({"acc": mule, "ip": ips[3]})

    graph.query(
        "UNWIND $rows AS r MATCH (a:Account {id: r.acc}), (d:Device {id: r.dev}) "
        "MERGE (a)-[:USED_DEVICE]->(d)",
        {"rows": dev_links},
    )
    graph.query(
        "UNWIND $rows AS r MATCH (a:Account {id: r.acc}), (i:IP {id: r.ip}) "
        "MERGE (a)-[:LOGGED_FROM]->(i)",
        {"rows": ip_links},
    )

    # Transfers. Background traffic among legit customers.
    txns = []
    for n in range(150):
        src, dst = random.sample(legit, 2)
        txns.append({
            "src": src, "dst": dst, "txn": f"TX-{n:04d}",
            "amount": round(random.uniform(20, 400), 2),
            "ts": (start + timedelta(days=random.randint(0, 300))).isoformat(),
        })
    # Ring cycles money around in just-under-threshold amounts (structuring).
    for i, src in enumerate(ring):
        dst = ring[(i + 1) % len(ring)]
        txns.append({"src": src, "dst": dst, "txn": f"TX-RING-{i}",
                     "amount": 9_400 + i * 20, "ts": "2026-09-01T10:00:00+00:00"})
    # The mule receives from the ring, then splits the money out to legit accounts.
    txns.append({"src": ring[2], "dst": mule, "txn": "TX-MULE-IN",
                 "amount": 9_500, "ts": "2026-09-02T09:00:00+00:00"})
    txns.append({"src": mule, "dst": "ACC-007", "txn": "TX-MULE-OUT-1",
                 "amount": 9_200, "ts": "2026-09-02T15:00:00+00:00"})
    txns.append({"src": mule, "dst": "ACC-008", "txn": "TX-MULE-OUT-2",
                 "amount": 250, "ts": "2026-09-02T15:05:00+00:00"})

    graph.query(
        "UNWIND $rows AS r MATCH (a:Account {id: r.src}), (b:Account {id: r.dst}) "
        "CREATE (a)-[:TRANSFERRED {txn_id: r.txn, amount: r.amount, ts: r.ts}]->(b)",
        {"rows": txns},
    )

    print(f"Seeded graph '{GRAPH_NAME}': {len(accounts)} accounts, "
          f"{len(devices)} devices, {len(ips)} IPs, {len(txns)} transfers.")
    print("Try: python agent.py ACC-RING-01")


if __name__ == "__main__":
    main()

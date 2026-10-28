"""The exact HGP query (Section 4) on SQLite and DuckDB.

Tables
  Incidence(unit_id, node_id)                       held by the producer
  Attribute(node_id, attr, group_val)               held by the auditor; a missing value is one more group
  Partition(node_id, cluster_id)                    held by the auditor
  Contingency(unit_id, attr, group_val, cluster_id, n)   the contingency table, if released directly
The query groups the joined rows by (unit, group, cluster) to obtain n_{e,g,C}, sums them by unit, by unit and
group, and by unit and cluster, and returns for every unit size k the number of units and the integer sum of
|e|^2 psi_e = sum_{g,C} max(|e| n_{e,g,C} - n_{e,g} n_{e,C}, 0). Two integers per unit size fix HGP exactly.
"""
from __future__ import annotations

import sqlite3
from fractions import Fraction

import numpy as np
import pandas as pd

DDL = ["CREATE TABLE Incidence(unit_id INTEGER NOT NULL, node_id INTEGER NOT NULL)",
       "CREATE TABLE Attribute(node_id INTEGER NOT NULL, attr TEXT NOT NULL, group_val TEXT NOT NULL)",
       "CREATE TABLE Partition(node_id INTEGER NOT NULL PRIMARY KEY, cluster_id INTEGER NOT NULL)"]

# n_{e,g,C} from incidence, attributes and partition
CONTINGENCY = """SELECT i.unit_id AS unit_id, a.attr AS attr, a.group_val AS group_val, p.cluster_id AS cluster_id,
       COUNT(*) AS n
FROM Incidence i JOIN Attribute a ON a.node_id = i.node_id JOIN Partition p ON p.node_id = i.node_id
WHERE a.attr = ? GROUP BY i.unit_id, a.attr, a.group_val, p.cluster_id"""

# per-size counts and sums from n_{e,g,C}
_TAIL = """,
u AS (SELECT e, SUM(n) AS k FROM s GROUP BY e),
gm AS (SELECT e, g, SUM(n) AS ng FROM s GROUP BY e, g),
cm AS (SELECT e, c, SUM(n) AS nc FROM s GROUP BY e, c)
SELECT u.k AS k, COUNT(DISTINCT s.e) AS units,
       SUM(CASE WHEN u.k * s.n > gm.ng * cm.nc THEN u.k * s.n - gm.ng * cm.nc ELSE 0 END) AS sum_y
FROM s JOIN u ON u.e = s.e JOIN gm ON gm.e = s.e AND gm.g = s.g JOIN cm ON cm.e = s.e AND cm.c = s.c
GROUP BY u.k ORDER BY u.k"""
HGP_FROM_INCIDENCE = """WITH s AS (SELECT i.unit_id AS e, a.group_val AS g, p.cluster_id AS c, COUNT(*) AS n
  FROM Incidence i JOIN Attribute a ON a.node_id = i.node_id JOIN Partition p ON p.node_id = i.node_id
  WHERE a.attr = ? GROUP BY 1, 2, 3)""" + _TAIL
HGP_FROM_CONTINGENCY = """WITH s AS (SELECT unit_id AS e, group_val AS g, cluster_id AS c, n
  FROM Contingency WHERE attr = ? AND n > 0)""" + _TAIL

# U_P from a released P(u, v, w) stored for u <= v (diagonal once)
UP_SCAN = """SELECT SUM(CASE WHEN lu.g <> lv.g AND lu.c <> lv.c THEN P.w ELSE 0 END),
       SUM(CASE WHEN P.u = P.v THEN P.w ELSE 2 * P.w END)
FROM P JOIN Lab lu ON lu.node_id = P.u JOIN Lab lv ON lv.node_id = P.v"""


def connect(engine: str):
    if engine == "sqlite":
        return sqlite3.connect(":memory:")
    if engine == "duckdb":
        import duckdb
        return duckdb.connect(":memory:")
    raise ValueError(engine)


def _fetch(con, sql, params=()):
    cur = con.execute(sql, params) if params else con.execute(sql)
    return cur.fetchall()


def load_tables(con, H, attr_name: str, attr_values: pd.Series, clusters: np.ndarray):
    for d in DDL:
        con.execute(d)
    unit = np.repeat(np.arange(H.shape[0], dtype=np.int64), np.diff(H.indptr))
    inc = pd.DataFrame({"unit_id": unit, "node_id": H.indices.astype(np.int64)})
    att = pd.DataFrame({"node_id": np.arange(H.shape[1]), "attr": attr_name,
                        "group_val": attr_values.fillna("__missing__").astype(str).to_numpy()})
    par = pd.DataFrame({"node_id": np.arange(H.shape[1]), "cluster_id": clusters.astype(np.int64)})
    if isinstance(con, sqlite3.Connection):
        con.executemany("INSERT INTO Incidence VALUES (?, ?)", inc.itertuples(index=False, name=None))
        con.executemany("INSERT INTO Attribute VALUES (?, ?, ?)", att.itertuples(index=False, name=None))
        con.executemany("INSERT INTO Partition VALUES (?, ?)", par.itertuples(index=False, name=None))
    else:
        for name, df in (("Incidence", inc), ("Attribute", att), ("Partition", par)):
            con.register("df_tmp", df)
            con.execute(f"INSERT INTO {name} SELECT * FROM df_tmp")
            con.unregister("df_tmp")


def per_size_counts_and_sums(engine, H, attr_values: pd.Series, clusters, attr_name="A"):
    """[(k, units of size k, sum of |e|^2 psi_e over them)] from the SQL query."""
    con = connect(engine)
    load_tables(con, H, attr_name, attr_values, clusters)
    rows = _fetch(con, HGP_FROM_INCIDENCE, (attr_name,))
    con.close()
    return [(int(k), int(u), int(y)) for k, u, y in rows]


def contingency_table(engine, H, attr_values, clusters, attr_name="A") -> pd.DataFrame:
    con = connect(engine)
    load_tables(con, H, attr_name, attr_values, clusters)
    rows = _fetch(con, CONTINGENCY, (attr_name,))
    con.close()
    return pd.DataFrame(rows, columns=["unit_id", "attr", "group_val", "cluster_id", "n"])


def hgp_from_contingency(engine, table: pd.DataFrame, attr_name="A"):
    con = connect(engine)
    con.execute("CREATE TABLE Contingency(unit_id INTEGER, attr TEXT, group_val TEXT, cluster_id INTEGER, n INTEGER)")
    if engine == "sqlite":
        con.executemany("INSERT INTO Contingency VALUES (?, ?, ?, ?, ?)", table.itertuples(index=False, name=None))
    else:
        con.register("df_tmp", table); con.execute("INSERT INTO Contingency SELECT * FROM df_tmp")
    rows = _fetch(con, HGP_FROM_CONTINGENCY, (attr_name,))
    con.close()
    return hgp_from_rows([(int(k), int(u), int(y)) for k, u, y in rows])


def hgp_from_rows(rows) -> Fraction:
    vol = sum(k * u for k, u, _ in rows)
    return sum((Fraction(y, k) for k, _, y in rows), Fraction(0)) / vol

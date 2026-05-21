### neo4j backend api (graph database)
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
import os
from typing import Literal
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware

# :use system
# CREATE USER public_reader SET PASSWORD 'Str0ngP@ssw0rd' CHANGE NOT REQUIRED;
# SHOW USERS;

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "x")
NEO4J_PASS = os.getenv("NEO4J_PASS", "y")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),)


# Frontend origins you’ll load the site from in dev
ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.10.108:3000",  # ← your LAN IP + port
]

app = FastAPI(title="Ratsinfo Public API", version="0.1.0")

# Allow your frontend origin during dev; tighten in prod
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # replace with ["https://your-site"] in prod
#     allow_credentials=False,
#     allow_methods=["GET"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,       # exact origins incl. scheme + port
    allow_credentials=True,      # keep False if you don’t use cookies/auth
    allow_methods=["*"],         # or ["GET","POST","OPTIONS"]
    allow_headers=["*"],         # e.g. "content-type", "authorization"
    max_age=3600,
)

def ensure_fulltext(driver):
    stmts = [
        """
        CREATE FULLTEXT INDEX ftx_ref IF NOT EXISTS
        FOR (r:Referenzvorlage) ON EACH [r.referenzvorlage]
        OPTIONS { indexConfig: { `fulltext.analyzer`: 'german' } }
        """,
        """
        CREATE FULLTEXT INDEX ftx_top IF NOT EXISTS
        FOR (t:TOP) ON EACH [t.vorlage, t.name]
        OPTIONS { indexConfig: { `fulltext.analyzer`: 'german' } }
        """,
    ]
    with driver.session() as s:
        for c in stmts:
            s.run(c)


ensure_fulltext(driver)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/search")
def search(q: str, gremium: str | None = None):
    # cypher_top = """
    # CALL db.index.fulltext.queryNodes('ftx_top', $q) YIELD node, score
    # OPTIONAL MATCH (node)<-[:HAS_TOP]-(m:Meeting)-[:HELD_BY]->(g:Gremium)
    # # OPTIONAL MATCH (node)-[:REFERS_TO]->(r:Referenzvorlage)
    # WHERE $gremium IS NULL OR g.name = $gremium
    # WITH node, max(m.date) AS mdate,
    #      head(collect(g.name)) AS gname,
    #      head(collect(r.referenzvorlage)) AS refv,
    #      max(score) AS s
    # RETURN
    #   node.top_id       AS top_id,
    #   node.name         AS name,
    #   node.vorlage      AS vorlage,
    #   node.vorlage_url  AS vorlage_url,
    #   gname             AS gremium,
    #   toString(date(mdate)) AS meeting_date,
    #   refv              AS referenzvorlage,
    #   s                 AS score
    # ORDER BY s DESC, mdate DESC
    # LIMIT 20
    # """
    cypher_top = """
        CALL db.index.fulltext.queryNodes('ftx_top', $q) YIELD node, score
        OPTIONAL MATCH (node)<-[:HAS_TOP]-(m:Meeting)-[:HELD_BY]->(g:Gremium)
        WHERE $gremium IS NULL OR g.name = $gremium
        WITH node, max(m.date) AS mdate,
             head(collect(g.name)) AS gname,
             max(score) AS s
        RETURN
          node.top_id       AS top_id,
          node.name         AS name,
          node.vorlage      AS vorlage,
          node.vorlage_url  AS vorlage_url,
          gname             AS gremium,
          toString(date(mdate)) AS meeting_date,
          s                 AS score
        ORDER BY s DESC, mdate DESC
        LIMIT 2000
        """
    params = {"q": q, "gremium": gremium}
    print(gremium)
    with driver.session() as s:
        return {"tops": s.run(cypher_top, **params).data()}

@app.get("/api/referenzvorlagen")
def referenzvorlagen(
    gremium: str = Query("Rat der Stadt"),
    min_tops: int = Query(2, ge=1),
    from_: str | None = Query(None, alias="from", description="YYYY-MM-DD"),
    to: str | None = Query(None, description="YYYY-MM-DD"),
    include_tops: bool = Query(True),
):
    cypher = """
        MATCH (:Gremium {name:$gremium})<-[:HELD_BY]-(m:Meeting)-[:HAS_TOP]->(t:TOP)-[:REFERS_TO]->(r:Referenzvorlage)
        WHERE ($from IS NULL OR date(m.date) >= date($from))
          AND ($to   IS NULL OR date(m.date)  < date($to))
        WITH r, collect(DISTINCT t) AS tops, collect(DISTINCT m) AS meetings, max(date(m.date)) AS latest_date
        WHERE size(tops) >= $min_tops
        
        CALL {
          WITH tops
          UNWIND tops AS x
          OPTIONAL MATCH (mx:Meeting)-[:HAS_TOP]->(x)
          WITH x, max(date(mx.date)) AS mx_date, split(coalesce(x.vorlage, ""), "/") AS parts
          WITH
            x, mx_date,
            coalesce(parts[0], "") AS pfx,
            // numeric suffix if present, else -1 so numbers sort above "no suffix" in DESC
            CASE WHEN size(parts) > 1 AND parts[1] =~ '^[0-9]+$' THEN toInteger(parts[1]) ELSE -1 END AS n1,
            coalesce(parts[1], "") AS s1
          ORDER BY mx_date DESC, pfx DESC, n1 DESC, s1 DESC, coalesce(x.top_id, "") DESC
          RETURN collect(x) AS tops_sorted
        }
        
        RETURN
          r.referenzvorlage AS referenzvorlage,
          size(tops_sorted) AS num_tops,
          size(meetings)    AS num_meetings,
          latest_date,
          CASE WHEN $include_tops THEN
            [x IN tops_sorted | {
              top_id:       x.top_id,
              name:         x.name,
              vorlage:      x.vorlage,
              vorlage_url:  x.vorlage_url,
              meeting_date: head([(x)<-[:HAS_TOP]-(mx:Meeting) | toString(date(mx.date))])
            }][0..50]
          ELSE [] END AS tops
        ORDER BY latest_date DESC, referenzvorlage DESC;
        """
    params = {
        "gremium": gremium,
        "min_tops": min_tops,
        "from": from_,  # note: Python var from_ maps to $from in Cypher
        "to": to,
        "include_tops": include_tops,
    }
    with driver.session() as s:
        return s.run(cypher, **params).data()

from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Optional, Tuple, Dict, Any, List
import json, re, datetime as dt
from neo4j import GraphDatabase

# ---- paths & config ----
BASE = Path("bürgerportal_dui")
PARSED = BASE / "parsed"
MEET_DIR = PARSED / "meetings"
TOP_DIR  = PARSED / "tops"
PERSON_DIR  = PARSED / "persons"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "secretpassword"

# ---- utils ----
def read_json(p: Path) -> Dict[str, Any]:
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def clean_text(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

# def meeting_id_from_url(url: str) -> int:
#     qs = parse_qs(urlparse(url).query)
#     k = qs.get("__ksinr", [None])[0]
#     if not k or not str(k).isdigit():
#         raise ValueError(f"Cannot derive meeting_id from url: {url}")
#     return int(k)

def to_iso_date(d_de: Optional[str]) -> Optional[str]:
    d_de = clean_text(d_de)
    if not d_de:
        return None
    return dt.datetime.strptime(d_de, "%d.%m.%Y").date().isoformat()

def split_time_range(traw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    t = clean_text(traw)
    if not t:
        return None, None
    t = t.replace(" Uhr", "")
    def iso(hm: str) -> str:
        return dt.datetime.strptime(hm, "%H:%M").time().isoformat()
    if "-" in t:
        a, b = [x.strip() for x in t.split("-", 1)]
        return iso(a), iso(b)
    return iso(t), None

# # ---- normalizers ----
# def normalize_meeting(rec: Dict[str, Any]) -> Dict[str, Any]:
#     m_id = meeting_id_from_url(rec.get("source_url", ""))
#     title = clean_text(rec.get("title"))
#     location = clean_text(rec.get("location"))
#     sitzung_code = clean_text(rec.get("Sitzung"))
#     gremium = clean_text(rec.get("Gremium"))
#     date_iso = to_iso_date(rec.get("Datum"))
#     start_time, end_time = split_time_range(rec.get("Zeit"))
#     saved_at = clean_text(rec.get("saved_at"))
#
#     return {
#         "meeting_id": m_id,                 # numeric __ksinr
#         "sitzung_code": sitzung_code,       # e.g. "B92/2025/0063"
#         "title": title,
#         "gremium": gremium,
#         "date": date_iso,
#         "start_time": start_time,
#         "end_time": end_time,
#         "location": location,
#         "source_url": rec.get("source_url"),
#         "saved_at": saved_at,
#     }
#
# def normalize_top(rec: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     - Rename vorlage_id -> vorlage
#     - Add referenzvorlage (prefix before '/')
#     """
#     n = dict(rec)
#     n["name"] = clean_text(n.get("name"))
#     n["meeting_id"] = clean_text(n.get("meeting_id"))  # Sitzung code style
#     vorlage = clean_text(n.get("vorlage_id"))
#     n["vorlage"] = vorlage
#     n["referenzvorlage"] = vorlage.split('/')[0] if vorlage else None
#     n["vorlage_url"] = clean_text(n.get("vorlage_url"))  # kept for reference, not used for keys
#
#     # files: keep as-is but make sure ids are strings and labels cleaned
#     files = []
#     for f in n.get("files", []) or []:
#         files.append({
#             "id": str(f.get("id")).strip() if f.get("id") is not None else None,
#             "url": clean_text(f.get("url")),
#             "label": clean_text(f.get("label")),
#             "type": clean_text(f.get("type")),
#         })
#     n["files"] = files
#
#     # ensure top_id is present (if not already)
#     # if your JSON already has top_id, remove this safeguard
#     if "top_id" not in n or n["top_id"] is None:
#         # fall back to a composite if necessary
#         # adjust this to your real unique rule if needed
#         n["top_id"] = clean_text(n.get("id")) or f"{n.get('meeting_id')}::{n.get('name')}"
#     return n

# ---- Neo4j ----



def upsert_politician(driver, politician: list[dict]):
    cypher = """
    UNWIND $batch AS m
    MERGE (pers:Person {person_id: m.person_id})
      SET pers.nachname = m.nachname,
          pers.vorname = m.vorname,
          pers.anrede = m.anrede,
          pers.partei = m.partei['handle'],
          pers.ende = m.ende,
          pers.href = m.href,
          pers.data_json = m["data_json"]
    """
    with driver.session() as s:
        s.run(cypher, batch=politician)


import uuid

pers = [read_json(p) for p in PERSON_DIR.glob("*.json")]
for p in pers:
    if not p['person_id']:
        p['person_id'] = uuid.uuid4().__str__()
    if isinstance(p.get("data"), dict):
        p["data_json"] = json.dumps(p["data"], ensure_ascii=False)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
# ensure_constraints(driver)
upsert_politician(driver, pers)
driver.close()


#### create party nodes and links/memberships: person -> party
def upsert_partei(driver, politician: list[dict]):
    cypher = """
    UNWIND $batch AS m
    MERGE (par:Partei {name: m.handle})
      SET par.name = m.handle,
          par.typ= m.typ,
          par.url = m.url
    """
    with driver.session() as s:
        s.run(cypher, batch=politician)


pars = {}
for p in PERSON_DIR.glob("*.json"):
    p=read_json(p)
    if p['partei']:
        if p['partei']['handle'] not in pars:
            pars.update({p['partei']['handle']: p['partei']})

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
# ensure_constraints(driver)
list(pars.values())
upsert_partei(driver, list(pars.values()))
driver.close()


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

cypher = """
   UNWIND $batch AS m

   MATCH (pers:Person {person_id: m.person_id})
   MATCH (party:Partei {name: m.partei.handle})

   MERGE (pers)-[:MITGLIED_BEI]->(party)
   """
add = [x for x in pers if x['partei'] and x['person_id'] ]

with driver.session() as s:
    s.run(cypher, batch= add)

driver.close()



### adding gremium
# extracting from meetings (incl first/last meeting date)
meets = {}
for p in MEET_DIR.glob("*.json"):
    p=read_json(p)

    if 'abgesagt' in p['title'].lower():
        continue
    short = p['Sitzung'].split('/')[0]
    datum_ = to_iso_date(d_de=p['Datum'])

    if short not in meets:
        meets[short] = {'short': short, 'gremium': p['Gremium'], 'dates': [datum_]}
    else:
        meets[short]['dates'].append(datum_)

for k,d in meets.items():
    d.update({'n_entries': len(d['dates']), 'start_datum': min(d['dates'])})


[d['name'] for k,d in meets.items() ]

import pandas as pd
df=pd.DataFrame(meets).T



driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

cypher = """
    UNWIND $batch AS m
    MERGE (gremium:Gremium {akronym: m.short})
      SET gremium.name = m.gremium,
          gremium.n_einträge= m.n_entries,
          gremium.start_datum = m.start_datum
    """

with driver.session() as s:
    s.run(cypher, batch= list(meets.values()))

driver.close()


########
# sitzungen (id, gremium, datum, urhzeit)... dann teilnahme unbeding mit "role": "mitglied" vs other, eg, "Sachkundige/r Einwohner/in",
meets = {}
links = []
for p in MEET_DIR.glob("*.json"):
    p=read_json(p)

    if 'abgesagt' in p['title'].lower():
        continue
    if 'keine präsenzsitzung' in p['title'].lower():
        Zeit = ''
    else:
        zeit = p['Zeit'].replace('\xa0', ' ')

    id = p['Sitzung']
    titel = p['title']
    short = p['Sitzung'].split('/')
    datum_ = to_iso_date(d_de=p['Datum'])
    location = p['location'] if p['location'] else ""
    if 'Gremium' in p:
        gremium = p['Gremium']
    elif 'Gremien' in p:
        gremium = p['Gremien']
    else:
        gremium = ""
    url = p['source_url']

    meets[id] = dict(id=id, titel=titel, datum=datum_, zeit=zeit, ort=location, gremium=gremium, source_url=url, snummer=short[2], jahr=short[1], gremium_k=short[0])
    meets[id]['participants'] = None

    if 'attendence' in p:
        links = [{'id': x['id'], 'name': x['name'], 'rolle': x['role']} for x in p['attendence']]
        meets[id]['participants'] = links

df = pd.DataFrame(meets).T


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

cypher = """
    UNWIND $batch AS m
    MERGE (sitz:Sitzung {id: m.id})
      SET sitz.titel = m.titel,
          sitz.jahr = m.jahr,
          sitz.gremium_kürzel = m.gremium_k,
          sitz.sitzungsnummer = m.snummer,
          sitz.datum= m.datum,
          sitz.urhzeit = m.uhrzeit,
          sitz.gremium = m.gremium,
          sitz.source_url = m.source_url
    """

with driver.session() as s:
    s.run(cypher, batch= list(meets.values()))

driver.close()


#### add links
df.participants.iloc[0]
df['id'].iloc[0]

mp_links = [{'mid': d['id']} | {'pid': p['id'], 'rolle': p['rolle']}  for k,d in meets.items() for p in d.get("participants") or []]


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

cypher = """
   UNWIND $batch AS m

   MATCH (pers:Person {person_id: m.pid})
   MATCH (meet:Sitzung {id: m.mid})

   MERGE (pers)-[r:teilnahme_an]->(meet)
   set r.rolle = m.rolle
   """

with driver.session() as s:
    s.run(cypher, batch= mp_links)

driver.close()







def ensure_constraints(driver):
    stmts = [
        # Uniqueness constraints
        "CREATE CONSTRAINT meeting_meeting_id IF NOT EXISTS FOR (m:Meeting) REQUIRE m.meeting_id IS UNIQUE",
        "CREATE CONSTRAINT top_top_id IF NOT EXISTS FOR (t:TOP) REQUIRE t.top_id IS UNIQUE",
        "CREATE CONSTRAINT gremium_name IF NOT EXISTS FOR (g:Gremium) REQUIRE g.name IS UNIQUE",
        "CREATE CONSTRAINT refvorlage_unique IF NOT EXISTS FOR (r:Referenzvorlage) REQUIRE r.referenzvorlage IS UNIQUE",
        "CREATE CONSTRAINT dokument_dok_id IF NOT EXISTS FOR (d:Dokument) REQUIRE d.dok_id IS UNIQUE",

        # Indexes
        "CREATE INDEX meeting_sitzung_code IF NOT EXISTS FOR (m:Meeting) ON (m.sitzung_code)",
        "CREATE INDEX top_referenzvorlage IF NOT EXISTS FOR (t:TOP) ON (t.referenzvorlage)",
        "CREATE INDEX top_vorlage IF NOT EXISTS FOR (t:TOP) ON (t.vorlage)",
    ]

    with driver.session() as s:
        for stmt in stmts:
            s.run(stmt)
    # def _apply(tx):
    #     for stmt in stmts:
    #         tx.run(stmt)
    #
    # with driver.session() as s:
    #     s.execute_write(_apply)

def upsert_meetings(driver, meetings: List[Dict[str, Any]]):
    cypher = """
    UNWIND $batch AS m
    MERGE (meet:Meeting {meeting_id: m.meeting_id})
      SET meet.title        = m.title,
          meet.sitzung_code = m.sitzung_code,
          meet.gremium      = m.gremium,
          meet.date         = m.date,
          meet.start_time   = m.start_time,
          meet.end_time     = m.end_time,
          meet.location     = m.location,
          meet.source_url   = m.source_url,
          meet.saved_at     = m.saved_at
    WITH meet, m
    WHERE m.gremium IS NOT NULL
    MERGE (g:Gremium {name: m.gremium})
    MERGE (meet)-[:HELD_BY]->(g);
    """
    with driver.session() as s:
        s.run(cypher, batch=meetings)

def upsert_tops_and_links(driver, tops):
    cypher = """
    UNWIND $batch AS t

    // 1) TOP
    MERGE (top:TOP {top_id: t.top_id})
      SET top.name            = t.name,
          top.aktenzeichen    = t.aktenzeichen,
          top.saved_at        = t.saved_at,
          top.public          = t.public,
          top.vorlage         = t.vorlage,
          top.referenzvorlage = CASE
                                   WHEN t.referenzvorlage IS NULL OR trim(toString(t.referenzvorlage)) = "" THEN NULL
                                   ELSE toString(t.referenzvorlage)
                                 END,
          top.vorlage_url     = t.vorlage_url

    // 2) link Meeting by Sitzung code
    WITH t, top
    OPTIONAL MATCH (m:Meeting {sitzung_code: t.meeting_id})
    FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
      MERGE (m)-[:HAS_TOP]->(top)
      SET top.meeting_ksinr = m.meeting_id
    )

    // 3) Referenzvorlage and link (keep `t` in scope!)
    WITH t, top
    WITH t, top,
         CASE
           WHEN t.referenzvorlage IS NULL OR trim(toString(t.referenzvorlage)) = "" THEN NULL
           ELSE toString(t.referenzvorlage)
         END AS ref
    FOREACH (_ IN CASE WHEN ref IS NOT NULL THEN [1] ELSE [] END |
      MERGE (r:Referenzvorlage {referenzvorlage: ref})
      MERGE (top)-[:REFERS_TO]->(r)
    )

    // 4) Dokumente from files[]
    WITH t, top
    UNWIND coalesce(t.files, []) AS f
    WITH top, f
    WHERE f.id IS NOT NULL
    MERGE (d:Dokument {dok_id: toInteger(f.id)})
      ON CREATE SET d.url = f.url, d.label = f.label, d.type = f.type
      ON MATCH  SET d.url = coalesce(d.url, f.url),
                   d.label = coalesce(d.label, f.label),
                   d.type = coalesce(d.type, f.type)
    MERGE (top)-[:HAS_FILE]->(d);
    """
    with driver.session() as s:
        s.run(cypher, batch=tops)

# ---- main ----
if __name__ == "__main__":
    meetings = [normalize_meeting(read_json(p)) for p in MEET_DIR.glob("*.json")]
    tops     = [normalize_top(read_json(p))     for p in TOP_DIR.glob("*.json")]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    ensure_constraints(driver)
    upsert_meetings(driver, meetings)
    upsert_tops_and_links(driver, tops)
    driver.close()
    print(f"Upserted {len(meetings)} meetings and {len(tops)} TOPs (with gremium/referenzvorlage/files links).")



### people in meetings



















from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
from typing import Optional, Tuple

from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json, re, datetime as dt
from neo4j import GraphDatabase

# assuming these globals exist (like in your savers)
RAW = Path("bürgerportal_dui/raw_html")
ATTACH = Path("bürgerportal_dui/attachments")

PARSED = Path("bürgerportal_dui/parsed")
MEET_DIR = PARSED / "meetings"
TOP_DIR  = PARSED / "tops"


# ---- config ----
MEET_DIR = PARSED / "meetings"  # folder with your meeting *.json files
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "secretpassword"



# --- helpers ---

# --- utils ---
def read_json(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def clean_text(s: Optional[str]) -> Optional[str]:
    """Replace NBSP, collapse whitespace; keep None as None."""
    if not isinstance(s, str):
        return None
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s or None

def meeting_id_from_url(url: str) -> int:
    qs = parse_qs(urlparse(url).query)
    k = qs.get("__ksinr", [None])[0]
    if not k or not k.isdigit():
        raise ValueError(f"Cannot derive meeting_id from url: {url}")
    return int(k)

def to_iso_date(d_de: Optional[str]) -> Optional[str]:
    d_de = clean_text(d_de)
    if not d_de:
        return None
    return dt.datetime.strptime(d_de, "%d.%m.%Y").date().isoformat()

def split_time_range(traw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (start_iso, end_iso). Accepts 'HH:MM', 'HH:MM-HH:MM', with optional ' Uhr' and NBSPs."""
    t = clean_text(traw)
    if not t:
        return None, None
    t = t.replace(" Uhr", "")
    def iso(hm: str) -> str:
        return dt.datetime.strptime(hm, "%H:%M").time().isoformat()
    if "-" in t:
        a, b = [x.strip() for x in t.split("-", 1)]
        return iso(a), iso(b)
    return iso(t), None

# --- robust normalizer ---

# --- robust normalizer ---
def normalize_meeting(rec: dict) -> dict:
    m = dict(rec)  # copy
    m_id = meeting_id_from_url(m.get("source_url", ""))

    title = clean_text(m.get("title"))
    location = clean_text(m.get("location"))
    sitzung_code = clean_text(m.get("Sitzung"))
    gremium = clean_text(m.get("Gremium"))
    date_iso = to_iso_date(m.get("Datum"))
    start_time, end_time = split_time_range(m.get("Zeit"))
    saved_at = clean_text(m.get("saved_at"))

    return {
        "meeting_id": m_id,
        "sitzung_code": sitzung_code,
        "title": title,
        "gremium": gremium,
        "date": date_iso,
        "start_time": start_time,
        "end_time": end_time,
        "location": location,
        "source_url": m.get("source_url"),
        "saved_at": saved_at,
    }
# --- safer loader that shows which file had issues ---
def load_all_meetings(meet_dir: Path) -> list[dict]:
    out = []
    for p in sorted(meet_dir.glob("*.json")):
        try:
            rec = read_json(p)
            out.append(normalize_meeting(rec))
        except Exception as e:
            print(f"[WARN] Skip {p.name}: {e}")
    return out


def normalize_meeting(rec: dict) -> dict:
    rec = dict(rec)
    rec["meeting_id"] = meeting_id_from_url(rec["source_url"])
    rec["date"] = dt.datetime.strptime(rec["Datum"], "%d.%m.%Y").date().isoformat()
    rec["title"] = rec["title"].replace("\xa0", " ")
    rec["location"] = rec["location"].replace("\xa0", " ")
    rec["time"] = rec["Zeit"].replace("\xa0", " ")
    return rec

def normalize_top(rec: dict) -> dict:
    rec = dict(rec)
    rec["name"] = rec["name"].replace("\xa0", " ") if rec.get("name") else None
    # meeting_id may be string like 'WIT/2025/0029' -> keep as-is, but store link separately
    return rec

# --- Neo4j setup ---
def ensure_constraints(driver):
    cypher = """
    CREATE CONSTRAINT meeting_id_unique IF NOT EXISTS
    FOR (m:Meeting) REQUIRE m.meeting_id IS UNIQUE;
    CREATE CONSTRAINT top_id_unique IF NOT EXISTS
    FOR (t:TOP) REQUIRE t.top_id IS UNIQUE;
    CREATE CONSTRAINT vorlage_id_unique IF NOT EXISTS
    FOR (v:Vorlage) REQUIRE v.vorlage_id IS UNIQUE;
    CREATE CONSTRAINT dokument_id_unique IF NOT EXISTS
    FOR (d:Dokument) REQUIRE d.dok_id IS UNIQUE;
    """
    with driver.session() as s:
        s.run(cypher)

def upsert_meetings(driver, meetings: list[dict]):
    cypher = """
    UNWIND $batch AS m
    MERGE (meet:Meeting {meeting_id: m.meeting_id})
      SET meet.title = m.title,
          meet.sitzung_code = m.Sitzung,
          meet.gremium = m.Gremium,
          meet.date = m.date,
          meet.time = m.time,
          meet.location = m.location,
          meet.source_url = m.source_url,
          meet.saved_at = m.saved_at
    """
    with driver.session() as s:
        s.run(cypher, batch=meetings)

def upsert_tops(driver, tops: list[dict]):
    cypher = """
    UNWIND $batch AS t
    MERGE (top:TOP {top_id: t.top_id})
      SET top.name = t.name,
          top.aktenzeichen = t.aktenzeichen,
          top.saved_at = t.saved_at,
          top.public = t.public
    WITH top, t
    MATCH (m:Meeting {sitzung_code: t.meeting_id})  // top.meeting_id is Sitzung-code style
    MERGE (m)-[:HAS_TOP]->(top)
    WITH top, t
    FOREACH (_ IN CASE WHEN t.vorlage_id IS NOT NULL THEN [1] ELSE [] END |
      MERGE (v:Vorlage {vorlage_id: t.vorlage_id})
        SET v.source_url = t.vorlage_url
      MERGE (top)-[:REFERS_TO]->(v)
    )
    WITH top, t
    UNWIND t.files AS f
      MERGE (d:Dokument {dok_id: f.id})
        SET d.label = f.label,
            d.url = f.url,
            d.type = f.type
      MERGE (top)-[:HAS_FILE]->(d)
    """
    with driver.session() as s:
        s.run(cypher, batch=tops)

# --- main ---
if __name__ == "__main__":
    meetings = [normalize_meeting(read_json(p)) for p in MEET_DIR.glob("*.json")]
    tops = [normalize_top(read_json(p)) for p in TOP_DIR.glob("*.json")]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    ensure_constraints(driver)
    upsert_meetings(driver, meetings)
    upsert_tops(driver, tops)
    driver.close()
    print(f"Imported {len(meetings)} meetings and {len(tops)} TOPs into Neo4j.")

























# ---- helpers ----
def read_json(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def meeting_id_from_url(url: str) -> int:
    qs = parse_qs(urlparse(url).query)
    k = qs.get("__ksinr", [None])[0]
    if not k or not k.isdigit():
        raise ValueError(f"Cannot derive meeting_id from url: {url}")
    return int(k)

def to_iso_date(d_de: str) -> str:
    # '09.09.2024' -> '2024-09-09'
    return dt.datetime.strptime(d_de.strip(), "%d.%m.%Y").date().isoformat()

def split_time_range(traw: str):
    """
    '15:03-15:50\xa0Uhr' -> ('15:03:00', '15:50:00')
    '15:00\xa0Uhr' -> ('15:00:00', None)
    """
    t = traw.replace("\xa0", " ").replace(" Uhr", "").strip()
    if "-" in t:
        a, b = [x.strip() for x in t.split("-", 1)]
        return _to_iso_time(a), _to_iso_time(b)
    else:
        return _to_iso_time(t), None

def _to_iso_time(t_hm: str) -> str:
    return dt.datetime.strptime(t_hm, "%H:%M").time().isoformat()  # 'HH:MM:SS'

def normalize_meeting(rec: dict) -> dict:
    m = dict(rec)  # copy
    m["meeting_id"] = meeting_id_from_url(m["source_url"])
    m["sitzung_code"] = m.get("Sitzung")
    m["gremium"] = m.get("Gremium")
    m["date"] = to_iso_date(m["Datum"]) if m.get("Datum") else None
    start_t, end_t = split_time_range(m.get("Zeit", "")) if m.get("Zeit") else (None, None)
    m["start_time"] = start_t
    m["end_time"] = end_t
    # clean NBSP in title/location
    for k in ("title", "location"):
        if k in m and isinstance(m[k], str):
            m[k] = re.sub(r"\s+", " ", m[k].replace("\xa0", " ")).strip()
    return {
        "meeting_id": m["meeting_id"],
        "sitzung_code": m["sitzung_code"],
        "title": m.get("title"),
        "gremium": m.get("gremium"),
        "date": m.get("date"),
        "start_time": m.get("start_time"),
        "end_time": m.get("end_time"),
        "location": m.get("location"),
        "source_url": m.get("source_url"),
        "saved_at": m.get("saved_at"),
    }

def load_all_meetings(meet_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(meet_dir.glob("*.json")):
        try:
            recs.append(normalize_meeting(read_json(p)))
        except Exception as e:
            print(f"Skip {p.name}: {e}")
    return recs

# ---- Neo4j load (MERGE nodes) ----
def ensure_constraints(driver):
    cypher = """
    CREATE CONSTRAINT meeting_id_unique IF NOT EXISTS
    FOR (m:Meeting) REQUIRE m.meeting_id IS UNIQUE;
    CREATE CONSTRAINT gremium_name_unique IF NOT EXISTS
    FOR (g:Gremium) REQUIRE g.name IS UNIQUE;
    """
    with driver.session() as s:
        s.run(cypher)

def upsert_meetings(driver, meetings: list[dict]):
    cypher = """
    UNWIND $batch AS m
    MERGE (meet:Meeting {meeting_id: m.meeting_id})
      SET meet.title = m.title,
          meet.sitzung_code = m.sitzung_code,
          meet.date = m.date,
          meet.start_time = m.start_time,
          meet.end_time = m.end_time,
          meet.location = m.location,
          meet.source_url = m.source_url,
          meet.saved_at = m.saved_at
    WITH meet, m
    WHERE m.gremium IS NOT NULL
    MERGE (g:Gremium {name: m.gremium})
    MERGE (meet)-[:HELD_BY]->(g);
    """
    # batch in chunks to avoid huge transactions if many files
    CHUNK = 1000
    with driver.session() as s:
        for i in range(0, len(meetings), CHUNK):
            s.run(cypher, batch=meetings[i:i+CHUNK])

if __name__ == "__main__":
    meetings = load_all_meetings(MEET_DIR)
    print(f"Loaded {len(meetings)} meetings")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    ensure_constraints(driver)
    upsert_meetings(driver, meetings)
    driver.close()
    print("Neo4j upsert done.")

























# ---------- utils ----------
def _read_json(p: Path) -> Dict[str, Any]:
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def _norm_sitzung_code(code: str) -> str:
    # your filenames replace '/' with '_'
    return code.replace("/", "_")

# ---------- meetings ----------
def list_meetings() -> Iterable[Dict[str, Any]]:
    """Yield all meeting dicts from MEET_DIR/*.json (lazy)."""
    for p in sorted(MEET_DIR.glob("*.json")):
        yield _read_json(p)

def get_meeting_by_sitzung_code(code: str) -> Optional[Dict[str, Any]]:
    """
    Load a meeting by its 'Sitzung' code (e.g. 'B95/2025/0192').
    Filename convention: {Sitzung.replace('/', '_')}.json
    """
    p = MEET_DIR / f"{_norm_sitzung_code(code)}.json"
    return _read_json(p) if p.exists() else None

def get_meeting_by_id(meeting_id: int) -> Optional[Dict[str, Any]]:
    """Find a meeting by meeting_id by scanning files (fast enough for a few thousand)."""
    for rec in list_meetings():
        if rec.get("meeting_id") == meeting_id:
            return rec
    return None

def latest_meeting() -> Optional[Dict[str, Any]]:
    """Return the most recently saved meeting (by saved_at)."""
    def key(p: Path) -> str:
        try:
            return _read_json(p).get("saved_at", "")
        except Exception:
            return ""
    files = list(MEET_DIR.glob("*.json"))
    return _read_json(max(files, key=key)) if files else None

# ---------- TOPs ----------
def list_tops() -> Iterable[Dict[str, Any]]:
    for p in sorted(TOP_DIR.glob("*.json")):
        yield _read_json(p)

def get_top(top_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a TOP by its top_id (your saver uses the normalized string as filename).
    """
    p = TOP_DIR / f"{top_id}.json"
    return _read_json(p) if p.exists() else None

def get_tops_for_meeting(meeting_id: int) -> list[Dict[str, Any]]:
    """Return all TOPs belonging to a meeting."""
    return [t for t in list_tops() if t.get("meeting_id") == meeting_id]

# ---------- convenience (joined) ----------
def meeting_with_tops_by_sitzung(code: str) -> Optional[Dict[str, Any]]:
    """Load a meeting by Sitzung code and attach its TOPs."""
    m = get_meeting_by_sitzung_code(code)
    if not m:
        return None
    m["tops"] = get_tops_for_meeting(m["meeting_id"])
    return m

def meeting_with_tops_by_id(meeting_id: int) -> Optional[Dict[str, Any]]:
    """Load a meeting by meeting_id and attach its TOPs."""
    m = get_meeting_by_id(meeting_id)
    if not m:
        return None
    m["tops"] = get_tops_for_meeting(meeting_id)
    return m


for x in list_meetings():
    print(x)
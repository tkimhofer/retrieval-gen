from pathlib import Path
import hashlib, re, time
import datetime as dt
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json, uuid
from pathlib import Path

from services.docGraph.helpers import list_files, read_topp

# Vector store: Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from opensearchpy import OpenSearch



##### import tops, then sanitise and
# - store for contextual search on opensearch db
# - chunk, embedd and store for semantic search on qdrant (BS24)
# - assign unique id for both db - merge query results later and dedupe/re-rank


# PARAMETERISATION

# vector database
QD_CLIENT = QdrantClient(
    host="127.0.0.1", port=6333, grpc_port=6334,
    prefer_grpc=True, timeout=120.0
)
QD_COLLECTION = "tops_chunks"
print(QD_CLIENT.get_collections())
# qdrant.delete_collection(collection_name="tops_chunks")

# embedding model
# EMB_MODEL= "all-MiniLM-L6-v2"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIM=384
EMBEDDER = SentenceTransformer(EMB_MODEL)

# open search - this is for semantic searches (using hybrid retrieval)
ops = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "Riv3r^Stone_2025"),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    http_compress=True)

INDEX = "tops_chunks_bm25"
ops.indices.create(index=INDEX, ignore=400, body={"settings":{"analysis":{"analyzer":{"default":{"type":"standard"}}}}})
print(ops.info())


def ensure_collection(dim):
    if not QD_CLIENT.collection_exists(QD_COLLECTION):
        QD_CLIENT.create_collection(
            collection_name=QD_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    # if COLL not in [c.name for c in qdrant.get_collections().collections]:
    #     QD_CLIENT.recreate_collection(
    #         collection_name=COLL,
    #         vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    #     )


def prepMeta(rec, meetings):

    vars = ['name', 'top_id', 'meeting_id', 'vorlage_id']

    out = {v:rec[v] for v in vars}
    out['typ'] = rec['type'] if 'type' in rec else None
    out['partei'] = ", ".join(rec['antrag_party']) if ('antrag_party' in rec)  and (len(rec['antrag_party']) >0) else None
    out['beratungsergebnis'] = rec['beschluss']['beraturngsergebnis_1'] if 'beschluss' in rec  else None
    out['stimmenverteilung'] = rec['beschluss']['stimmenverteilung'] if 'beschluss' in rec else None

    if rec['meeting_id'] not in meetings:
        raise ValueError('meeting id not found in meetings')

    return out | meetings[rec['meeting_id']]


def stable_uuid_from_str(s: str):
    return uuid.uuid5(uuid.NAMESPACE_URL, s)

def to_point(rec):
    vec = EMBEDDER.encode([rec["name"]], normalize_embeddings=True)[0].tolist()
    return PointStruct(
        id=stable_uuid_from_str(rec["top_id"]).__str__(),
        vector=vec,
        payload=rec  # all metadata here
    )

def upsert_batch(records, batch_size=128):
    buf = []
    for r in records:
        buf.append(to_point(r))
        if len(buf) >= batch_size:
            QD_CLIENT.upsert(collection_name=QD_COLLECTION, points=buf)
            buf = []
    if buf:
        QD_CLIENT.upsert(collection_name=QD_COLLECTION, points=buf)


def bm25_search(search_str: str, k:int, ops:OpenSearch, gremium=None, vorlage_id=None, typ=None, partei=None):
    # search_str = 'AfD'
    # typ = ["Antrag", "Anträge/Anfragen"]
    k=90
    filter_l = []
    if gremium:
        tt = 'term' if isinstance(gremium, str) else 'terms'
        filter_l.append({tt: {"gremium.keyword": gremium}})

    if vorlage_id:
        tt = 'term' if isinstance(vorlage_id, str) else 'terms'
        filter_l.append({tt: {"vorlage_id.keyword": vorlage_id}})

    if typ:
        tt = 'term' if isinstance(typ, str) else 'terms'
        filter_l.append({tt: {"typ.keyword": typ}})

    if partei:
        tt = 'term' if isinstance(partei, str) else 'terms'
        filter_l.append({tt: {"partei.keyword": partei}})


    bm = ops.search(index=INDEX, body={
        "size": k,
        "query": {"bool": {
            "must": [{"multi_match": {"query": search_str, "fields": ["name",], "fuzziness": "AUTO" }}],
            "filter": filter_l
    }}})

    # len(bm['hits']['hits'])


    for i in range(min([20, len(bm['hits']['hits'])])):
        rec = bm['hits']['hits'][i]
        print(
            f"{dt.date.fromtimestamp(rec['_source']['datum']).strftime('%Y-%m-%y')}: {rec['_source']['text']} - {rec['_source']['top_id']} ({rec['_source']['vorlage_id']})")

    # for i in range(min([20, len(bm['hits']['hits'])])):
    #     rec = bm['hits']['hits'][i]
    #     print(f"{dt.date.fromtimestamp(rec['_source']['datum']).strftime('%Y-%m-%y')} {rec['_source']['partei']}: {''.join(rec['_source']['text'].split(':')[1:]).strip()} - {rec['_source']['top_id']} ({rec['_source']['vorlage_id']})")



ensure_collection(EMB_DIM)

### read tops
# file_path=Path("bürgerportal_dui/parsed/tops/AOB_2021_0109:Ö:1.json")
# read_topp(file_path)
ff = [read_topp(p) for p in list_files("bürgerportal_dui/parsed/tops/")]


### prep meeting
meets = [read_topp(p) for p in list_files("bürgerportal_dui/parsed/meetings/")]

meetings = {}
for x in meets:
    gremium = x['Gremium'] if 'Gremium' in x else x['Gremien']
    datum = int(dt.datetime.strptime(x['Datum'], '%d.%m.%Y').timestamp())
    meetings[x['Sitzung']] = {
        'gremium': gremium,
        'datum': datum,
        'ort': x['location'],
        'url': x['source_url']
     }

rec_prepped = [prepMeta(rec, meetings) for rec in ff if rec['meeting_id'] in meetings]
len(rec_prepped) / len(ff)



# def upsert_batch(records):
#     points = [to_point(r) for r in records]
#     qdrant.upsert(collection_name=COLL, points=points)
# upsert_batch(rec_prepped)


from httpx import Timeout
from qdrant_client import QdrantClient

# qdrant = QdrantClient(
#     url="http://localhost:6333",              # or your URL
#     timeout=Timeout(connect=10.0, read=60.0, write=120.0, pool=60.0)
# )

# for x in rec_prepped:
#     if 'hochheide' in x['name'].lower():
#         print(x)
# # for x in rec_prepped[0:300]:
# #     to_point(x)

# upsert_batch(rec_prepped)

# rec_prepped[0]
# upsert_batch(rec_prepped[0:300])


for i, rec in enumerate(rec_prepped):
    pid = stable_uuid_from_str(rec["top_id"]).__str__()
    payload = dict(rec, chunk_index=i, text=rec['name'], text_length=len(rec['name']))
    ops.index(index=INDEX, id=pid, body={
        "text": rec['name'], **payload
    })



bm25_search(search_str='Rollator', k=50, ops=ops, vorlage_id=None, typ=None, partei=None)

#### sanitise intput strings, create db, then sanitise search strings

# BM25
# query
# user_groups
# k_vec=50
k_bm25=50
# top_n=10
bm = ops.search(index=INDEX, body={"size": k_bm25, "query": {"bool": {
    "must": [{"multi_match": {"query": 'SPD', "fields": ["name",]}}],
    "filter": [
                    # {"term": {"typ": "Antrag"}}
                    {"term": {"typ.keyword": "Anträge/Anfragen"}}
                    # If typ is already keyword, use {"term": {"typ": "Anträge/Anfragen"}}
                ]
}}})

for i in range(min([20, len(bm['hits']['hits'])])):
    print(bm['hits']['hits'][i]['_source']['text'])

#### value in typ für anträge
FIELD='typ.keyword'

body = {
    "size": 0,
    "aggs": {
        "uniq": {
            "terms": {"field": FIELD, "size": 5000, "order": {"_count": "desc"}}
        }
    },
}
resp = ops.search(index=INDEX, body=body)
unk = [b["key"] for b in resp["aggregations"]["uniq"]["buckets"]]

import re
import numpy as np
import pandas as pd
patt=re.compile(r"Nachtrag:\s*[0-9]{2}\.[0-9]{2}\.[0-9]{4}\s?")
patt_ = [re.sub(patt, '', x) for x in unk]


df=pd.DataFrame(pd.Series(patt_).value_counts()).reset_index(names=['typ', 'c'])

antrag = df.typ.loc[df.typ.str.contains(r'\bantrag\b|\banträge\b|\banfrage\b', case=False)]
antrag.to_list()
antrag = ['Anträge/Anfragen', 'Anträge', 'Anträge / Anfragen', 'Anfrage', 'Antrag', 'Anträge und Anfragen', 'Anfragen/Anträge', ' Nummer 1 Anträge', ' Nummer 1 Anträge/Anfragen', 'Anträge/Anfrage', 'Anträge und Anfrage', ' Nummer 1 Anfrage', ' Nummer 1 Anträge/Anfrage', 'Antrag zur Kenntnisnahme', ' Nummer 1 Anträge und Anfragen', ' Nummer 1 Antrag', ' Nummer 2 Antrag', ' Nummer 1 Antrag/Anfrage', 'Antrag und Anfragen', 'Antrag/Anfrage', 'Antrag/Anfragen', 'Anfragen / Anträge', ' Nummer 2 Anträge und Anfragen', 'Anfrage der AfD-Fraktion', ' Nummer 2 Tischvorlagen / Anträge', ' Nummer 1 Antrag zur Kenntnisnahme', ' Nummer 2 Antrag/Anfrage', ' Nummer 1 Antrag und Anfrage', 'Antrag/Anfagen', 'Antrag und Anfrage', 'Anträge der Fraktion Tierschutz/DAL', 'Anträge/ Anfragen', 'Anträge/Nachfragen', ' Nummer 1 Antrag/Anfragen']
#
# dfr = df.typ.loc[~df.typ.str.contains(r'\bantrag\b|\banträge\b|\banfrage\b', case=False)]
#
# re.sub(r"Nachtrag:\s*[0-9]{2}\.[0-9]{2}\.[0-9]{4}\s?", '', 'Nachtrag: 31.10.2024 Beantragte Erweiterung der Tagesordnung')


### create a simgple ui that allows searching and lists tops, with meeting id and date





#############

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

def search_titles(query, *, gremium=None, typ=None, datum_von=None, datum_bis=None, vorlage_id=None, limit=10):

    qvec = EMB.dencode([query], normalize_embeddings=True)[0].tolist()

    query = 'spd'
    gremium = None #'Ausschuss für Umwelt, Klima und Naturschutz'
    typ = 'Antrag'
    datum_von = '2019-01-01'
    datum_bis = None
    vorlage_id = None
    limit = 10

    qvec = EMB.encode([query], normalize_embeddings=True)[0].tolist()

    must = []
    if vorlage_id:
        must.append(FieldCondition(key="vorlage_id", match=MatchValue(value=vorlage_id)))
    if gremium:
        must.append(FieldCondition(key="gremium", match=MatchValue(value=gremium)))
    if typ:
        must.append(FieldCondition(key="typ", match=MatchValue(value=typ)))
    if datum_von or datum_bis:
        conds = {}
        # if datum_von: conds["gte"] = datum_von
        # if datum_bis:   conds["lte"] = datum_bis
        if datum_von: conds["gte"] = int(dt.datetime.fromisoformat(datum_von).timestamp())
        if datum_bis:   conds["lte"] = int(dt.datetime.fromisoformat(datum_bis).timestamp())
        must.append(FieldCondition(key="datum", range=Range(**conds)))

    flt = Filter(must=must) if must else None

    # hits = qdrant.search(
    #     collection_name=COLL,
    #     query_vector=qvec,
    #     query_filter=flt,
    #     limit=limit,
    #     with_payload=True,
    #     # search_params=SearchParams(hnsw_ef=256)  # a bit higher recall
    # )


    # run query
    hits = qdrant.query_points(
        collection_name="tops_chunks",
        query = qvec,
        # filter=flt,
        limit=limit,
        with_payload=True
    )

    for point in hits.points:
        print(round(point.score,2), point.payload["name"], point.payload["datum"])


    # hits = qdrant.search(
    #     collection_name=COLL,
    #     query_vector=qvec,
    #     query_filter=flt,
    #     limit=limit,
    #     with_payload=True
    # )
    # Format output
    return [
        {
          "tops_id": h.payload["tops_id"],
          "title": h.payload["title"],
          "committee": h.payload.get("committee"),
          "meeting_date": h.payload.get("meeting_date"),
          "status": h.payload.get("status"),
          "score": float(h.score),
          "source_url": h.payload.get("source_url")
        }
        for h in hits
    ]

# Example query
results = search_titles(
    "Kita Förderung",
    committee="Jugendhilfeausschuss",
    date_from="2025-01-01",
    date_to="2025-12-31",
    user_groups=["group:ratsinfo"],
    limit=5
)
for r in results:
    print(r)













def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def extract_text(bytes_, mime, filename):
    # very abbreviated; plug in proper extractors per mime
    if mime == "application/pdf" or filename.lower().endswith(".pdf"):
        import fitz
        doc = fitz.open(stream=bytes_, filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc])
    elif filename.lower().endswith(".docx"):
        import docx
        from io import BytesIO
        d = docx.Document(BytesIO(bytes_))
        return "\n".join([p.text for p in d.paragraphs])
    else:
        # fallback: try textract
        import textract, tempfile
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as f:
            f.write(bytes_); f.flush()
            return textract.process(f.name).decode("utf-8", errors="ignore")

def normalize(t):
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def chunk_text(t, target_tokens=500, overlap_tokens=80):
    # simple sentence-ish splitter
    sentences = re.split(r'(?<=[\.!?])\s+', t)
    chunks, cur = [], []
    cur_len = 0
    for s in sentences:
        n = len(s.split())
        if cur_len + n > target_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap
            cur = cur[-overlap_tokens//10:]  # approx: 10 words ~ 1/10 of tokens
            cur_len = len(" ".join(cur).split())
        cur.append(s); cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def embed_chunks(chunks):
    return EMB.encode(chunks, normalize_embeddings=True).tolist()






# # Keyword store: OpenSearch (or Elasticsearch)
# from opensearchpy import OpenSearch
# os = OpenSearch(
#     hosts=[{"host": "localhost", "port": 9200}],
#     http_auth=("admin", "Riv3r^Stone_2025"),
#     use_ssl=True,
#     verify_certs=False,         # self-signed; skip verification (dev only)
#     ssl_assert_hostname=False,
#     ssl_show_warn=False,
#     http_compress=True)
# INDEX = "tops_chunks_bm25"
# os.indices.create(index=INDEX, ignore=400, body={"settings":{"analysis":{"analyzer":{"default":{"type":"standard"}}}}})
#
# print(os.info())         # sanity check



def upsert_attachment(tops_id, attachment):
    # attachment = {bytes, filename, mime, author, created_at, tags, permissions, ...}
    raw = attachment["bytes"]
    text = normalize(extract_text(raw, attachment["mime"], attachment["filename"]))
    if not text: return

    lang = attachment.get("lang") or detect(text[:2000])
    chunks = chunk_text(text)
    vecs = EMB.encode(chunks, normalize_embeddings=True)

    meta_common = {
        "tops_id": tops_id,
        "attachment_id": attachment["id"],
        "filename": attachment["filename"],
        "mime": attachment["mime"],
        "created_at": attachment["created_at"],
        "author": attachment.get("author"),
        "tags": attachment.get("tags", []),
        "permissions": attachment.get("permissions", []),
        "lang": lang,
    }

    ensure_collection(dim=len(vecs[0]))

    points = []
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        pid = f"{tops_id}:{attachment['id']}:{i}"
        payload = dict(meta_common, chunk_index=i, text=c, text_length=len(c))
        points.append(PointStruct(id=pid, vector=v, payload=payload))

        os.index(index=INDEX, id=pid, body={
            "text": c, **payload
        })

    qdrant.upsert(collection_name=COLL, points=points)


from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def search(query, user_groups, k_vec=50, k_bm25=50, top_n=10):
    # Vector
    qv = EMB.encode([query], normalize_embeddings=True)[0]
    vec_hits = qdrant.search(
        collection_name=COLL,
        query_vector=qv,
        limit=k_vec,
        query_filter=Filter(  # ACL
            must=[FieldCondition(key="permissions", match=MatchValue(any=user_groups))]
        )
    )

    # BM25
    bm = os.search(index=INDEX, body={"size": k_bm25, "query": {"bool":{
        "must":[{"multi_match":{"query": query, "fields":["text^3","tags","filename","tops_id"]}}],
        "filter":[{"terms":{"permissions": user_groups}}]
    }}})

    # fuse
    candidates = {}
    for h in vec_hits:
        candidates[h.id] = {"text": h.payload["text"], "meta": h.payload, "score_v": h.score}
    for hit in bm["hits"]["hits"]:
        _id = hit["_id"]
        s = hit["_score"]
        if _id not in candidates:
            candidates[_id] = {"text": hit["_source"]["text"], "meta": hit["_source"], "score_b": s}
        else:
            candidates[_id]["score_b"] = s

    # rerank
    items = [(cid, c) for cid, c in candidates.items()]
    pairs = [[query, c["text"]] for _, c in items]
    rerank_scores = reranker.predict(pairs).tolist()
    ranked = sorted(
        [(cid, c, r) for (cid, c), r in zip(items, rerank_scores)],
        key=lambda x: x[2], reverse=True
    )[:top_n]

    # group results and return with citations
    out = []
    for cid, c, r in ranked:
        out.append({
            "id": cid,
            "score": r,
            "tops_id": c["meta"]["tops_id"],
            "attachment_id": c["meta"]["attachment_id"],
            "filename": c["meta"]["filename"],
            "snippet": c["text"][:400] + ("…" if len(c["text"])>400 else "")
        })
    return out



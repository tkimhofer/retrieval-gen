from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple
import time

from services.retrieval.Retriever import retr

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    retrieve_k: int = 50
    top_k: int = 10
    include_text: bool = True

# the "dict" part (allow any extra keys you didn't list explicitly)
class ResultDict(BaseModel):
    model_config = ConfigDict(extra="allow")
    qtype: str | None = None
    chunk_id: str | None = None
    vorlage_id: str | None = None
    aktenzeichen: str | None = None
    typ: str | None = None
    betreff: str | None = None
    text: str | None = None
    text_betr: str | None = None
    hit_score_norm: float | None = None
    hit_score: float | None = None
    hit_id: str | None = None
    source: str | None = None
    datum_min: str | None = None
    datum_max: str | None = None


# each item is (score, dict)
SearchItem = tuple[float, ResultDict]

class SearchResponse(BaseModel):
    query: str
    items: list[SearchItem]
    took_ms: float



app = FastAPI(title="Retriever API")
# CORS for your dev FE (adjust domain/port as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://192.168.178.84:3000",
        "http://192.168.10.157:3000"
        # "http://localhost:3000",
        # "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search", response_model=SearchResponse)
def search(req: QueryRequest):

    t0 = time.time()
    try:
        print(req.query)
        retr.query_db(q = req.query)
        items = retr.rerank_search_results(return_n=req.top_k)
        # print(items)
        # items = retr.search(req.query, retrieve_k=req.retrieve_k, return_k=req.top_k)
        # if not req.include_text:
        #     for it in items:
        #         it.pop("text", None)
        took = int((time.time() - t0) * 1000)
        print(took)
        obj = SearchResponse(query=req.query, items=items, took_ms=took)
        print(len(items))

        return obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# self = retr
#
# qq = 'Bundesgartenschau'
# qq = 'Familienzentrum Wrangelstraße'
# qq = 'Bärenstark und Fuchsschlau'
# self.sparse_search(q=qq)
# self.spars_norm

### get information about vorlage id
# [x.payload['vorlage_id'] for x in self.hits]

#### gather text data for each vorlagen


#
#
# self.query_db(q = qq)
# self.rerank_search_results()
# #
# # self.sparse_search('Grünen Ring Mitte')
# #
# self.dense_search('Bebauungsplan 687 ZebraPark')
# # from sentence_transformers import CrossEncoder

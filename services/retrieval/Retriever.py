import re
from typing import List, Dict
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import os, json
from qdrant_client import QdrantClient, models
from opensearchpy import OpenSearch

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F

dim=384


# host = "https://your-cluster-endpoint:9200"
# auth = ("admin", "your-password")  # or IAM, SigV4 for AWS OpenSearch Service


index_name = collection = "duisburg_chunks"
ops = OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}],
            http_auth=("admin", "Riv3r^Stone_2025"),
            use_ssl=True,
            verify_certs=False,      # dev only
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            http_compress=True,
            timeout=60,
        )

qdrant_client = QdrantClient(host="127.0.0.1", port=6333, grpc_port=6334, prefer_grpc=True, timeout=120.0)




from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
emb = model.encode(['Bundesgartenschau'], normalize_embeddings=True, convert_to_tensor=True)
len(emb)
emb[0].shape

from sentence_transformers import CrossEncoder

class Retriever:

    def __init__(self, ops, model, qdrant):
        self.ops = ops
        self.qdrant_client = qdrant
        self.encoder = model

    def dense_search(self, q:str, k=20):

        typ = 'dense'
        self.query = q
        # 4) Query vector
        query_vec = self.encoder.encode(self.query, normalize_embeddings=True, convert_to_tensor=True)
        #
        # # (Optional) Filter & search params
        # flt = models.Filter(
        #     must=[models.FieldCondition(key="lang", match=models.MatchValue(value="en"))]
        # )
        params = models.SearchParams(hnsw_ef=128, exact=False)  # exact=True for exact kNN

        # 5) Search for the closest match
        self.hits = qdrant_client.search(
            collection_name=index_name,
            query_vector=query_vec,
            limit=k,
            with_payload=True,
            score_threshold=None,  # or e.g. 0.75 for cosine similarity
            search_params=params,
            # query_filter=flt,
        )


        self.score_norm = self.minmax([hit.score  for hit in self.hits])


        ### retrieving text data
        # return text of search results from opensearch
        chunk_ids = [hit.payload['chunk_id'] for hit in self.hits]

        resp = ops.search(
            index=index_name,
            body={
                # "_source": ["text", "chunk_id"],  # return only what you need
                "size": len(chunk_ids),  # default is 10; bump it
                "track_total_hits": True,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"chunk_id": chunk_ids}}  # or "chunk_id.keyword"
                        ]
                    }
                }
            }
        )

        hits_ = {hit['_id']: hit['_source'] for hit in resp["hits"]["hits"]}

        # cc86a58cc14752ad9b7d1370c777e9e2-000001-c0430fb75cbe
        # '_source': {
        # 'betreff': 'Einrichtung eines befristeten Teilstandortes an der Theodor-König-Gesamtschule in Duisburg-Meiderich/Beeck',
        # 'vorlage_id': '19-1389',
        # 'typ': None,
        # 'aktenzeichen': 'III/40-13 Tönges, 2522',
        # 'datum_min': '2020-01-16', 'datum_max': '2020-02-17',
        # 'source': '/Users/tk/py/retrieval-gen/bürgerportal_dui/attachments/2025/09/1632003_Dokument.pdf',
        # 'ocr': True, 'ocr_languages': 'deu+eng', 'chunk': 1, 'n_tokens': 24, 'char_start': 37, 'char_end': 458,
        # 'chunk_id': 'cc86a58cc14752ad9b7d1370c777e9e2-000001-c0430fb75cbe',
        # 'text': 'der Theodor-König-Gesamtschule in Duisburg-Meiderich/Beeck Beschlussentwurf',
        # 'text_length': 75
        # }

        # self.hits[0]
        # {
        # 'datum_max': '2020-02-17',
        # 'chunk_id': 'cc86a58cc14752ad9b7d1370c777e9e2-000001-c0430fb75cbe',
        # 'datum_min': '2020-01-16',
        # 'betreff': 'Einrichtung eines befristeten Teilstandortes an der Theodor-König-Gesamtschule in Duisburg-Meiderich/Beeck',
        # 'typ': None,
        # 'vorlage_id': '19-1389',
        # 'aktenzeichen': 'III/40-13 Tönges, 2522'
        # }

        # add to this
        # 'source': '/Users/tk/py/retrieval-gen/bürgerportal_dui/attachments/2025/09/1632003_Dokument.pdf',
        # 'text': 'der Theodor-König-Gesamtschule in Duisburg-Meiderich/Beeck Beschlussentwurf',
        # 'betreff' + 'text' (for re-ranking)

        self.dense_hits = []
        kord = ['qtype',  'chunk_id', 'vorlage_id', 'aktenzeichen', 'typ', 'betreff', 'text','hit_score_norm', 'hit_score', 'hit_id', 'source',  'text_betr', 'datum_max', 'datum_min',]
        for hit, sn in zip(self.hits, self.score_norm):

            chunk_id = hit.payload['chunk_id']
            if chunk_id in hits_:
                text = hits_[chunk_id]['text']
                source = hits_[chunk_id]['source']
            else:
                text = ''
                source = None

            text_betr = ". ".join([hit.payload['betreff'], text])

            out = hit.payload | {'qtype': typ, 'text': text, 'text_betr': text_betr, 'hit_score_norm': sn, 'hit_score': hit.score, 'hit_id': hit.id, 'source': source}

            out_re = {x: out[x] for x in kord}
            self.dense_hits.append(out_re)


        # self.dense_hits = [{'qtype': typ} | hit.payload | {'text': hits_[hit.payload['chunk_id']]['text'] if hit.payload['chunk_id'] in hits_ else ''} | {'hit_score_norm': sn, 'hit_score': hit.score, 'hit_id': hit.id} for hit, sn in zip(self.hits, self.score_norm)]

        # self.dense_norm = [('dense', i, score, hit.payload['betreff'] ) for i, (score, hit) in enumerate(zip(self.score_norm, self.hits))]


    def sparse_search(self, q:str, k=100):

        typ = 'sparse'

        self.query = q
        response = ops.search(
            index=index_name,
            body={
                "size": k,
                "query": {
                    "multi_match": {
                        "query": self.query,
                        "fields": ["betreff", "text", ],
                        "operator": "and"  # OR is default, can switch to "and"
                    }
                }
            }
        )


        # kord = ['qtype', 'chunk_id', 'vorlage_id', 'aktenzeichen', 'typ', 'betreff', 'text', 'hit_score_norm',
        #         'hit_score', 'hit_id', 'source', 'text_betr', 'datum_max', 'datum_min', ]

        self.sparse_hits = []
        if len(response["hits"]["hits"]) > 0:

            # kkeep = ['chunk_id', 'vorlage_id', 'typ', 'aktenzeichen', 'datum_max', 'datum_min', 'betreff', 'text',
            #          'source']

            scores = [hit['_score'] for hit in response["hits"]["hits"]]
            scores_norm = self.minmax(scores)

            for hit, s, sn in zip(response["hits"]["hits"], scores, scores_norm):
                d1 = {'qtype': typ, 'chunk_id': hit['_source']['chunk_id']}
                d2 = {k: hit['_source'][k] for k in [ 'vorlage_id', 'aktenzeichen', 'typ', 'betreff', 'text',]}
                d3 = {'hit_score_norm': sn, 'hit_score': s,}
                d4 = {'source': hit['_source']['source'], 'text_betr': ". ".join([hit['_source']['betreff'], hit['_source']['text']])}
                d5 = {k: hit['_source'][k] for k in [ 'datum_max', 'datum_min',]}

                out = d1 | d2 | d3 | d4 | d5

                self.sparse_hits.append(out)

        # if len(response["hits"]["hits"]) > 0:
        #     scores = self.minmax([hit['_score'] for hit in response["hits"]["hits"]])
        #     self.spars_norm = [('spars', i, score, d['_source']['text']) for i, (score, d) in
        #                        enumerate(zip(scores, response["hits"]["hits"]))]
        # else:
        #     self.spars_norm = []



    @staticmethod
    def minmax(x:list):
        if not x: return []
        x_min = min(x); x_max = max(x)
        if x_max == x_min:
            # all scores identical -> pick a safe constant (0.0 or 1.0)
            return [0.0 for _ in x]
        width = max(x)-min(x)
        return [(xx-x_min)/width for xx in x]


    def query_db(self, q:str):
        self.dense_search(q)
        self.sparse_search(q)

    def rerank_search_results(self, return_n:int=10):
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # res_cbn = sorted(self.spars_norm + self.dense_norm, key=lambda x: x['hit_score_norm'])
        res_cbn = sorted(self.sparse_hits + self.dense_hits, key=lambda x: x['hit_score_norm'])

        pairs = [(self.query, doc['text_betr']) for doc in res_cbn]
        scores = reranker_model.predict(pairs)
        print(scores)

        out = sorted(zip(scores, res_cbn), key=lambda x: -x[0])

        return out[0:return_n]


        #
        # out = []
        # for k, (s, d) in enumerate(sorted(zip(scores, res_cbn), key=lambda x: -x[0])):
        #     if d[0] == 'spars':
        #         add = self.spars_norm[d[1]]
        #     elif d[0] == 'dense':
        #         add = self.dense_norm[d[1]]
        #     else:
        #         raise ValueError('check query type')
        #     out.append(add)
        #     if k > return_n: break
        #
        # return out


retr = Retriever(ops, model, qdrant_client)


# qq = 'Bundesgartenschau'
# qq = 'Familienzentrum Wrangelstraße'
# qq = 'Bärenstark und Fuchsschlau'
# self = retr
# self.query_db(q = qq)
# test = self.rerank_search_results()






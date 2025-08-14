import re
from typing import List, Dict
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import os, json


class Retriever:
    ### embedds query and retrieves chunks/tops related (nearest dist.)
    ### method: search dense - searching with dist of chunks & tops
    ### method: search hypbrid - searching chunks and re-score them using term frequency method (TF-IDF)

    def __init__(self, transformer: SentenceTransformer, simMat_dense: faiss, simMat_top: faiss, pl_dense, pl_top):

        self.payloads_top = pl_top
        self.simMat_top = simMat_top

        self.payload_dense = pl_dense
        self.simMat_dense = simMat_dense

        self.transformer = transformer

        # bm25 = BM25Okapi([t.split() for t in chunk_texts])
        self.bm25 = BM25Okapi([t['text'].split() for t in self.payload_dense])

    def search_hybrid(self, query: str, k=8, alpha=0.6):
        vecs = self.transformer.encode(
            query,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity ready
            show_progress_bar=False,
        )
        qv = (vecs.astype("float32")[..., np.newaxis]).T

        # dense
        ds, di = self.simMat_top.search(qv, k)
        dense_scores = {i: float(s) for s, i in zip(ds[0], di[0]) if i != -1}
        # sparse
        bm_scores = self.bm25.get_scores(query.split())
        # combine (simple linear)
        fused = []
        for i, d in dense_scores.items():
            s = alpha * d + (1 - alpha) * (bm_scores[i] / (max(bm_scores) + 1e-9))
            fused.append((s, i))
        fused.sort(reverse=True)
        return [(s, self.payload_dense[i]) for s, i in fused[:k]]

    def search_dense(self, query: str, k_top: int = 5, k_chunk: int = 8):
        vecs = self.transformer.encode(
            query,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity ready
            show_progress_bar=False,
        )
        qv = (vecs.astype("float32")[..., np.newaxis]).T

        # search tops
        ts, ti = self.simMat_top.search(qv, k_top)
        top_hits = [(float(s), self.payloads_top[idx]) for s, idx in zip(ts[0], ti[0]) if idx != -1]

        # search chunks
        cs, ci = self.simMat_dense.search(qv, k_chunk)
        chunk_hits = [(float(s), self.payload_dense[idx]) for s, idx in zip(cs[0], ci[0]) if idx != -1]

        return top_hits, chunk_hits

class Transformer:
    ### class for creating similarity matrices from tops and text chunks:
    #### 1. select embedding model and embed text for top/chunk -> embedding vector
    #### 2. normalise vector length for cosine dist
    #### 3. calc similarity between embedding vectors

    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.no_process = []
        self.chunk_payloads = []
        self.top_payloads = []
        self.emb_chunks_mat = []
        self.emb_top_mat = []

    def create_faiss(self, dist: str = 'cos'):

        if dist == 'cos':
            # normalise len
            self.emb_top_norm = self.emb_top / np.linalg.norm(self.emb_top)
            self.emb_chunk_norm =  self.emb_chunk / np.linalg.norm(self.emb_chunk)
        else:
            self.emb_top_norm = self.emb_top
            self.emb_chunk_norm = self.emb_chunk

        self.index_top = faiss.IndexFlatIP(self.emb_top_norm.shape[1])
        self.index_top.add(self.emb_top_norm)

        self.index_chunk = faiss.IndexFlatIP(self.emb_chunk_norm.shape[1])
        self.index_chunk.add(self.emb_chunk_norm)


    def run(self, path='corpus/chunks', pattern=''):

        for a, b, c in os.walk(path):
            for fh in c:
                if pattern in fh:
                    with open(os.path.join(path, fh), 'r') as f:
                        data = f.read()

                    dj = json.loads(data)
                    top_id = f"{dj['meta']['meeting_date']}_{dj['meta']['top_id']}"

                    try:
                        ### top summary embedding
                        oj = dj['output_json']

                        top_text = self.build_top_summary_text(oj)
                        emb_top = self.embed(top_text)
                        self.emb_top_mat.append(emb_top)
                        top_pl = {"kind": "top", "top_key": top_id, "meta": dj['meta'], "text": top_text}
                        self.top_payloads.append(top_pl)

                        ### text chunk embedding
                        chunk_inp = dj['user_input']
                        chunks = self.simple_paragraph_chunks(chunk_inp)
                        chunk_texts = [c["text"] for c in chunks]
                        chunk_vecs = self.embed(chunk_texts)
                        self.emb_chunks_mat.append(chunk_vecs)

                        for i, c in enumerate(chunks):
                            chunk_id = f"{top_id}_p{i}"
                            self.chunk_payloads.append({
                                "kind": "chunk",
                                "chunk_id": chunk_id,
                                "top_key": top_id,
                                "meta": {**dj['meta'], "span": [c["start"], c["end"]]},
                                "text": c["text"]
                            })
                    except:
                        print(top_id)
                        self.no_process.append(top_id)

        self.emb_top = np.matrix(self.emb_top_mat)
        self.emb_chunk = np.matrix([y for x in self.emb_chunks_mat for y in x])

        self.create_faiss(dist='cos')

    @staticmethod
    def build_top_summary_text(output_json) -> str:
        """Compact, search-friendly summary text for TOP-level embedding."""
        parts = []
        title =  output_json.get("titel") or  output_json.get("topic")
        if title:
            parts.append(f"Titel: {title}")
        kf =  output_json.get("kurzfassung")
        if kf:
            parts.append(f"Kurzfassung: {kf}")

        # Measures / decisions – very useful for retrieval
        for s in  output_json.get("massnahmen_beschluesse", [])[:6]:
            parts.append(f"Beschluss: {s}")

        # Legal references & risks improve matching for queries with statutes
        rl =  output_json.get("rechtliche_bezuege", [])
        if rl:
            parts.append("Recht: " + "; ".join(rl[:6]))

        # Provenance snippets help anchor exact wording
        prov =  output_json.get("provenienz", [])
        for p in prov[:3]:
            parts.append(f"Zitat: {p.get('auszug')}")

        # Optional: zeitliche Bezüge
        zb =  output_json.get("zeitraum_bezug", [])
        if zb:
            parts.append("Zeitraum: " + ", ".join(zb[:6]))

        return "\n".join(parts)

    @staticmethod
    def simple_paragraph_chunks(text, max_chars: int = 1200, overlap: int = 150) -> List[Dict]:
        """Split into roughly paragraph-sized windows with overlap (char-based approximation)."""
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + max_chars, n)
            # try to end at a sentence boundary close to end
            boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary - start < max_chars * 0.5:
                boundary = end
            chunk = text[start:boundary].strip()
            if chunk:
                chunks.append({"text": chunk, "start": start, "end": boundary})
            if boundary >= n:
                break
            start = max(0, boundary - overlap)
        return chunks

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity ready
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    # # 2) build dense index (cosine via dot product on normalized vectors)
    # @staticmethod
    # def build_faiss_index(X: np.ndarray) -> faiss.Index:
    #     index = faiss.IndexFlatIP(X.shape[1])  # inner product == cosine on normalized vectors
    #     index.add(X)
    #     return index
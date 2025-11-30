import json
import numpy as np

from dataclasses import dataclass
from typing import List
from langchain_ollama import OllamaEmbeddings

@dataclass
class VectorDoc:
    id: str
    text: str
    embedding: np.ndarray

class VectorDB:
    def __init__(self, model="nomic-embed-text"):
        self.emb = OllamaEmbeddings(model=model)
        self.docs = []

    def embed(self, texts):     return np.array(self.emb.embed(texts), dtype="float32")
    def embed_query(self, text): return np.array(self.emb.embed_query(text), dtype="float32")

    def add(self, texts, ids):
        vectors = self.embed(texts)
        for t,i,v in zip(texts, ids, vectors):
            self.docs.append(VectorDoc(i,t,v))

    def search(self, query, k=5, threshold=0.45):
        q = self.embed_query(query)
        matrix = np.stack([d.embedding for d in self.docs])
        sims = matrix@q/(np.linalg.norm(matrix,axis=1)*np.linalg.norm(q))
        ranked = [(self.docs[i],float(s)) for i,s in enumerate(sims) if s>=threshold]
        return sorted(ranked, key=lambda x:-x[1])[:k]

    def save(self, path="storage/vector_store.json"):
        with open(path,"w") as f: json.dump([{**d.__dict__,"embedding":d.embedding.tolist()} for d in self.docs],f)

    def load(self,path="storage/vector_store.json"):
        with open(path) as f: data=json.load(f)
        self.docs=[VectorDoc(d["id"],d["text"],np.array(d["embedding"])) for d in data]

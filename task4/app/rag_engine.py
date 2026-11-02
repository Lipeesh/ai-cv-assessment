"""
Retrieval-Augmented Generation (RAG) Engine for Industrial Safety & SOP Assistance.
Embeds safety manuals, retrieves top-k relevant sections, and synthesizes grounded answers.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class IndustrialSafetyRAGEngine:
    def __init__(
        self,
        corpus_path: str = "/home/ub/Desktop/COMBINATION/ai-cv-assessment/task4/data/industrial_sop_corpus.json",
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.corpus_path = corpus_path
        self.model_name = model_name
        self.device = device
        
        print(f"[RAG Engine] Loading embedding model '{self.model_name}' on {self.device}...")
        self.embedder = SentenceTransformer(self.model_name, device=self.device)
        
        self.documents: List[Dict[str, Any]] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.load_and_index_corpus()

    def load_and_index_corpus(self):
        """Load corpus JSON and compute normalized vector embeddings for each document."""
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus not found at: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        print(f"[RAG Engine] Indexing {len(self.documents)} SOP documents...")
        # Prepare rich textual representations
        texts = [
            f"Title: {doc['title']}\nCategory: {doc['category']}\nContent: {doc['content']}\nOSHA: {doc.get('osha_reference', '')}"
            for doc in self.documents
        ]
        raw_embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize for fast cosine similarity via dot product
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.doc_embeddings = raw_embs / norms
        print(f"[RAG Engine] Indexing complete. Embedding matrix shape: {self.doc_embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top_k documents by cosine similarity to query."""
        q_emb = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = np.dot(self.doc_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.documents[idx].copy()
            doc["similarity_score"] = round(float(scores[idx]), 4)
            doc["rank"] = rank + 1
            results.append(doc)

        return results

    def synthesize_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Synthesize grounded, actionable industrial safety guidance with citations.
        """
        if not retrieved_docs or retrieved_docs[0]["similarity_score"] < 0.25:
            return (
                "Insufficient procedural context found in the SOP database for this inquiry. "
                "Please consult the site Safety Officer or Environmental Health & Safety (EHS) supervisor."
            )

        top_doc = retrieved_docs[0]
        
        # Build synthesis grounded strictly in retrieved context
        bullets = []
        for doc in retrieved_docs:
            if doc["similarity_score"] >= 0.35:
                bullets.append(f"• [{doc['doc_id']} — {doc['title']}]: {doc['content']}")

        answer = (
            f"According to standard plant safety protocols ({top_doc.get('osha_reference', 'Industrial SOP')}):\n\n"
            + "\n\n".join(bullets)
            + f"\n\nApplicable Enforcement Zones: {', '.join(top_doc.get('applicable_zones', ['Plant-wide']))}."
        )
        return answer

    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """End-to-end RAG query interface."""
        t0 = time.time()
        retrieved = self.retrieve(query, top_k=top_k)
        answer = self.synthesize_answer(query, retrieved)
        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved,
            "latency_ms": latency_ms,
            "top_similarity_score": retrieved[0]["similarity_score"] if retrieved else 0.0
        }

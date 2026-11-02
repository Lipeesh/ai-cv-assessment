# Task 4 — Industrial Safety & Maintenance RAG Assistant

## Overview
This module implements a production-grade Retrieval-Augmented Generation (RAG) API service tailored for industrial plants, construction sites, and manufacturing facilities. The system indexes plant Standard Operating Procedures (SOPs), OSHA safety standards, PPE requirements, lock-out/tag-out (LOTO) procedures, and high-voltage electrical clearances, providing real-time, grounded safety advisories with citations.

---

## Deliverables Included
1. **Application Code**:
   - `app/rag_engine.py`: Vector indexing and retrieval engine using dense semantic embeddings (`all-MiniLM-L6-v2`), cosine similarity ranking, and grounded citation synthesis.
   - `app/main.py`: Asynchronous FastAPI REST service with `/query`, `/documents`, and `/health` endpoints.
2. **Knowledge Base**:
   - `data/industrial_sop_corpus.json`: Structured plant safety standard operating procedures and regulatory references (OSHA 1926.100, OSHA 1910.212, OSHA 1910.147, NFPA 70E).
3. **Outputs**:
   - `outputs/results.json`: Quantitative benchmark results across 5 industrial safety evaluation queries including similarity scores, response latencies, and citations.

---

## System Architecture

```
User Query (via REST API / cURL)
             │
             ▼
   FastAPI Endpoint (/query)
             │
             ▼
 Dense Semantic Embedding (all-MiniLM-L6-v2)
             │
             ▼
Cosine Similarity Index Search (SOP Corpus)
             │
             ▼
  Grounded Context Retrieval (Top-K)
             │
             ▼
  Synthesizer with OSHA & SOP Citations
             │
             ▼
 JSON Response (Answer, Sources, Similarity, Latency)
```

---

## Quickstart: Running the RAG Service

### 1. Start the API Server
```bash
# Navigate to repository root
cd /home/ub/Desktop/COMBINATION/ai-cv-assessment

# Launch FastAPI server using Uvicorn on port 8088
uvicorn task4.app.main:app --host 0.0.0.0 --port 8088 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://localhost:8088/docs`.

---

## API Documentation & `curl` Examples

### 1. Health Check
```bash
curl -X GET http://localhost:8088/health
```
**Example Response**:
```json
{
  "status": "healthy",
  "documents_indexed": 6,
  "embedding_dimensions": 384,
  "device": "cpu"
}
```

---

### 2. Query Safety Protocols (`POST /query`)

#### Example A: PPE Compliance in Restricted Work Zones
```bash
curl -X POST http://localhost:8088/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the hard hat requirements and PPE rules for workers in Zone Alpha?",
       "top_k": 2
     }'
```

**Example Response**:
```json
{
  "query": "What are the hard hat requirements and PPE rules for workers in Zone Alpha?",
  "answer": "According to standard plant safety protocols (OSHA 1926.100(a)):\n\n• [SOP-PPE-001 — Industrial Personal Protective Equipment (PPE) Compliance Standard]: All personnel, contractors, and visitors entering active industrial fabrication zones or heavy construction areas must wear Type I or Type II hard hats conforming to ANSI/ISEA Z89.1-2014. Hard hats must be inspected daily for cracks, dents, and suspension strap degradation. In designated Zone Alpha and overhead crane operating envelopes, Class E (electrical rated up to 20,000V) or Class G (rated up to 2,200V) hard hats are mandatory. Personnel detected without safety helmets in restricted zones must be immediately alerted, and work must halt until compliance is restored.\n\nApplicable Enforcement Zones: zone_alpha_danger, crane_envelope, fabrication_bay.",
  "retrieved_documents": [
    {
      "doc_id": "SOP-PPE-001",
      "title": "Industrial Personal Protective Equipment (PPE) Compliance Standard",
      "category": "PPE_Policy",
      "content": "All personnel, contractors, and visitors entering active industrial fabrication zones or heavy construction areas must wear Type I or Type II hard hats conforming to ANSI/ISEA Z89.1-2014...",
      "applicable_zones": ["zone_alpha_danger", "crane_envelope", "fabrication_bay"],
      "osha_reference": "OSHA 1926.100(a)",
      "similarity_score": 0.7272,
      "rank": 1
    }
  ],
  "latency_ms": 8.52,
  "top_similarity_score": 0.7272
}
```

---

#### Example B: Idle Worker Detection & Dwell Protocols
```bash
curl -X POST http://localhost:8088/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What happens if a worker stays idle in a restricted hazard zone for more than 5 seconds?",
       "top_k": 2
     }'
```

---

#### Example C: Lockout / Tagout (LOTO) Before Machinery Maintenance
```bash
curl -X POST http://localhost:8088/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the proper Lockout Tagout (LOTO) procedure before servicing an induction motor or pump?",
       "top_k": 2
     }'
```

---

### 3. List All Indexed SOP Documents
```bash
curl -X GET http://localhost:8088/documents
```

---

## Benchmark Results (Summary from `outputs/results.json`)

| Query ID | Topic / Focus | Top Matched Document | Cosine Similarity | Latency (ms) |
| :---: | :--- | :---: | :---: | :---: |
| **Q1** | PPE Rules in Zone Alpha | `SOP-PPE-001` | **0.7272** | 9.14 ms |
| **Q2** | 5s Idle Dwell Violation | `SOP-ZONE-002` | **0.6566** | 8.28 ms |
| **Q3** | LOTO Motor Maintenance | `SOP-LOTO-003` | **0.4854** | 8.34 ms |
| **Q4** | 11kV Flash Clearances | `SOP-VOLT-004` | **0.6539** | 7.99 ms |
| **Q5** | OCR Asset Voltage Mismatch | `SOP-EQUIP-005` | **0.6109** | 7.79 ms |
| **Average** | **Comprehensive EHS Queries** | — | **0.6268** | **8.31 ms** |

Full query responses and structured JSON outputs are archived in `outputs/results.json`.

# RAG + Attention Visualization System

This project provides a compact, end-to-end RAG demo built around a synthetic story dataset: **"The Strange Day of Tommy and the Blue Grass."** It is designed to make RAG gains easy to observe in both generated answers and attention behavior.

## Project Requirements Coverage

This implementation satisfies the assignment goals:

1. **Index a document collection**: the story is chunked by paragraph and indexed with FAISS embeddings.
2. **Front-facing website**: a FastAPI + HTML/CSS/JavaScript UI supports question selection, custom queries, and parameter tuning.
3. **Query an LLM with RAG**: users can set `k_retrieve` and `k_context`.
4. **Show answers before and after RAG**: side-by-side non-RAG vs RAG answer panels.
5. **Show comparative heatmap**: IzzyViz comparison heatmaps are generated for non-RAG vs RAG attention.
6. **Prepare demo highlighting major changes**: see `DEMO_GUIDE.md` for curated queries where outputs and attention differ substantially.

## Dataset

- **Story file**: `data/story.txt`
- **Question set**: `data/questions.json` (20 benchmark-style factual questions)

The story intentionally contains unusual facts (blue grass, a barking cat, "2 + 2 = 7", etc.) so that correct answers should come from retrieval rather than model priors.

## Architecture

- `backend/rag_system.py`: document loading, chunking, indexing, retrieval, reranking, generation, attention extraction.
- `backend/attention_visualizer.py`: IzzyViz integration for individual and comparative heatmaps.
- `backend/app.py`: FastAPI API and static frontend serving.
- `frontend/index.html`, `frontend/style.css`, `frontend/app.js`: interactive UI.

## Quick Start

```bash
cd rag_attention_viz
pip install -r requirements.txt
cd ..
pip install .
cd rag_attention_viz/backend
python app.py
```

Open: `http://localhost:8000`

## Web UI Features

- Preset question dropdown + custom question input.
- RAG controls:
  - `k_retrieve` (3–20)
  - `k_context` (1–10)
- Attention controls:
  - `layer` (-1 to 11)
  - `head` (0 to 11)
- Side-by-side answer comparison:
  - ❌ Without RAG
  - ✅ With RAG
- Retrieved document list with context-used markers.
- Heatmap panels:
  - Comparative prefill heatmap (IzzyViz circles)
  - Non-RAG prefill
  - RAG prefill
  - Non-RAG decode stage
  - RAG decode stage

## API Endpoints

- `GET /api/health`
- `GET /api/questions`
- `POST /api/query`
- `GET /api/visualization/{filename}`

## Engineering Notes

- Code comments and user-facing strings are in English.
- Documentation is fully in English and consistent with the actual dataset and benchmark questions.
- The demo guide includes a requirement-traceable presentation flow for grading.

## Testing

You can run:

```bash
cd rag_attention_viz
python test_system.py
```

> Note: the first run may be slow due to model downloads.

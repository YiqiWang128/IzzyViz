"""
FastAPI Backend for RAG + Attention Visualization System
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os
from pathlib import Path
import uvicorn
import base64

from rag_system import RAGSystemWithAttention
from attention_visualizer import AttentionVisualizer

# Initialize FastAPI app
app = FastAPI(title="RAG Attention Visualization API",
              description="API for RAG system with attention visualization")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system and visualizer (lazy loading)
rag_system = None
visualizer = None

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "attention_heatmaps"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    k_retrieve: int = 10
    k_context: int = 3
    layer: int = -1
    head: int = 0
    max_tokens: int = 48


class QueryResponse(BaseModel):
    query: str
    answer_no_rag: str
    answer_with_rag: str
    retrieved_docs: List[str]
    context_used: Optional[str]
    visualization_paths: dict
    tokens_no_rag_length: int
    tokens_rag_length: int


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global rag_system, visualizer

    print("="*60)
    print("Starting RAG Attention Visualization Server...")
    print("="*60)

    try:
        # Initialize RAG system
        print("\n[1/2] Initializing RAG System...")
        rag_system = RAGSystemWithAttention(
            document_path=str(DATA_DIR / "story.txt")
        )
        print("✓ RAG System ready")

        # Initialize visualizer
        print("\n[2/2] Initializing Attention Visualizer...")
        visualizer = AttentionVisualizer(output_dir=str(OUTPUT_DIR))
        print("✓ Visualizer ready")

        # Check frontend
        print("\n[3/3] Checking Frontend...")
        if FRONTEND_DIR.exists():
            index_html = FRONTEND_DIR / "index.html"
            if index_html.exists():
                print(f"✓ Frontend ready at: {FRONTEND_DIR}")
                print(f"✓ index.html found")
            else:
                print(f"⚠ Warning: index.html not found in {FRONTEND_DIR}")
        else:
            print(f"⚠ Warning: Frontend directory not found: {FRONTEND_DIR}")

        print("\n" + "="*60)
        print("Server is ready to accept requests!")
        print("Open browser at: http://localhost:8000")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system_loaded": rag_system is not None,
        "visualizer_loaded": visualizer is not None
    }


@app.get("/api/questions")
async def get_questions():
    """Get all available questions"""
    try:
        questions_path = DATA_DIR / "questions.json"
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading questions: {str(e)}")


@app.post("/api/query")
async def process_query(request: QueryRequest):
    """
    Process a query with RAG and generate attention visualizations

    This endpoint:
    1. Queries the LLM with and without RAG
    2. Extracts attention weights from both
    3. Generates comparison visualizations
    4. Returns results with paths to visualizations
    """
    global rag_system, visualizer

    if rag_system is None or visualizer is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        print(f"\n{'='*60}")
        print(f"Processing Query: {request.question}")
        print(f"Parameters: k_retrieve={request.k_retrieve}, k_context={request.k_context}")
        print(f"Visualization: layer={request.layer}, head={request.head}")
        print(f"{'='*60}\n")

        # Get comparison results from RAG system
        results = rag_system.query_with_comparison(
            query=request.question,
            k_retrieve=request.k_retrieve,
            k_context=request.k_context
        )

        # Extract results
        no_rag_result = results['no_rag']
        rag_result = results['with_rag']

        # Safely get tokens and retrieved docs with None checks
        tokens_no_rag = no_rag_result.get('tokens', [])
        tokens_rag = rag_result.get('tokens', [])
        retrieved_docs = rag_result.get('retrieved_docs', [])
        token_type_ids_no_rag = no_rag_result.get('token_type_ids', None)
        token_type_ids_rag = rag_result.get('token_type_ids', None)

        # Ensure they're not None
        if tokens_no_rag is None:
            tokens_no_rag = []
        if tokens_rag is None:
            tokens_rag = []
        if retrieved_docs is None:
            retrieved_docs = []

        # Create visualizations
        print("\n" + "="*60)
        print("Generating Attention Visualizations...")
        print("="*60)

        visualization_paths = visualizer.visualize_comparison(
            tokens_no_rag=tokens_no_rag,
            attentions_no_rag=no_rag_result.get('attentions'),
            tokens_rag=tokens_rag,
            attentions_rag=rag_result.get('attentions'),
            layer=request.layer,
            head=request.head,
            max_tokens=request.max_tokens,
            save_name=f"query_{hash(request.question)}.pdf",
            query=request.question,
            token_type_ids_no_rag=token_type_ids_no_rag,
            token_type_ids_rag=token_type_ids_rag
        )

        # Convert PDFs to base64 for embedding in HTML (optional)
        # For now, we'll just return paths

        response = {
            "query": request.question,
            "answer_no_rag": no_rag_result.get('answer', 'No answer generated'),
            "answer_with_rag": rag_result.get('answer', 'No answer generated'),
            "retrieved_docs": retrieved_docs[:request.k_retrieve],
            "context_used": rag_result.get('context', ''),
            "visualization_paths": visualization_paths,
            "tokens_no_rag_length": len(tokens_no_rag),
            "tokens_rag_length": len(tokens_rag),
            "parameters": {
                "k_retrieve": request.k_retrieve,
                "k_context": request.k_context,
                "layer": request.layer,
                "head": request.head
            }
        }

        print("\n" + "="*60)
        print("✓ Query processed successfully")
        print("="*60 + "\n")

        return response

    except Exception as e:
        print(f"\n❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization PDF files"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename
    )


@app.get("/api/visualization_image/{filename}")
async def get_visualization_as_image(filename: str):
    """
    Convert PDF visualization to image and serve
    (Requires pdf2image library)
    """
    try:
        from pdf2image import convert_from_path
        import io

        file_path = OUTPUT_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Visualization not found")

        # Convert first page of PDF to image
        images = convert_from_path(file_path, first_page=1, last_page=1)

        if not images:
            raise HTTPException(status_code=500, detail="Failed to convert PDF to image")

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return as base64
        img_base64 = base64.b64encode(img_byte_arr.read()).decode()

        return {"image": f"data:image/png;base64,{img_base64}"}

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="pdf2image not installed. Install with: pip install pdf2image"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting PDF: {str(e)}")


# Mount frontend static files - this must be AFTER all API route definitions
if FRONTEND_DIR.exists():
    print(f"Mounting frontend from: {FRONTEND_DIR}")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {FRONTEND_DIR}")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   RAG + Attention Visualization Server                       ║
    ║   Based on "The Strange Day of Tommy and the Blue Grass"     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

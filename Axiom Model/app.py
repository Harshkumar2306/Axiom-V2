import os
import json
import torch
import tiktoken
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Optional, List, Dict, Any

from generate import load_model, generate_stream_generator
from rag_engine import RAGEngine, WebSearchEngine

from contextlib import asynccontextmanager

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
ENC = tiktoken.get_encoding("cl100k_base")
rag_engine: Optional[RAGEngine] = None
web_engine: Optional[WebSearchEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, rag_engine, web_engine
    checkpoint_path = "best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: Model not found at {checkpoint_path}!")
    else:
        MODEL, val_loss = load_model(checkpoint_path, DEVICE)
        print(f"✅ Model loaded successfully! (Val Loss: {val_loss})")

    # Initialize RAG & Web Search Engine
    try:
        rag_engine = RAGEngine()
        web_engine = WebSearchEngine(rag_encoder=rag_engine.encoder)
        print("✅ Deep RAG & Web Search Engines initialized.")
    except Exception as e:
        print(f"⚠️ Failed to initialize RAG/Web engines: {e}")
    yield
    print("🛑 Axiom server shutting down...")

app = FastAPI(title="Axiom AI Engine with Deep RAG & Live Web Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2
    system_prompt: Optional[str] = None
    repetition_penalty: float = 1.05
    web_search: bool = False
    rag_search: bool = True

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if MODEL is None:
        return {"response": "Model is not loaded. Please check server logs.", "speed": "0.0"}
        
    print(f"📩 Chat request: max_tokens={request.max_tokens}, temperature={request.temperature}, web_search={request.web_search}")
    base_sys = request.system_prompt or "You are a highly intelligent, logical, and helpful AI assistant named Axiom."
    
    collected_sources: List[Dict[str, Any]] = []
    context_sections: List[str] = []

    # 🌐 1. Live Web Search with Deep Content Scraping
    if request.web_search and web_engine:
        try:
            print(f"🔍 Executing live web search for: '{request.prompt}'...")
            search_data = web_engine.search_and_deep_scrape(request.prompt, max_results=3)
            if search_data.get("context"):
                context_sections.append(search_data['context'])
                for s in search_data.get("sources", []):
                    collected_sources.append({
                        "type": "web",
                        "title": s.get("title", "Web Result"),
                        "url": s.get("url", ""),
                        "snippet": s.get("snippet", "")
                    })
                print(f"✅ Web context injected ({len(collected_sources)} sources).")
        except Exception as e:
            print(f"⚠️ Web search error: {e}")

    # 📚 2. Local Document Vector RAG Search
    if request.rag_search and rag_engine and len(rag_engine.chunks) > 0:
        try:
            doc_matches = rag_engine.search(request.prompt, top_k=3, min_similarity=0.25)
            if doc_matches:
                doc_context_items = []
                for idx, match in enumerate(doc_matches):
                    src_name = match.get("source", "Document")
                    text_content = match.get("text", "")
                    clean_doc_slice = text_content.strip()[:350]
                    doc_context_items.append(f"- From {src_name}: {clean_doc_slice}")
                    collected_sources.append({
                        "type": "doc",
                        "title": src_name,
                        "url": "",
                        "snippet": text_content[:180] + "..." if len(text_content) > 180 else text_content
                    })
                context_sections.append("\n".join(doc_context_items))
                print(f"✅ Document RAG injected ({len(doc_matches)} matching chunks).")
        except Exception as e:
            print(f"⚠️ Document RAG search error: {e}")

    # Inject grounded context cleanly without rogue turn delimiters
    if context_sections:
        base_sys += "\n\nReference Information:\n" + "\n".join(context_sections)
        base_sys += "\n\nInstruction: Answer the user's question clearly and completely in full sentences based on the reference information above."

    # Universal Structural Formatting Injection for Small Models
    formatting_rule = " Format your response clearly. Use markdown code blocks for code, standard structural formatting for letters/emails (greetings, line breaks, sign-offs), and clear headings or bullet points for lists and essays."
    if "format" not in base_sys.lower():
        base_sys += formatting_rule

    formatted_prompt = f"### System:\n{base_sys}\n\n### User:\n{request.prompt}\n\n### Assistant:\n"
    rep_penalty = request.repetition_penalty if request.repetition_penalty is not None else 1.05

    def token_generator():
        generator = generate_stream_generator(
            MODEL, 
            ENC, 
            formatted_prompt, 
            max_new_tokens=request.max_tokens, 
            temperature=request.temperature, 
            repetition_penalty=rep_penalty
        )
        for chunk in generator:
            yield f"data: {chunk}\n\n"

        # Emit structured citations / sources to the client
        if collected_sources:
            sources_json = json.dumps(collected_sources)
            yield f"data: __AXIOM_SOURCES__{sources_json}__\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

# --- Document RAG Management APIs ---

@app.post("/rag/upload")
async def upload_rag_document(file: UploadFile = File(...)):
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine is not initialized")
    try:
        content = await file.read()
        res = rag_engine.ingest_file(file.filename, content)
        return {"status": "success", "data": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/rag/documents")
async def list_rag_documents():
    if rag_engine is None:
        return {"documents": []}
    return {"documents": rag_engine.list_documents()}

@app.delete("/rag/documents")
async def delete_rag_documents(filename: Optional[str] = None):
    if rag_engine is None:
        return {"status": "ok", "documents": []}
    if filename:
        rag_engine.delete_document(filename)
    else:
        rag_engine.clear()
    return {"status": "ok", "documents": rag_engine.list_documents()}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
        "rag_chunks": len(rag_engine.chunks) if rag_engine else 0,
        "rag_documents": len(rag_engine.list_documents()) if rag_engine else 0
    }

@app.get("/")
async def serve_ui():
    template_path = "templates/index.html"
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(
        content="""
        <html>
            <body style="background:#09090b;color:#f4f4f5;font-family:sans-serif;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;margin:0;">
                <h1 style="color:#06b6d4;">Axiom AI Backend Engine is Running</h1>
                <p>FastAPI inference server with Deep RAG & Live Web Search active.</p>
                <p>Open <a href="http://localhost:5173" style="color:#38bdf8;">http://localhost:5173</a> to use the React UI.</p>
                <p>Explore <a href="/docs" style="color:#38bdf8;">/docs</a> for Swagger API specifications.</p>
            </body>
        </html>
        """,
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")

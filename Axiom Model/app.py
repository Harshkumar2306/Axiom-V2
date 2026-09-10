import os
import torch
import tiktoken
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Optional

from generate import load_model, generate_stream_generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
ENC = tiktoken.get_encoding("cl100k_base")

@app.on_event("startup")
async def startup_event():
    global MODEL
    checkpoint_path = "best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: Model not found at {checkpoint_path}!")
        return
        
    MODEL, val_loss = load_model(checkpoint_path, DEVICE)
    print(f"✅ Model loaded successfully! (Val Loss: {val_loss})")

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 300
    temperature: float = 0.2
    system_prompt: Optional[str] = None
    repetition_penalty: float = 1.05

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if MODEL is None:
        return {"response": "Model is not loaded. Please check server logs.", "speed": "0.0"}
        
    base_sys = request.system_prompt or "You are a highly intelligent, logical, and helpful AI assistant named Axiom."
    
    prompt_lower = request.prompt.lower()
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
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.get("/")
async def serve_ui():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")

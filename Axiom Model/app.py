import os
import torch
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate import load_model, generate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
ENC = None
DEVICE = None

@app.on_event("startup")
async def startup_event():
    global MODEL, ENC, DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading Axiom V2 on {DEVICE}...")
    ENC = tiktoken.get_encoding("cl100k_base")
    
    # Path to your downloaded model
    checkpoint_path = "best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: Model not found at {checkpoint_path}!")
        return
        
    MODEL, val_loss = load_model(checkpoint_path, DEVICE)
    print(f"✅ Model loaded successfully! (Val Loss: {val_loss})")

from typing import Optional

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
    
    # If the user is specifically asking for code or functions, guide the formatting explicitly
    prompt_lower = request.prompt.lower()
    if any(k in prompt_lower for k in ["code", "python", "function", "script", "def ", "program"]):
        if "code" not in base_sys.lower():
            base_sys += " When writing code, provide clean, executable Python code in markdown blocks (```python ... ```)."

    formatted_prompt = f"### System:\n{base_sys}\n\n### User:\n{request.prompt}\n\n### Assistant:\n"
    
    rep_penalty = request.repetition_penalty if request.repetition_penalty is not None else 1.05
    
    generated_tokens, speed = generate(
        MODEL, 
        ENC, 
        formatted_prompt, 
        max_new_tokens=request.max_tokens, 
        temperature=request.temperature, 
        repetition_penalty=rep_penalty,
        stream=False
    )
    
    response_text = ENC.decode(generated_tokens)
    response_text = response_text.replace("<|endoftext|>", "").strip()
    
    return {"response": response_text, "speed": f"{speed:.2f}"}

@app.get("/")
async def serve_ui():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

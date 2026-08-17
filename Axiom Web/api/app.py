import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import tiktoken
import uvicorn
import asyncio
import json

import sys
# Path is now ../../Axiom Model because app.py is inside api/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Axiom Model'))

from axiom_model.core.model import AxiomV2
from generate import load_model, sample_top_p

app = FastAPI(title="Axiom V2 Local API")

# Add CORS Middleware so React (localhost:5173) can talk to FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
enc = tiktoken.get_encoding("cl100k_base")

print(f"Initializing Axiom V2 on {device}...")
model, val_loss = load_model("../../Axiom Model/best.pt", device)
print("✅ Brain Loaded into Memory!")

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 512

def format_prompt(message: str) -> str:
    return f"### System:\nYou are a highly intelligent, logical, and helpful AI assistant named Axiom.\n\n### User:\n{message}\n\n### Assistant:\n"

async def generate_stream(prompt: str, max_new_tokens: int, temperature: float):
    tokens = enc.encode(prompt, allowed_special="all")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    kv_cache = None
    start_pos = 0
    stop_token = 100257 
    
    generated_text = ""
    with torch.inference_mode():
        for i in range(max_new_tokens):
            if i == 0:
                input_ids = tokens
            else:
                input_ids = tokens[:, -1:]
                start_pos = tokens.shape[1] - 1
                
            try:
                logits, kv_cache = model(input_ids, start_pos=start_pos, kv_cache=kv_cache, return_cache=True)
            except ValueError:
                break 
                
            next_token_logits = logits[:, -1, :]
            
            # Repetition Penalty
            repetition_penalty = 1.1
            for t in tokens[0]:
                if next_token_logits[0, t] > 0:
                    next_token_logits[0, t] /= repetition_penalty
                else:
                    next_token_logits[0, t] *= repetition_penalty
            
            if temperature == 0.0:
                idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                
                # Top-K Sampling
                top_k = 50
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                idx_next = sample_top_p(probs, 0.9)
                
            token_id = idx_next.item()
            if token_id == stop_token:
                break 
                
            tokens = torch.cat((tokens, idx_next), dim=1)
            token_str = enc.decode([token_id])
            generated_text += token_str
            
            yield f"data: {json.dumps({'token': token_str})}\n\n"
            await asyncio.sleep(0.001) 
            
            if "<|endoftext|>" in generated_text:
                break

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "device": device}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    prompt = format_prompt(req.message)
    return StreamingResponse(generate_stream(prompt, req.max_tokens, req.temperature), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)

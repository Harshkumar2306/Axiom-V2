import os
import sys

# Forward directly to the master production backend in Axiom Model
MASTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Axiom Model'))
if MASTER_DIR not in sys.path:
    sys.path.insert(0, MASTER_DIR)

# Change directory context so checkpoints and knowledge_base resolve correctly
os.chdir(MASTER_DIR)

from app import app, MODEL, rag_engine, web_engine

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")

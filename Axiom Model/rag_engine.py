import os
import io
import json
import re
import torch
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
import pypdf

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(KNOWLEDGE_DIR, "index.pt")
METADATA_PATH = os.path.join(KNOWLEDGE_DIR, "metadata.json")

class RAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Initializing RAG Embedding Engine on {self.device}...")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.chunks: List[Dict[str, Any]] = [] # [{ "text": str, "source": str, "chunk_id": int }]
        self.embeddings: Optional[torch.Tensor] = None # Tensor [N, 384]
        self.load_index()

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            try:
                self.embeddings = torch.load(INDEX_PATH, map_location=self.device, weights_only=True)
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                print(f"📚 Loaded {len(self.chunks)} document chunks from disk cache.")
            except Exception as e:
                print(f"⚠️ Failed to load RAG cache: {e}")
                self.chunks = []
                self.embeddings = None

    def save_index(self):
        try:
            if self.embeddings is not None and len(self.chunks) > 0:
                torch.save(self.embeddings, INDEX_PATH)
                with open(METADATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.chunks, f, indent=2)
                print(f"💾 Saved {len(self.chunks)} chunks to knowledge base.")
            else:
                if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
                if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
        except Exception as e:
            print(f"⚠️ Failed to save RAG cache: {e}")

    def chunk_text(self, text: str, source: str, chunk_size: int = 500, overlap: int = 80) -> List[Dict[str, Any]]:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if not cleaned:
            return []
            
        chunks = []
        start = 0
        chunk_id = 0
        text_len = len(cleaned)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Try to break at nearest space or punctuation if not at the end
            if end < text_len:
                last_space = cleaned.rfind(' ', start, end)
                if last_space > start + (chunk_size // 2):
                    end = last_space
            
            chunk_slice = cleaned[start:end].strip()
            if len(chunk_slice) > 20:
                chunks.append({
                    "text": chunk_slice,
                    "source": source,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
            
            start = end - overlap if end < text_len else text_len

        return chunks

    def ingest_file(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        extracted_text = ""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages_text = []
                for i, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        pages_text.append(f"[Page {i+1}] {t}")
                extracted_text = "\n".join(pages_text)
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF '{filename}': {e}")
        else:
            # Plaintext, Markdown, Python, Code, JSON, CSV
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1", errors="ignore")

        if not extracted_text.strip():
            raise ValueError(f"No extractable text found in '{filename}'")

        # Remove existing chunks for this file if re-uploading
        self.delete_document(filename, auto_save=False)

        new_chunks = self.chunk_text(extracted_text, source=filename)
        if not new_chunks:
            raise ValueError("Document was too short to create meaningful chunks.")

        texts = [c["text"] for c in new_chunks]
        new_embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False).to(self.device)
        # Normalize for cosine similarity via dot product
        new_embeddings = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)

        if self.embeddings is None or len(self.chunks) == 0:
            self.embeddings = new_embeddings
            self.chunks = new_chunks
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
            self.chunks.extend(new_chunks)

        self.save_index()
        return {
            "filename": filename,
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
            "total_documents": len(self.list_documents())
        }

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.25) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_emb = self.encoder.encode([query], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)

        # Dot product of normalized vectors = Cosine Similarity
        scores = torch.mm(query_emb, self.embeddings.T).squeeze(0) # [N]
        top_k_val = min(top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(scores, k=top_k_val)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            if score >= min_similarity:
                chunk = dict(self.chunks[idx])
                chunk["score"] = round(float(score), 4)
                results.append(chunk)

        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        docs: Dict[str, int] = {}
        for c in self.chunks:
            src = c.get("source", "Unknown")
            docs[src] = docs.get(src, 0) + 1
        return [{"filename": name, "chunks": count} for name, count in docs.items()]

    def delete_document(self, filename: str, auto_save: bool = True) -> int:
        if not self.chunks:
            return 0

        keep_indices = [i for i, c in enumerate(self.chunks) if c.get("source") != filename]
        removed_count = len(self.chunks) - len(keep_indices)

        if removed_count > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.chunks and self.embeddings is not None:
                self.embeddings = self.embeddings[keep_indices]
            else:
                self.embeddings = None
            if auto_save:
                self.save_index()

        return removed_count

    def clear(self):
        self.chunks = []
        self.embeddings = None
        self.save_index()


class WebSearchEngine:
    def __init__(self, rag_encoder: Optional[SentenceTransformer] = None):
        self.rag_encoder = rag_encoder
        self.client = httpx.Client(
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            },
            follow_redirects=True,
            timeout=4.0
        )

    def scrape_url_paragraphs(self, url: str) -> List[str]:
        try:
            resp = self.client.get(url)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            # Strip noise
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
                tag.decompose()
            
            paragraphs = []
            for p in soup.find_all(["p", "h1", "h2", "h3", "li"]):
                txt = re.sub(r'\s+', ' ', p.get_text()).strip()
                if len(txt) > 40 and not any(skip in txt.lower() for skip in ["cookie", "privacy policy", "all rights reserved", "subscribe"]):
                    paragraphs.append(txt)
            return paragraphs[:25] # top 25 candidate paragraphs
        except Exception:
            return []

    def search_wikipedia_fallback(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        try:
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={httpx.URL(query).raw_path.decode()}&utf8=&format=json"
            resp = self.client.get(search_url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("query", {}).get("search", [])
                results = []
                for item in items[:max_results]:
                    title = item.get("title", "")
                    clean_snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text()
                    results.append({
                        "title": f"Wikipedia: {title}",
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "snippet": clean_snippet,
                        "body": clean_snippet
                    })
                return results
        except Exception:
            pass
        return []

    def search_and_deep_scrape(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        raw_results = []
        try:
            ddgs_res = DDGS().text(query, max_results=max_results)
            if ddgs_res:
                for r in ddgs_res:
                    raw_results.append({
                        "title": r.get("title", "Web Page"),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "body": r.get("body", "")
                    })
        except Exception as e:
            print(f"⚠️ DDGS primary search notice: {e}")

        # If primary search produced nothing, invoke Wikipedia fallback
        if not raw_results:
            raw_results = self.search_wikipedia_fallback(query, max_results=max_results)

        if not raw_results:
            return {"context": "", "sources": []}

        sources = []
        detailed_context_blocks = []

        for i, item in enumerate(raw_results[:max_results]):
            title = item["title"]
            url = item["url"]
            snippet = item["snippet"]
            
            # Record source metadata for frontend citations
            sources.append({
                "id": i + 1,
                "title": title,
                "url": url,
                "snippet": snippet[:180] + ("..." if len(snippet) > 180 else "")
            })

            # Deep scrape top 2 web pages if valid URL
            page_text = ""
            if url and url.startswith("http") and i < 2:
                paragraphs = self.scrape_url_paragraphs(url)
                if paragraphs:
                    # Select top most informative paragraphs
                    page_text = " ".join(paragraphs[:3])

            block_content = page_text if page_text and len(page_text) > len(snippet) else snippet
            clean_slice = re.sub(r'\s+', ' ', block_content).strip()[:350]
            if clean_slice:
                detailed_context_blocks.append(f"- {clean_slice}")

        combined_context = "\n".join(detailed_context_blocks)
        return {
            "context": combined_context,
            "sources": sources
        }

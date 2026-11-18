import os
import io
import re
import json
import math
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import torch
import httpx
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
import pypdf

# Check FAISS availability (with seamless PyTorch fallback)
try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    np = None

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(KNOWLEDGE_DIR, "index.pt")
FAISS_PATH = os.path.join(KNOWLEDGE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(KNOWLEDGE_DIR, "metadata.json")


# =====================================================================
# 1. HIERARCHICAL RECURSIVE TEXT SPLITTER
# =====================================================================
class RecursiveTextSplitter:
    """Token-aware recursive text splitter that respects markdown headers,

    code blocks, paragraphs, and sentence boundaries.
    """
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = [
            "\n## ", "\n### ", "\n#### ",
            "\n\n",
            "```\n",
            "\n",
            ". ", "! ", "? ",
            "; ", ", ", " "
        ]

    def split_text(self, text: str, source: str = "document") -> List[Dict[str, Any]]:
        cleaned = re.sub(r'\r\n', '\n', text).strip()
        if not cleaned:
            return []

        raw_chunks = self._recursive_split(cleaned, self.separators)
        final_chunks: List[Dict[str, Any]] = []

        current_pos = 0
        chunk_id = 0
        for chunk in raw_chunks:
            chunk_str = chunk.strip()
            if len(chunk_str) >= 25:
                # Detect active section heading if present
                heading_match = re.search(r'^(#{1,4}\s+[^\n]+)', chunk_str)
                heading = heading_match.group(1) if heading_match else ""

                final_chunks.append({
                    "text": chunk_str,
                    "source": source,
                    "chunk_id": chunk_id,
                    "heading": heading,
                    "char_len": len(chunk_str),
                    "word_count": len(chunk_str.split())
                })
                chunk_id += 1
            current_pos += len(chunk)

        return final_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep in text:
            splits = text.split(sep)
            chunks: List[str] = []
            current_chunk = ""

            for i, part in enumerate(splits):
                segment = (sep + part) if (i > 0 and sep.startswith("\n#")) else (part if i == 0 else sep + part)
                if len(current_chunk) + len(segment) <= self.chunk_size:
                    current_chunk += segment
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(segment) > self.chunk_size:
                        # Recursively split large segment with smaller separators
                        sub_chunks = self._recursive_split(segment, remaining_seps)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = segment

            if current_chunk:
                chunks.append(current_chunk)

            # Apply overlap across contiguous chunks
            return self._merge_with_overlap(chunks)

        return self._recursive_split(text, remaining_seps)

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            overlap_prefix = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            # Avoid duplicating exact overlap strings
            if not curr.startswith(overlap_prefix):
                overlapped.append(overlap_prefix + " ... " + curr)
            else:
                overlapped.append(curr)
        return overlapped


# =====================================================================
# 2. BM25 SPARSE LEXICAL SEARCH ENGINE
# =====================================================================
class BM25Index:
    """Pure-Python, zero-dependency BM25Okapi implementation for exact

    keyword matching, acronyms, code identifiers, and numbers.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text.lower())

    def fit(self, documents: List[str]):
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            self.doc_lens = []
            self.doc_freqs = []
            self.idf = {}
            return

        total_len = 0
        self.doc_lens = []
        self.doc_freqs = []
        df: Dict[str, int] = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)
            self.doc_freqs.append(token_counts)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            for word in token_counts.keys():
                df[word] += 1

        self.avg_doc_len = total_len / self.corpus_size

        # Compute IDF with standard Lucene smoothing
        self.idf = {}
        for word, freq in df.items():
            idf_val = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = max(0.1, idf_val)

    def get_scores(self, query: str) -> List[float]:
        if self.corpus_size == 0:
            return []

        q_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in q_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx, doc_counts in enumerate(self.doc_freqs):
                tf = doc_counts.get(token, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / (self.avg_doc_len or 1.0)))
                    scores[idx] += idf_val * (tf * (self.k1 + 1.0)) / denom

        return scores


# =====================================================================
# 3. FAISS & PYTORCH HYBRID VECTOR STORE
# =====================================================================
class FAISSVectorStore:
    """High-speed vector index with native FAISS (IndexFlatIP) and automatic

    PyTorch fallback for sub-millisecond similarity queries.
    """
    def __init__(self, dimension: int = 384, device: str = "cpu"):
        self.dimension = dimension
        self.device = device
        self.faiss_index = None
        self.torch_embeddings: Optional[torch.Tensor] = None
        self.use_faiss = HAS_FAISS

        if self.use_faiss:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)

    def add(self, embeddings_tensor: torch.Tensor):
        # Ensure L2 normalization for Inner Product to strictly equal Cosine Similarity
        norm_embs = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)

        if self.use_faiss:
            np_data = norm_embs.detach().cpu().numpy().astype("float32")
            self.faiss_index.add(np_data)

        # Always maintain torch tensor as backup / persistence bridge
        if self.torch_embeddings is None:
            self.torch_embeddings = norm_embs.detach().cpu()
        else:
            self.torch_embeddings = torch.cat([self.torch_embeddings, norm_embs.detach().cpu()], dim=0)

    def search(self, query_tensor: torch.Tensor, top_k: int) -> Tuple[List[float], List[int]]:
        if self.count == 0:
            return [], []

        k = min(top_k, self.count)
        norm_q = torch.nn.functional.normalize(query_tensor, p=2, dim=1)

        if self.use_faiss and self.faiss_index is not None:
            q_np = norm_q.detach().cpu().numpy().astype("float32")
            scores, indices = self.faiss_index.search(q_np, k)
            return scores[0].tolist(), indices[0].tolist()
        else:
            # High-performance PyTorch Fallback
            embs = self.torch_embeddings.to(query_tensor.device)
            sims = torch.mm(norm_q, embs.T).squeeze(0)
            top_scores, top_idx = torch.topk(sims, k=k)
            return top_scores.tolist(), top_idx.tolist()

    @property
    def count(self) -> int:
        if self.use_faiss and self.faiss_index is not None:
            return self.faiss_index.ntotal
        return len(self.torch_embeddings) if self.torch_embeddings is not None else 0

    def reset(self):
        if self.use_faiss:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.torch_embeddings = None

    def save(self, faiss_path: str, torch_path: str):
        if self.count == 0:
            if os.path.exists(faiss_path): os.remove(faiss_path)
            if os.path.exists(torch_path): os.remove(torch_path)
            return

        if self.use_faiss and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, faiss_path)

        if self.torch_embeddings is not None:
            torch.save(self.torch_embeddings, torch_path)

    def load(self, faiss_path: str, torch_path: str) -> bool:
        loaded = False
        if self.use_faiss and os.path.exists(faiss_path):
            try:
                self.faiss_index = faiss.read_index(faiss_path)
                loaded = True
            except Exception as e:
                print(f"⚠️ FAISS load notice: {e}, falling back to PyTorch weights")

        if os.path.exists(torch_path):
            try:
                self.torch_embeddings = torch.load(torch_path, map_location="cpu", weights_only=True)
                if not loaded and self.use_faiss:
                    self.faiss_index = faiss.IndexFlatIP(self.dimension)
                    np_data = self.torch_embeddings.numpy().astype("float32")
                    self.faiss_index.add(np_data)
                loaded = True
            except Exception as e:
                print(f"⚠️ Torch embeddings load notice: {e}")

        return loaded


# =====================================================================
# 4. NEXT-GEN RAG ENGINE (HYBRID FAISS + BM25 + RRF)
# =====================================================================
class RAGEngine:
    """Production-grade RAG engine combining FAISS dense vector search with

    BM25 lexical retrieval and Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        engine_mode = "FAISS (Native Accelerated)" if HAS_FAISS else "PyTorch Dense (Optimized Fallback)"
        print(f"⚡ Initializing RAG Engine ({engine_mode}) on {self.device}...")

        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.splitter = RecursiveTextSplitter(chunk_size=750, chunk_overlap=100)
        self.vector_store = FAISSVectorStore(dimension=384, device=self.device)
        self.bm25 = BM25Index()

        self.chunks: List[Dict[str, Any]] = []
        self.load_index()

    def load_index(self):
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                loaded = self.vector_store.load(FAISS_PATH, INDEX_PATH)
                if loaded and self.chunks:
                    # Fit BM25 on loaded chunk texts
                    texts = [c["text"] for c in self.chunks]
                    self.bm25.fit(texts)
                    print(f"📚 Loaded {len(self.chunks)} chunks into FAISS & BM25 indices.")
                else:
                    self.clear()
            except Exception as e:
                print(f"⚠️ Failed to load RAG cache: {e}")
                self.clear()

    def save_index(self):
        try:
            if len(self.chunks) > 0 and self.vector_store.count > 0:
                self.vector_store.save(FAISS_PATH, INDEX_PATH)
                with open(METADATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.chunks, f, indent=2)
                print(f"💾 Saved {len(self.chunks)} chunks to FAISS vector index & metadata store.")
            else:
                self.vector_store.reset()
                if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
                if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
                if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
        except Exception as e:
            print(f"⚠️ Failed to save RAG cache: {e}")

    def ingest_file(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower()
        extracted_text = ""

        if ext == ".pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages = []
                for i, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        pages.append(f"[Page {i+1}]\n{t.strip()}")
                extracted_text = "\n\n".join(pages)
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF '{filename}': {e}")
        else:
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1", errors="ignore")

        if not extracted_text.strip():
            raise ValueError(f"No extractable text found in '{filename}'")

        # Delete any previous version of this document
        self.delete_document(filename, auto_save=False)

        new_chunks = self.splitter.split_text(extracted_text, source=filename)
        if not new_chunks:
            raise ValueError("Document was too short to create meaningful chunks.")

        # Compute dense embeddings
        texts = [c["text"] for c in new_chunks]
        new_embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(self.device)

        self.vector_store.add(new_embeddings)
        self.chunks.extend(new_chunks)

        # Refresh BM25 index across all current chunks
        all_texts = [c["text"] for c in self.chunks]
        self.bm25.fit(all_texts)

        self.save_index()
        return {
            "filename": filename,
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
            "total_documents": len(self.list_documents()),
            "index_backend": "FAISS" if HAS_FAISS else "PyTorch"
        }

    def search(self, query: str, top_k: int = 4, alpha: float = 0.65, min_score: float = 0.01) -> List[Dict[str, Any]]:
        """Hybrid Search: Blends FAISS dense cosine similarity with BM25 sparse

        lexical scores using Reciprocal Rank Fusion (RRF).
        """
        if len(self.chunks) == 0 or self.vector_store.count == 0:
            return []

        # 1. Dense FAISS Search
        query_emb = self.encoder.encode([query], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        dense_scores, dense_indices = self.vector_store.search(query_emb, top_k=min(top_k * 3, len(self.chunks)))

        # 2. Sparse BM25 Search
        bm25_all_scores = self.bm25.get_scores(query)
        bm25_ranked = sorted(enumerate(bm25_all_scores), key=lambda x: x[1], reverse=True)[:min(top_k * 3, len(self.chunks))]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = alpha * (1 / (k + rank_dense)) + (1 - alpha) * (1 / (k + rank_bm25))
        rrf_constant = 60.0
        rrf_scores: Dict[int, float] = {}
        dense_score_map: Dict[int, float] = {}
        bm25_score_map: Dict[int, float] = {}

        for rank, (idx, score) in enumerate(zip(dense_indices, dense_scores)):
            dense_score_map[idx] = float(score)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + alpha * (1.0 / (rrf_constant + rank + 1.0))

        for rank, (idx, score) in enumerate(bm25_ranked):
            bm25_score_map[idx] = float(score)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 - alpha) * (1.0 / (rrf_constant + rank + 1.0))

        # Sort by composite RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, composite_score in sorted_candidates:
            if idx < len(self.chunks):
                chunk = dict(self.chunks[idx])
                chunk["score"] = round(float(composite_score), 5)
                chunk["dense_score"] = round(dense_score_map.get(idx, 0.0), 4)
                chunk["bm25_score"] = round(bm25_score_map.get(idx, 0.0), 4)
                results.append(chunk)

        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        for c in self.chunks:
            src = c.get("source", "Unknown")
            if src not in docs:
                docs[src] = {"filename": src, "chunks": 0, "total_words": 0}
            docs[src]["chunks"] += 1
            docs[src]["total_words"] += c.get("word_count", 0)
        return list(docs.values())

    def delete_document(self, filename: str, auto_save: bool = True) -> int:
        if not self.chunks:
            return 0

        keep_indices = [i for i, c in enumerate(self.chunks) if c.get("source") != filename]
        removed_count = len(self.chunks) - len(keep_indices)

        if removed_count > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.chunks and self.vector_store.torch_embeddings is not None:
                remaining_embs = self.vector_store.torch_embeddings[keep_indices]
                self.vector_store.reset()
                self.vector_store.add(remaining_embs)
                self.bm25.fit([c["text"] for c in self.chunks])
            else:
                self.clear()

            if auto_save:
                self.save_index()

        return removed_count

    def clear(self):
        self.chunks = []
        self.vector_store.reset()
        self.bm25 = BM25Index()
        self.save_index()


# =====================================================================
# 5. ASYNC DEEP WEB SEARCH & SEMANTIC RERANKING ENGINE
# =====================================================================
class WebSearchEngine:
    """High-speed asynchronous Internet Search & Deep Scraping engine with

    multi-engine resilience, parallel page scraping, and semantic reranking.
    """
    def __init__(self, rag_encoder: Optional[SentenceTransformer] = None):
        self.rag_encoder = rag_encoder
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _clean_query(self, query: str) -> str:
        """Strip conversational prefixes to generate high-density search terms."""
        cleaned = re.sub(
            r'^(can you please |please |could you |can you |tell me |search for |find me |look up |what is |who is |where is )',
            '', query, flags=re.IGNORECASE
        ).strip()
        cleaned = re.sub(r'[\?\.!]+$', '', cleaned).strip()
        return cleaned if len(cleaned) >= 3 else query

    async def _async_scrape_url(self, client: httpx.AsyncClient, url: str) -> List[str]:
        """Asynchronously scrape candidate webpage and extract informative paragraphs."""
        try:
            resp = await client.get(url, timeout=3.5)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg"]):
                tag.decompose()

            paragraphs: List[str] = []
            for p in soup.find_all(["p", "h1", "h2", "h3", "li"]):
                txt = re.sub(r'\s+', ' ', p.get_text()).strip()
                if len(txt) > 45 and not any(skip in txt.lower() for skip in ["cookie", "privacy policy", "terms of service", "all rights reserved", "subscribe"]):
                    paragraphs.append(txt)
            return paragraphs[:30]
        except Exception:
            return []

    async def _search_wikipedia(self, client: httpx.AsyncClient, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        """Fast Wikipedia search and REST summary fallback."""
        results = []
        try:
            import urllib.parse
            encoded = urllib.parse.quote_plus(query)
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&utf8=&format=json"
            wiki_headers = {"User-Agent": "AxiomBot/2.0 (https://axiom.ai; search@axiom.ai)"}
            resp = await client.get(search_url, headers=wiki_headers, timeout=3.5)
            if resp.status_code == 200:
                items = resp.json().get("query", {}).get("search", [])[:max_results]
                for item in items:
                    title = item.get("title", "")
                    clean_snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text()
                    results.append({
                        "title": f"Wikipedia: {title}",
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "snippet": clean_snippet,
                        "paragraphs": [clean_snippet]
                    })
        except Exception as e:
            print(f"⚠️ Wikipedia search notice: {e}")
        return results

    def _sync_ddgs(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Worker function executed in worker thread to avoid blocking asyncio event loop."""
        results = []
        try:
            ddgs_res = DDGS().text(query, max_results=max_results)
            if ddgs_res:
                for r in ddgs_res:
                    results.append({
                        "title": r.get("title", "Web Page"),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "paragraphs": [r.get("body", "")]
                    })
        except Exception as e:
            print(f"⚠️ Primary DuckDuckGo search notice: {e}")
        return results

    def _semantic_rerank_passages(self, query: str, candidate_paragraphs: List[str], top_n: int = 3) -> List[str]:
        """Rerank candidate passages using SentenceTransformer to pick the most informative answers."""
        if not candidate_paragraphs or self.rag_encoder is None:
            return candidate_paragraphs[:top_n]

        try:
            device = next(self.rag_encoder.parameters()).device if hasattr(self.rag_encoder, "parameters") else "cpu"
            q_emb = self.rag_encoder.encode([query], convert_to_tensor=True, show_progress_bar=False)
            p_embs = self.rag_encoder.encode(candidate_paragraphs, convert_to_tensor=True, show_progress_bar=False)

            q_norm = torch.nn.functional.normalize(q_emb, p=2, dim=1)
            p_norm = torch.nn.functional.normalize(p_embs, p=2, dim=1)

            scores = torch.mm(q_norm, p_norm.T).squeeze(0)
            top_k = min(top_n, len(candidate_paragraphs))
            _, indices = torch.topk(scores, k=top_k)

            return [candidate_paragraphs[i] for i in indices.tolist()]
        except Exception:
            return candidate_paragraphs[:top_n]

    async def search_and_deep_scrape(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """100% Asynchronous parallel search, multi-page deep scrape, and semantic passage reranking."""
        clean_q = self._clean_query(query)
        raw_items: List[Dict[str, Any]] = []

        # 1. Non-blocking Primary Search via ThreadPool
        raw_items = await asyncio.to_thread(self._sync_ddgs, clean_q, max_results)

        # 2. Resilient Fallbacks
        async with httpx.AsyncClient(headers=self.headers, follow_redirects=True, timeout=4.0) as client:
            if not raw_items:
                raw_items = await self._search_wikipedia(client, clean_q, max_results=max_results)

            if not raw_items:
                return {"context": "", "sources": []}

            # 3. Parallel Asynchronous Web Scraping across Top URLs
            urls_to_scrape = [item["url"] for item in raw_items if item.get("url", "").startswith("http")]
            scrape_tasks = [self._async_scrape_url(client, u) for u in urls_to_scrape[:max_results]]
            scraped_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        sources = []
        all_candidate_passages: List[str] = []

        for i, item in enumerate(raw_items[:max_results]):
            title = item.get("title", "Web Source")
            url = item.get("url", "")
            snippet = item.get("snippet", "")

            sources.append({
                "id": i + 1,
                "type": "web",
                "title": title,
                "url": url,
                "snippet": snippet[:180] + ("..." if len(snippet) > 180 else "")
            })

            # Harvest scraped paragraphs
            if i < len(scraped_results) and isinstance(scraped_results[i], list) and scraped_results[i]:
                all_candidate_passages.extend(scraped_results[i])
            elif snippet:
                all_candidate_passages.append(snippet)

        # 4. Semantic Passage Selection & Reranking
        best_passages = self._semantic_rerank_passages(query, all_candidate_passages, top_n=4)

        formatted_blocks = []
        for p in best_passages:
            p_clean = re.sub(r'\s+', ' ', p).strip()[:400]
            if p_clean:
                formatted_blocks.append(f"- {p_clean}")

        return {
            "context": "\n".join(formatted_blocks),
            "sources": sources
        }

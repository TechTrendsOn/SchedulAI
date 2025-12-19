import os
import uuid
import json
from typing import List, Dict, Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


class KnowledgeAgentRAG:
    """
    Build and query a persistent RAG store over compliance and rostering artifacts.

    - Ingests Excel/CSV/JSON into text documents with metadata.
    - Chunks and embeds docs with SentenceTransformers.
    - Stores vectors in a persistent ChromaDB collection.
    - Answers questions with a Flan‑T5 text2text model using retrieved context.
    """

    def __init__(
        self,
        persistence_dir: str = "./chroma_knowledge",
        collection_name: str = "compliance_knowledge",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ):
        # Storage
        self.persistence_dir = persistence_dir
        self.collection_name = collection_name
        os.makedirs(self.persistence_dir, exist_ok=True)

        # Chunking config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Vector store (persistent)
        self.client = chromadb.PersistentClient(path=self.persistence_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        # LLM (Flan‑T5)
        hf_pipe = pipeline(
            "text2text-generation",
            model=llm_model_name,
            device=-1,   # CPU; change to 0 for GPU if available
            max_length=512,
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)

    
    # Data loading
    # ------------
    def load_excel(self, path: str, sheet: Optional[str] = None) -> List[Dict]:
        if sheet:
            df = pd.read_excel(path, sheet_name=sheet)
            return self._df_to_docs(df, source=path, sheet=sheet)
        else:
            xl = pd.ExcelFile(path)
            docs: List[Dict] = []
            for sh in xl.sheet_names:
                df = xl.parse(sh)
                docs.extend(self._df_to_docs(df, source=path, sheet=sh))
            return docs

    def load_csv(self, path: str) -> List[Dict]:
        df = pd.read_csv(path)
        return self._df_to_docs(df, source=path, sheet=None)

    def load_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def flatten(d, prefix: str = "") -> List[str]:
            items: List[str] = []
            for k, v in d.items():
                new_key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten(v, prefix=new_key + "."))
                else:
                    items.append(f"{new_key}: {v}")
            return items

        text = "\n".join(flatten(data))
        meta = {"source": path, "sheet": "", "row_index": -1, "columns": ""}
        return [{"text": text, "metadata": meta}]

    def _df_to_docs(self, df: pd.DataFrame, source: str, sheet: Optional[str]) -> List[Dict]:
        docs: List[Dict] = []
        df = df.copy()
        df.columns = [str(c) for c in df.columns]

        for i, row in df.iterrows():
            kv_pairs = []
            for col in df.columns:
                val = row.get(col)
                if pd.isna(val):
                    continue
                kv_pairs.append(f"{col}: {val}")
            if not kv_pairs:
                continue

            text = "\n".join(kv_pairs)
            meta = {
                "source": source,
                "sheet": sheet or "",
                "row_index": int(i),
                "columns": ", ".join(df.columns.tolist()),
            }
            docs.append({"text": text, "metadata": meta})

        return docs

    
    # Chunking and embeddings
    # -----------------------
    def _chunk_text(self, text: str) -> List[str]:
        """
        Simple character-based chunking with overlap.
        Keeps it robust for mixed-format text from tables and JSON.
        """
        if not text:
            return []
        tokens = list(text)
        chunks: List[str] = []
        start = 0
        n = len(tokens)

        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = "".join(tokens[start:end])
            chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - self.chunk_overlap)

        return chunks

    def add_documents(self, docs: List[Dict]):
        # Embed and add documents (with chunking) into the Chroma collection.
        ids, texts, metadatas, embeddings = [], [], [], []

        for doc in docs:
            chunks = self._chunk_text(doc["text"])
            for ch in chunks:
                _id = str(uuid.uuid4())
                ids.append(_id)
                texts.append(ch)
                metadatas.append(doc["metadata"])
                vec = self.embedding_model.encode(ch, convert_to_numpy=True)
                embeddings.append(vec)

        if not ids:
            return

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        try:
            self.client.persist()
        except Exception:
            # Persistence failures shouldn't crash the app; collection still works in-memory.
            pass

    
    # Retrieval and answering
    # -----------------------
    def retrieve(self, query: str, k: int = 6) -> Tuple[List[str], List[Dict]]:
        # Retrieve top-k context chunks and their metadata for a query.
        q_emb = self.embedding_model.encode(query, convert_to_numpy=True)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
        )
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        return docs, metas

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        ctx_block = "\n\n".join(f"- {c}" for c in contexts) if contexts else "None."
        prompt = (
            "You are a compliance assistant for Australian restaurant rostering.\n"
            "Use ONLY the provided context to answer the question precisely. "
            "If the answer is not in context, say you don’t have enough information.\n\n"
            f"Context:\n{ctx_block}\n\n"
            f"Question:\n{query}\n\n"
            "Answer succinctly and cite specific fields from the context when possible."
        )
        return prompt

    def answer(self, query: str, k: int = 6) -> Dict:
        """
        RAG-style answer:
        - retrieve top-k chunks
        - build a grounded prompt
        - generate an answer with Flan‑T5
        - return answer + contexts + lightweight source metadata
        """
        contexts, metas = self.retrieve(query, k=k)
        prompt = self._build_prompt(query, contexts)
        out = self.llm.invoke(prompt)

        # HuggingFacePipeline usually returns a list of dicts with "generated_text"
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            answer_text = out[0].get("generated_text", "").strip()
        else:
            answer_text = str(out).strip()

        sources = [
            {
                "source": (m or {}).get("source", ""),
                "sheet": (m or {}).get("sheet", ""),
                "row_index": (m or {}).get("row_index", -1),
            }
            for m in metas
        ]

        return {
            "query": query,
            "answer": answer_text,
            "contexts": contexts,
            "sources": sources,
        }

    
    # Convenience: load and index multiple files
    # ------------------------------------------
    def ingest_files(self, files: List[Dict[str, str]]):
        """
        Convenience method to load and index a list of files.
        Each file dict: {"path": "...", "type": "excel|csv|json", "sheet": optional}.
        """
        all_docs: List[Dict] = []
        for f in files:
            path = f["path"]
            ftype = f.get("type", "csv").lower()
            sheet = f.get("sheet")

            if not os.path.exists(path):
                # Skip missing files silently; they may be optional artifacts.
                continue

            if ftype == "excel":
                docs = self.load_excel(path, sheet=sheet)
            elif ftype == "csv":
                docs = self.load_csv(path)
            elif ftype == "json":
                docs = self.load_json(path)
            else:
                raise ValueError(f"Unsupported file type: {ftype}")

            all_docs.extend(docs)

        self.add_documents(all_docs)

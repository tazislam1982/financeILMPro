# src/services/chromaservice.py

import os
import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.errors import NotFoundError

# Prefer your project's logservice; fall back to stdlib logging if unavailable
try:
    from src.services import logservice  # expects logservice.logging
except Exception:  # pragma: no cover
    import logging as _fallback_logging
    class _LogSvc:  # minimal shim
        logging = _fallback_logging
    logservice = _LogSvc()

# OpenAI client for client-side embeddings
from openai import OpenAI

load_dotenv()


class ChromaService:
    """
    ChromaDB 1.0.15/1.0.16 service

    - Uses **client-side** embeddings for all queries/writes to avoid server default/size mismatches.
    - Never changes a collection's persisted embedding_function (prevents conflicts).
    - All blocking Chroma calls run via asyncio.to_thread (safe for FastAPI).
    - Returns lists of (text, distance, metadata).
    """

    def __init__(self) -> None:
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        # Make sure this matches your server (your logs showed 8007)
        chroma_port = int(os.getenv("CHROMA_PORT", "8007"))
        self.default_k = int(os.getenv("DEFAULT_K", "6"))

        # OpenAI API key (supports either OPENAI_API_KEY or ALIM_API_KEY)
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("ALIM_API_KEY")
        if not openai_key:
            logservice.logging.error("chromaservice.py: Missing OPENAI_API_KEY/ALIM_API_KEY.")
            raise RuntimeError("Missing OPENAI_API_KEY/ALIM_API_KEY.")
        self._openai = OpenAI(api_key=openai_key)

        # IMPORTANT: set this to the SAME model you used when inserting documents
        # text-embedding-3-small and ada-002 are both 1536-d; MiniLM/SBERT are often 384-d.
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        # HTTP client with token auth (adjust envs if needed)
        self.chroma_client = HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=os.getenv("CHROMA_AUTH_TOKEN"),
                chroma_auth_token_transport_header=os.getenv("CHROMA_AUTH_TOKEN_HEADER", "X-Chroma-Token"),
            ),
        )

    # ---------- collection helpers ----------

    @lru_cache(maxsize=16)
    def _cached_collection(self, collection_name: str) -> Any:
        """
        Get existing collection; if missing, create one **without** server-side embedding_function.
        We keep everything client-side for consistency (no dimension surprises).
        """
        try:
            return self.chroma_client.get_collection(name=collection_name)
        except NotFoundError:
            return self.chroma_client.create_collection(name=collection_name)

    def get_chroma_collection(self, collection_name: str) -> Any:
        return self._cached_collection(collection_name)

    # ---------- embedding helpers ----------

    def _embed_one(self, text: str) -> List[float]:
        emb = self._openai.embeddings.create(model=self.embedding_model, input=text)
        return emb.data[0].embedding

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        # Batch in one request when possible; OpenAI supports list inputs
        emb = self._openai.embeddings.create(model=self.embedding_model, input=texts)
        # Preserve original order
        return [d.embedding for d in emb.data]

    # ---------- write helpers (optional, but recommended for consistency) ----------

    async def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 64,
    ) -> None:
        """
        Upsert texts with **client-side embeddings** so stored vectors match query vectors.
        """
        if not texts:
            return
        if ids is not None and len(ids) != len(texts):
            raise ValueError("len(ids) must equal len(texts)")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("len(metadatas) must equal len(texts)")

        col = self.get_chroma_collection(collection_name)

        # Chunk to avoid very large payloads
        start = 0
        while start < len(texts):
            end = min(start + batch_size, len(texts))
            chunk_texts = texts[start:end]
            chunk_ids = ids[start:end] if ids else None
            chunk_mds = metadatas[start:end] if metadatas else None

            # Embed client-side
            embeds = await asyncio.to_thread(self._embed_many, chunk_texts)

            # Add/Upsert (choose one; here: add if new, upsert if ids exist)
            # If you want strict upsert behavior, use upsert instead.
            def _do_upsert() -> None:
                col.upsert(
                    embeddings=embeds,
                    documents=chunk_texts,
                    ids=chunk_ids,
                    metadatas=chunk_mds,
                )

            await asyncio.to_thread(_do_upsert)
            start = end

    # ---------- read/search helpers ----------

    async def similarity_search_optimized(
        self,
        query: str,
        collection_name: str,
        index_key: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Metadata-filtered search using client-side embeddings to guarantee dimension match.
        Returns: [(text, distance, metadata), ...]
        """
        k = k or self.default_k
        col = self.get_chroma_collection(collection_name)

        q_emb = await asyncio.to_thread(self._embed_one, query)
        where = {"source_file": index_key} if index_key else None

        raw = await asyncio.to_thread(
            col.query,
            query_embeddings=[q_emb],
            n_results=k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )

        docs = raw.get("documents", [[]])[0] if raw.get("documents") else []
        dists = raw.get("distances", [[]])[0] if raw.get("distances") else []
        metas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else [{} for _ in docs]
        return list(zip(docs, dists, metas))

    async def search(
        self,
        collection: str,
        query_text: str,
        k: int = 6,
        metadata_filter: Optional[Dict[str, Any]] = None,
        document_filter: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Full search with metadata + document filters.
        Uses client-side embeddings to avoid any server-default dimension issues.
        """
        include = include or ["documents", "distances", "metadatas"]
        col = self.get_chroma_collection(collection)

        q_emb = await asyncio.to_thread(self._embed_one, query_text)
        raw = await asyncio.to_thread(
            col.query,
            query_embeddings=[q_emb],
            n_results=k,
            where=metadata_filter,
            where_document=document_filter,
            include=include,
        )

        docs = raw.get("documents", [[]])[0] if raw.get("documents") else []
        dists = raw.get("distances", [[]])[0] if raw.get("distances") else []
        metas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else [{} for _ in docs]
        return list(zip(docs, dists, metas))

    # ---------- higher-level helpers (match your previous usage) ----------

    async def process_source_results(self, question: str, source_type: str, params: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        """
        Runs a simple search in a named collection, returns (texts, scores).
        """
        try:
            results = await self.similarity_search_optimized(
                query=question,
                collection_name=params.get("collection", "financeilm"),
                index_key=params.get("index_key"),
                k=params.get("k", 5),
            )
            texts = [text for (text, _dist, _meta) in results]
            scores = [dist for (_text, dist, _meta) in results]

            if params.get("suffix"):
                suffix = params["suffix"]
                texts = [f"{t}\n{suffix}" for t in texts]

            return texts, scores
        except Exception as e:
            logservice.logging.error(f"chromaservice.process_source_results: Error processing {source_type}: {e}")
            return [], []

    async def get_context_info_optimized(self, question: str, source: str) -> Tuple[List[str], Dict[str, Any], List[float]]:
        """
        Default: search 'financeilm' collection and return (texts, link_extracted, scores).
        """
        results = await self.similarity_search_optimized(
            question,
            collection_name="financeilm",
            index_key=None,
            k=8,
        )
        text_l = [t for (t, _dist, _meta) in results]
        scores = [dist for (_t, dist, _m) in results]
        link_extracted: Dict[str, Any] = {}
        return text_l, link_extracted, scores

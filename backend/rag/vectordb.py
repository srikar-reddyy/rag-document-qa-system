"""
Local vector store for document chunks.

This replaces the ChromaDB dependency with a small JSON-backed store so the
backend can run on Windows without a C++ build toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging
import math
import threading
import uuid

logger = logging.getLogger(__name__)

STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"
COLLECTION_NAME = "documents"
_STORE_LOCK = threading.Lock()
_collection = None


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0

    limit = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0

    for index in range(limit):
        left_value = float(left[index])
        right_value = float(right[index])
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    return dot / math.sqrt(left_norm * right_norm)


def _load_store() -> List[Dict[str, Any]]:
    if not STORE_PATH.exists():
        return []

    try:
        with STORE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to load vector store: {exc}")
        return []

    if not isinstance(data, list):
        return []

    records: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        record_id = str(item.get("id") or "").strip()
        document = str(item.get("document") or "")
        embedding = item.get("embedding") or []
        metadata = item.get("metadata") or {}
        if not record_id:
            continue
        if not isinstance(embedding, list):
            continue
        if not isinstance(metadata, dict):
            metadata = {}
        records.append({
            "id": record_id,
            "document": document,
            "embedding": [float(value) for value in embedding],
            "metadata": metadata,
        })

    return records


def _save_store(records: List[Dict[str, Any]]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STORE_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, default=str)
    tmp_path.replace(STORE_PATH)


def recreate_chroma_storage():
    global _collection

    logger.warning(f"Recreating local vector store at {STORE_PATH}")
    _collection = None
    with _STORE_LOCK:
        if STORE_PATH.exists():
            STORE_PATH.unlink()


@dataclass
class LocalCollection:
    name: str

    def _records(self) -> List[Dict[str, Any]]:
        with _STORE_LOCK:
            return _load_store()

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        with _STORE_LOCK:
            _save_store(records)

    def count(self) -> int:
        return len(self._records())

    def _matches_where(self, metadata: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
        if not where:
            return True

        if "$and" in where:
            clauses = where.get("$and") or []
            return all(self._matches_where(metadata, clause) for clause in clauses)

        for key, expected in where.items():
            value = metadata.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                options = expected.get("$in") or []
                if value not in options and str(value) not in {str(option) for option in options}:
                    return False
                continue

            if value == expected:
                continue
            if str(value) == str(expected):
                continue
            return False

        return True

    def get(
        self,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        records = self._records()
        filtered = [record for record in records if self._matches_where(record.get("metadata", {}), where)]

        if limit is not None:
            filtered = filtered[: int(limit)]

        return {
            "ids": [record["id"] for record in filtered],
            "documents": [record["document"] for record in filtered],
            "metadatas": [record["metadata"] for record in filtered],
            "embeddings": [record["embedding"] for record in filtered],
        }

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        records = self._records()
        index_by_id = {record["id"]: position for position, record in enumerate(records)}

        for record_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            record = {
                "id": str(record_id),
                "document": document,
                "embedding": [float(value) for value in embedding],
                "metadata": dict(metadata or {}),
            }
            position = index_by_id.get(record["id"])
            if position is None:
                index_by_id[record["id"]] = len(records)
                records.append(record)
            else:
                records[position] = record

        self._write_records(records)

    def delete(self, ids: Optional[List[str]] = None) -> None:
        if not ids:
            return

        id_set = {str(item) for item in ids}
        records = [record for record in self._records() if record["id"] not in id_set]
        self._write_records(records)

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        records = self._records()
        if not records or not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embedding = query_embeddings[0]
        scored_records: List[tuple[float, Dict[str, Any]]] = []
        for record in records:
            if not self._matches_where(record.get("metadata", {}), where):
                continue
            similarity = _cosine_similarity(query_embedding, record.get("embedding", []))
            scored_records.append((similarity, record))

        scored_records.sort(key=lambda item: item[0], reverse=True)
        top_records = scored_records[: int(n_results)]

        return {
            "ids": [[record["id"] for _, record in top_records]],
            "documents": [[record["document"] for _, record in top_records]],
            "metadatas": [[record["metadata"] for _, record in top_records]],
            "distances": [[max(0.0, 1.0 - similarity) for similarity, _ in top_records]],
        }


class _LocalClient:
    def __init__(self):
        self.name = COLLECTION_NAME

    def get_collection(self, name: str):
        if name != COLLECTION_NAME:
            raise KeyError(name)
        return get_collection()

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        if name != COLLECTION_NAME:
            raise KeyError(name)
        return get_collection()

    def delete_collection(self, name: str):
        if name != COLLECTION_NAME:
            raise KeyError(name)
        recreate_chroma_storage()


def get_chroma_client() -> _LocalClient:
    return _LocalClient()


def get_collection() -> LocalCollection:
    global _collection
    if _collection is None:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _collection = LocalCollection(name=COLLECTION_NAME)
    return _collection


def add_documents(
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    document_id: Optional[str] = None,
) -> str:
    collection = get_collection()

    if document_id is None:
        document_id = str(uuid.uuid4())

    ids = [f"{document_id}_{index}" for index in range(len(texts))]

    sanitized_metadatas = []
    for metadata in metadatas:
        metadata = dict(metadata or {})
        metadata["document_id"] = document_id

        keys_to_remove = []
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del metadata[key]
        sanitized_metadatas.append(metadata)

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=sanitized_metadatas)
    logger.info(f"Added {len(texts)} chunks to local store with document_id: {document_id}")
    return document_id


def query_documents(
    query_embedding: List[float],
    top_k: int = 5,
    document_ids: Optional[List[str]] = None,
) -> Dict:
    collection = get_collection()

    query_params = {"query_embeddings": [query_embedding], "n_results": top_k}
    if document_ids:
        query_params["where"] = {"document_id": {"$in": document_ids}}
        logger.info(f"Filtering query by document_ids: {document_ids}")
    else:
        logger.warning("No document_ids filter - querying entire local store")

    collection_count = collection.count()
    logger.info(f"Collection has {collection_count} total chunks")

    results = collection.query(**query_params)
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    logger.info(f"Retrieved {len(documents)} chunks from local store")
    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
    }


def get_all_document_metadata() -> List[Dict]:
    collection = get_collection()
    results = collection.get(include=["metadatas"])

    summaries = {}
    for metadata in results.get("metadatas", []) or []:
        if not metadata:
            continue

        document_id = metadata.get("document_id")
        if not document_id:
            continue

        if document_id not in summaries:
            summaries[document_id] = {
                "document_id": document_id,
                "file_name": metadata.get("file_name", "Unknown"),
                "document_name": metadata.get("document_name", metadata.get("file_name", "Unknown")),
                "pdf_title": metadata.get("pdf_title", metadata.get("file_name", "Unknown")),
                "pdf_author": metadata.get("pdf_author", "Unknown"),
                "total_pages": metadata.get("total_pages", 1),
            }

    return list(summaries.values())


def get_document_metadata(document_ids: List[str]) -> List[Dict]:
    if not document_ids:
        return []

    collection = get_collection()
    results = collection.get(
        where={"document_id": {"$in": document_ids}},
        include=["metadatas"],
    )

    summaries = {}
    for metadata in results.get("metadatas", []) or []:
        if not metadata:
            continue

        document_id = metadata.get("document_id")
        if not document_id:
            continue

        existing = summaries.get(document_id)
        page_number = _to_int(metadata.get("page", metadata.get("page_number")), 1)

        if existing is None or page_number < _to_int(existing.get("page_number", 10**9), 10**9):
            summaries[document_id] = {
                "document_id": document_id,
                "file_name": metadata.get("file_name", "Unknown"),
                "document_name": metadata.get("document_name", metadata.get("file_name", "Unknown")),
                "pdf_title": metadata.get("pdf_title", metadata.get("file_name", "Unknown")),
                "pdf_author": metadata.get("pdf_author", "Unknown"),
                "total_pages": metadata.get("total_pages", 1),
                "page_number": page_number,
            }

    return [summaries[doc_id] for doc_id in document_ids if doc_id in summaries]


def get_document_chunks(document_ids: List[str], max_chunks_per_document: int = 40) -> List[Dict]:
    if not document_ids:
        return []

    collection = get_collection()
    results = collection.get(
        where={"document_id": {"$in": document_ids}},
        include=["documents", "metadatas"],
    )

    grouped_chunks = {document_id: [] for document_id in document_ids}

    for document, metadata in zip(results.get("documents", []) or [], results.get("metadatas", []) or []):
        if not metadata:
            continue

        document_id = metadata.get("document_id")
        if document_id not in grouped_chunks:
            continue

        grouped_chunks[document_id].append({
            "text": document,
            "metadata": metadata,
            "relevance_score": 1.0,
        })

    ordered_chunks = []
    for document_id in document_ids:
        chunks = grouped_chunks.get(document_id, [])
        chunks.sort(
            key=lambda chunk: (
                int(chunk["metadata"].get("page", chunk["metadata"].get("page_number", 0)) or 0),
                int(chunk["metadata"].get("chunk_index", 0) or 0),
            )
        )

        if len(chunks) <= max_chunks_per_document:
            selected_chunks = chunks
        else:
            step = len(chunks) / max_chunks_per_document
            selected_indices = sorted({min(int(index * step), len(chunks) - 1) for index in range(max_chunks_per_document)})
            selected_chunks = [chunks[index] for index in selected_indices]

        ordered_chunks.extend(selected_chunks)

    return ordered_chunks


def get_chunks_by_page(document_ids: List[str], page_number: int, limit: Optional[int] = None) -> List[Dict]:
    if not document_ids:
        return []

    collection = get_collection()
    page_number = int(page_number)

    kwargs = {
        "where": {
            "$and": [
                {"document_id": {"$in": document_ids}},
                {"page": page_number},
            ]
        },
        "include": ["documents", "metadatas"],
    }
    if limit is not None:
        kwargs["limit"] = int(limit)

    results = collection.get(**kwargs)
    documents = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []

    if not documents:
        fallback_kwargs = {
            "where": {
                "$and": [
                    {"document_id": {"$in": document_ids}},
                    {"page_number": str(page_number)},
                ]
            },
            "include": ["documents", "metadatas"],
        }
        if limit is not None:
            fallback_kwargs["limit"] = int(limit)

        results = collection.get(**fallback_kwargs)
        documents = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []

    chunks = []
    for document, metadata in zip(documents, metadatas):
        if not metadata:
            continue
        chunks.append({
            "text": document,
            "metadata": metadata,
            "relevance_score": 1.0,
        })

    chunks.sort(
        key=lambda chunk: (
            int(chunk["metadata"].get("page", chunk["metadata"].get("page_number", 0)) or 0),
            int(chunk["metadata"].get("chunk_index", 0) or 0),
        )
    )

    return chunks


def delete_by_document_id(document_id: str):
    collection = get_collection()
    results = collection.get(where={"document_id": document_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info(f"Deleted document {document_id} with {len(results['ids'])} chunks")
    else:
        logger.warning(f"No chunks found for document_id: {document_id}")


def get_collection_stats() -> Dict:
    collection = get_collection()
    count = collection.count()
    return {
        "collection_name": collection.name,
        "total_documents": count,
        "persist_directory": str(STORE_PATH.parent),
    }


def reset_collection():
    global _collection
    _collection = None
    recreate_chroma_storage()
    get_collection()
    logger.info("Collection reset complete")

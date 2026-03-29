"""
Vector Database using ChromaDB
Handles document storage and similarity search
"""

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from typing import List, Dict, Optional
import uuid
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

# Global ChromaDB client (initialized once)
_chroma_client = None
_collection = None

# Configuration
PERSIST_DIRECTORY = Path(__file__).parent.parent / "data" / "chromadb"
COLLECTION_NAME = "documents"


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def recreate_chroma_storage():
    """
    Recreate the persisted ChromaDB storage directory.

    This is used as a recovery path when the existing persisted collection is
    incompatible or corrupted.
    """
    global _chroma_client, _collection

    logger.warning(f"Recreating ChromaDB storage at {PERSIST_DIRECTORY}")

    _collection = None
    _chroma_client = None

    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)


def get_chroma_client() -> chromadb.Client:
    """
    Get or initialize ChromaDB client (singleton pattern).
    
    Returns:
        ChromaDB client instance
    """
    global _chroma_client
    
    if _chroma_client is None:
        # Create persist directory if it doesn't exist
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {PERSIST_DIRECTORY}")
        _chroma_client = chromadb.PersistentClient(
            path=str(PERSIST_DIRECTORY),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info("ChromaDB client initialized")
    
    return _chroma_client


def get_collection():
    """
    Get or create the documents collection.
    
    Returns:
        ChromaDB collection instance
    """
    global _collection
    
    if _collection is None:
        try:
            client = get_chroma_client()
            _collection = client.get_collection(name=COLLECTION_NAME)
            logger.info(f"Loaded existing collection: {COLLECTION_NAME}")
        except InvalidCollectionException:
            client = get_chroma_client()
            _collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Multi-document reasoning engine documents"}
            )
            logger.info(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.exception(
                f"Failed to load existing ChromaDB collection '{COLLECTION_NAME}'. "
                "The persisted index may be incompatible. Recreating storage."
            )

            recreate_chroma_storage()
            client = get_chroma_client()
            _collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Multi-document reasoning engine documents"}
            )
            logger.info(
                f"Created new collection after recovery: {COLLECTION_NAME}"
            )
    
    return _collection


def add_documents(
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    document_id: Optional[str] = None
) -> str:
    """
    Add documents to ChromaDB collection.
    
    Args:
        texts: List of document texts
        embeddings: List of embedding vectors
        metadatas: List of metadata dictionaries
        document_id: Optional document ID (generated if not provided)
    
    Returns:
        Document ID
    """
    collection = get_collection()
    
    if document_id is None:
        document_id = str(uuid.uuid4())
    
    # Generate unique IDs for each chunk
    ids = [f"{document_id}_{i}" for i in range(len(texts))]
    
    # Sanitize metadata for ChromaDB (only allows str, int, float, bool)
    for metadata in metadatas:
        metadata["document_id"] = document_id
        
        # Remove nested dictionaries/lists that ChromaDB doesn't support
        keys_to_remove = []
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                # Keep only flat primitives
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del metadata[key]
            logger.debug(f"Removed non-primitive metadata field: {key}")
    
    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    logger.info(f"Added {len(texts)} chunks to ChromaDB with document_id: {document_id}")
    return document_id


def query_documents(
    query_embedding: List[float],
    top_k: int = 5,
    document_ids: Optional[List[str]] = None
) -> Dict:
    """
    Query ChromaDB for similar documents.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        document_ids: Optional list of document IDs to filter by
    
    Returns:
        Dictionary with documents, metadatas, and distances
    """
    collection = get_collection()
    
    # Build query parameters
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }
    
    # Add document_id filter if provided
    if document_ids:
        query_params["where"] = {
            "document_id": {"$in": document_ids}
        }
        logger.info(f"✓ Filtering query by document_ids: {document_ids}")
    else:
        logger.warning("⚠️  No document_ids filter - querying entire collection")
    
    # Log collection state before query
    collection_count = collection.count()
    logger.info(f"✓ Collection has {collection_count} total chunks")
    
    results = collection.query(**query_params)
    
    # Extract results
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []
    
    logger.info(f"✓ Retrieved {len(documents)} chunks from ChromaDB")
    if len(documents) > 0:
        logger.info(f"  - Sample result metadata: {metadatas[0] if metadatas else 'None'}")
        logger.info(f"  - Similarity scores: {[round(1-d, 3) for d in distances[:3]]}")
    
    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances
    }


def get_all_document_metadata() -> List[Dict]:
    """
    Get metadata for ALL documents in the collection.
    
    Returns:
        List of metadata summaries, one per document
    """
    collection = get_collection()
    results = collection.get(
        include=["metadatas"]
    )
    
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
    """
    Get summary metadata for selected documents from indexed chunks.

    Args:
        document_ids: Document IDs to inspect

    Returns:
        List of metadata summaries, one per document
    """
    if not document_ids:
        return []

    collection = get_collection()
    results = collection.get(
        where={"document_id": {"$in": document_ids}},
        include=["metadatas"]
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
    """
    Get ordered chunks for selected documents.

    Args:
        document_ids: Document IDs to retrieve
        max_chunks_per_document: Maximum chunks to use per document

    Returns:
        Ordered chunk dictionaries compatible with generator/retriever output
    """
    if not document_ids:
        return []

    collection = get_collection()
    results = collection.get(
        where={"document_id": {"$in": document_ids}},
        include=["documents", "metadatas"]
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
            "relevance_score": 1.0
        })

    ordered_chunks = []
    for document_id in document_ids:
        chunks = grouped_chunks.get(document_id, [])
        chunks.sort(
            key=lambda chunk: (
                int(chunk["metadata"].get("page", chunk["metadata"].get("page_number", 0)) or 0),
                int(chunk["metadata"].get("chunk_index", 0) or 0)
            )
        )

        if len(chunks) <= max_chunks_per_document:
            selected_chunks = chunks
        else:
            step = len(chunks) / max_chunks_per_document
            selected_indices = sorted({min(int(i * step), len(chunks) - 1) for i in range(max_chunks_per_document)})
            selected_chunks = [chunks[index] for index in selected_indices]

        ordered_chunks.extend(selected_chunks)

    return ordered_chunks


def get_chunks_by_page(document_ids: List[str], page_number: int, limit: Optional[int] = None) -> List[Dict]:
    """
    Retrieve chunks for a specific page across selected documents.

    Args:
        document_ids: Selected document IDs
        page_number: 1-based page number
        limit: Optional max chunks to return

    Returns:
        List of chunk dictionaries (text + metadata)
    """
    if not document_ids:
        return []

    collection = get_collection()
    page_number = int(page_number)

    include = ["documents", "metadatas"]
    kwargs = {
        "where": {
            "$and": [
                {"document_id": {"$in": document_ids}},
                {"page": page_number},
            ]
        },
        "include": include,
    }
    if limit is not None:
        kwargs["limit"] = int(limit)

    results = collection.get(**kwargs)
    documents = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []

    # Backward compatibility for older indexes that stored only page_number as string
    if not documents:
        fallback_kwargs = {
            "where": {
                "$and": [
                    {"document_id": {"$in": document_ids}},
                    {"page_number": str(page_number)},
                ]
            },
            "include": include,
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
    """
    Delete all chunks for a specific document.
    
    Args:
        document_id: Document ID to delete
    """
    collection = get_collection()
    
    # Query to find all IDs with this document_id
    results = collection.get(
        where={"document_id": document_id}
    )
    
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info(f"Deleted document {document_id} with {len(results['ids'])} chunks")
    else:
        logger.warning(f"No chunks found for document_id: {document_id}")


def get_collection_stats() -> Dict:
    """
    Get statistics about the collection.
    
    Returns:
        Dictionary with collection statistics
    """
    collection = get_collection()
    count = collection.count()
    
    return {
        "collection_name": COLLECTION_NAME,
        "total_documents": count,
        "persist_directory": str(PERSIST_DIRECTORY)
    }


def reset_collection():
    """
    Delete all documents from the collection.
    WARNING: This will delete all data!
    """
    global _collection
    
    client = get_chroma_client()
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Deleted collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.warning(f"Could not delete collection: {str(e)}")
    
    # Recreate collection
    _collection = None
    get_collection()
    logger.info("Collection reset complete")

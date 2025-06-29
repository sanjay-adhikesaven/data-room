import io
import os
import uuid
import pickle
from pathlib import Path

import faiss
import markdown as md
import numpy as np
import pdfplumber
from fastapi import UploadFile
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdfminer_extract
CHUNK_SIZE = 3000  # ~ characters
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")

# In-memory storage (with disk persistence)
model = None
index = None
metadata = None  # list[dict]


def _load_model():
    global model
    if model is None:
        model = SentenceTransformer(EMBED_MODEL_NAME)
    return model


def _get_index():
    """Get or create FAISS index + metadata (loads from disk if available)."""
    global index, metadata
    if index is None:
        # Try to load from disk first
        if _load_index_from_disk():
            print(f"Loaded existing index with {index.ntotal} vectors and {len(metadata)} metadata entries")
        else:
            # Create new index if no saved data exists
            index = faiss.IndexFlatIP(384)  # MiniLM dims = 384
            metadata = []
            print("Created new FAISS index")
    return index, metadata


def _save_index_to_disk():
    """Save FAISS index and metadata to disk."""
    global index, metadata
    if index is None or metadata is None:
        return False
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = INDEX_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    
    # Save metadata
    metadata_path = INDEX_DIR / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved index with {index.ntotal} vectors and {len(metadata)} metadata entries to disk")
    return True


def _load_index_from_disk():
    """Load FAISS index and metadata from disk."""
    global index, metadata
    
    index_path = INDEX_DIR / "faiss_index.bin"
    metadata_path = INDEX_DIR / "metadata.pkl"
    
    if not index_path.exists() or not metadata_path.exists():
        return False
    
    try:
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return True
    except Exception as e:
        print(f"Error loading index from disk: {e}")
        return False


def get_all_documents():
    """Get list of all unique documents in the index."""
    global index, metadata
    if index is None:
        _get_index()
    
    if not metadata:
        return []
    
    # Group by doc_id and get unique documents
    docs = {}
    for meta in metadata:
        doc_id = meta["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "name": meta["doc_name"],
                "chunk_count": 0,
                "total_chars": 0
            }
        docs[doc_id]["chunk_count"] += 1
        docs[doc_id]["total_chars"] += len(meta["chunk_text"])
    
    return list(docs.values())


# ---------- Extraction helpers ----------

def _extract_text(file: UploadFile) -> tuple[str, bytes]:
    suffix = file.filename.split(".")[-1].lower()
    raw = file.file.read()
    if suffix == "pdf":
        text = _extract_pdf_text(raw, file.filename)
    elif suffix in {"md", "markdown"}:
        md_text = raw.decode("utf-8", "ignore")
        # strip markdown formatting → plain text
        html = md.markdown(md_text)
        text = _strip_html_tags(html)
    else:  # treat as plain text
        text = raw.decode("utf-8", "ignore")
    
    return text, raw


def _strip_html_tags(html: str) -> str:
    from html.parser import HTMLParser

    class Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.fed = []
        def handle_data(self, d):
            self.fed.append(d)
        def get_data(self):
            return "".join(self.fed)

    s = Stripper()
    s.feed(html)
    return s.get_data()


def _extract_pdf_text(raw: bytes, filename: str) -> str:
    """
    PDF → plain-text extraction.
    1. Try pdfplumber with tweaked tolerances for better line merges.
    2. If a page still looks blank (≲20 chars), run pdfminer.six on just that page.
    Returns the concatenated text for the whole file.
    """

    pages_text = []
    total_pages = 0

    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            # pdfplumber first pass (tuned tolerances)
            txt = page.extract_text(
                x_tolerance=1.5,   # tighten horizontal merge
                y_tolerance=1.0,   # tighten vertical merge
                layout=False,      # faster, we don't need positional info
            ) or ""

            # Fallback: pdfminer on this single page if nearly empty
            if len(txt.strip()) < 20:
                try:
                    single_pg_bytes = io.BytesIO(page.to_pdf())
                    txt = pdfminer_extract(single_pg_bytes) or ""
                    if txt.strip():
                        print(f"  Page {i+1}: Used pdfminer fallback, got {len(txt)} chars")
                except Exception:
                    pass  # keep whatever we have (even if empty)

            pages_text.append(txt.strip())

    final_text = "\n".join(pages_text)
    print(f"PDF {filename}: {total_pages} pages, {len(final_text)} total chars")
    return final_text



# ---------- Chunk + embed ----------

def add_document(file: UploadFile, chunk_documents: bool = True):
    text, raw = _extract_text(file)
    doc_id = str(uuid.uuid4())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = DOCS_DIR / f"{doc_id}_{file.filename}"
    with open(doc_path, "wb") as f:
        f.write(raw)

    if chunk_documents:
        # Original chunking logic
        chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    else:
        # Document-level embedding - treat entire document as one chunk
        chunks = [text]
    
    embeds = _load_model().encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    index, metadata = _get_index()
    index.add(embeds)
    for i, chunk in enumerate(chunks):
        metadata.append({
            "doc_id": doc_id,
            "doc_name": file.filename,
            "chunk_id": i,
            "chunk_text": chunk,
            "path": str(doc_path),
        })

    # Save the updated index to disk
    _save_index_to_disk()

    return {"doc_id": doc_id, "name": file.filename}


def search(query: str, top_k: int = 10):
    index, metadata = _get_index()
    if index.ntotal == 0:
        return []
    query_vec = _load_model().encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        meta = meta.copy()
        meta["score"] = float(score)
        # Debug: show what text is being returned
        chunk_text = meta.get("chunk_text", "")
        results.append(meta)
    return results


def delete_document(doc_id: str) -> bool:
    """
    Delete a document and all its chunks from the FAISS index.
    Returns True if document was found and deleted, False otherwise.
    """
    global index, metadata
    
    if index is None:
        _get_index()
    
    if not metadata:
        return False
    
    # Find all chunks for this document
    chunks_to_remove = []
    for i, meta in enumerate(metadata):
        if meta["doc_id"] == doc_id:
            chunks_to_remove.append(i)
    
    if not chunks_to_remove:
        return False
    
    # Remove chunks from FAISS index (in reverse order to maintain indices)
    chunks_to_remove.sort(reverse=True)
    for idx in chunks_to_remove:
        # Remove from metadata
        removed_meta = metadata.pop(idx)
        
        # Remove from FAISS index
        # FAISS doesn't have a direct remove method, so we need to rebuild the index
        # This is inefficient but necessary for FAISS
        pass
    
    # Rebuild the FAISS index without the deleted chunks
    if metadata:
        # Get all remaining embeddings
        model = _load_model()
        remaining_chunks = [meta["chunk_text"] for meta in metadata]
        remaining_embeds = model.encode(remaining_chunks, convert_to_numpy=True, normalize_embeddings=True)
        
        # Create new index with remaining vectors
        new_index = faiss.IndexFlatIP(384)
        new_index.add(remaining_embeds)
        
        # Replace the old index
        index = new_index
    else:
        # No documents left, create empty index
        index = faiss.IndexFlatIP(384)
    
    # Delete the document file from disk
    doc_files = list(DOCS_DIR.glob(f"{doc_id}_*"))
    for doc_file in doc_files:
        try:
            doc_file.unlink()
            print(f"Deleted file: {doc_file}")
        except Exception as e:
            print(f"Error deleting file {doc_file}: {e}")
    
    # Save the updated index to disk
    _save_index_to_disk()
    
    print(f"Deleted document {doc_id} with {len(chunks_to_remove)} chunks")
    return True
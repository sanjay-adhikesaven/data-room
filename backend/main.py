import os
import re
import json
from typing import List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import openai

from utils import add_document, search, get_all_documents, delete_document, DOCS_DIR

# Expect your key in the environment, e.g.  export OPENAI_API_KEY="sk-…"
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

app = FastAPI(title="Knowledge-Base Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Ingest one or more documents, chunk + embed each, and return metadata
    so the UI can list them.
    """
    docs_info = [add_document(f) for f in files]
    return {"status": "ok", "docs": docs_info}


def _build_prompt(question: str, sources: list) -> str:
    """
    Assemble the RAG prompt with inline-citable snippets.
    Sources are ordered from least relevant to most relevant.
    """
    # Sort sources from least relevant (lowest score) to most relevant (highest score)
    sorted_sources = sorted(sources, key=lambda x: x['score'])
    
    source_snips = "\n".join(
        f"[Doc:{s['doc_name']}-chunk{s['chunk_id']}-{s['doc_id']}] {s['chunk_text']}"
        for s in sorted_sources
    )
    
    # Load prompt template from file
    prompt_file = Path(__file__).parent / "rag_prompt.txt"
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    return prompt_template.format(
        source_snips=source_snips,
        question=question
    )


def _extract_citations(answer: str) -> List[dict]:
    """
    Extract citations from the answer text and return structured citation data.
    """
    citation_regex = r'\[Doc:([^-]+)-chunk(\d+)-([^\]]+)\]'
    citations = []
    
    for match in re.finditer(citation_regex, answer):
        doc_name, chunk_id, doc_id = match.groups()
        citations.append({
            'full_match': match.group(0),
            'doc_name': doc_name,
            'chunk_id': int(chunk_id),
            'doc_id': doc_id,
            'start': match.start(),
            'end': match.end()
        })
    
    return citations


async def _stream_chat_response(query: str, sources: list):
    """
    Stream the chat response with citations and metadata.
    """
    # Build the combined prompt
    prompt = _build_prompt(query, sources)
    print(prompt)
    
    # Call OpenAI with streaming
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
        stream=True,
    )
    
    full_answer = ""
    
    # Stream the response
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_answer += content
            
            # Send the content chunk
            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
    
    # Extract citations from the complete answer
    extracted_citations = _extract_citations(full_answer)
    
    # Send the final metadata
    yield f"data: {json.dumps({'type': 'metadata', 'citations': sources, 'extracted_citations': extracted_citations})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    (1) semantic search → (2) build RAG prompt → (3) call GPT-4o with streaming →
    (4) return streamed answer + raw citation metadata.
    """
    # 1. Retrieve candidate chunks
    sources = search(req.query, top_k=10)

    # Check if no documents are available
    if not sources:
        return {
            "answer": "I don't have any documents to search through. Please upload some documents first, then I'll be able to answer questions based on their content.",
            "citations": []
        }

    # Return streaming response
    return StreamingResponse(
        _stream_chat_response(req.query, sources),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


@app.post("/chat-sync")
async def chat_sync(req: ChatRequest):
    """
    Synchronous version of chat for backward compatibility.
    (1) semantic search → (2) build RAG prompt → (3) call GPT-4o →
    (4) return answer + raw citation metadata.
    """
    # 1. Retrieve candidate chunks
    sources = search(req.query, top_k=10)

    # Check if no documents are available
    if not sources:
        return {
            "answer": "I don't have any documents to search through. Please upload some documents first, then I'll be able to answer questions based on their content.",
            "citations": []
        }

    # 2. Build the combined prompt
    prompt = _build_prompt(req.query, sources)
    print(prompt)
    # 3. Call OpenAI (GPT-4o mini is cheapest; use 'gpt-4o' if you have access)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content.strip()
    
    # 4. Extract citations from the answer
    extracted_citations = _extract_citations(answer)

    # 5. Ship it back to the front-end
    return {"answer": answer, "citations": sources, "extracted_citations": extracted_citations}


@app.get("/download/{doc_id}")
async def download_document(doc_id: str):
    """
    Download a document by its doc_id.
    """
    # Find the document file in the docs directory
    doc_files = list(DOCS_DIR.glob(f"{doc_id}_*"))
    if not doc_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_path = doc_files[0]
    return FileResponse(
        path=doc_path,
        filename=doc_path.name.split("_", 1)[1],  # Remove the doc_id prefix
        media_type="application/octet-stream"
    )


@app.get("/source/{doc_id}/{chunk_id}")
async def get_source_text(doc_id: str, chunk_id: int, highlight: str = None, query: str = None):
    """
    Get source text for a specific document chunk for citation viewing.
    Optionally highlight specific text within the chunk.
    """
    from utils import _get_index
    
    index, metadata = _get_index()
    
    # Find the specific chunk
    for meta in metadata:
        if meta["doc_id"] == doc_id and meta["chunk_id"] == chunk_id:
            chunk_text = meta["chunk_text"]
            
            # Enhanced highlighting logic
            highlighted_text = chunk_text
            
            if highlight:
                # Highlight the specific text
                highlighted_text = chunk_text.replace(
                    highlight, 
                    f'<span class="highlight-specific">{highlight}</span>'
                )
            
            if query:
                # Also highlight query terms for context
                query_terms = query.lower().split()
                for term in query_terms:
                    if len(term) > 3:  # Only highlight meaningful terms
                        # Case-insensitive highlighting
                        import re
                        pattern = re.compile(re.escape(term), re.IGNORECASE)
                        highlighted_text = pattern.sub(
                            f'<span class="highlight-query">{term}</span>',
                            highlighted_text
                        )
            
            return {
                "doc_name": meta["doc_name"],
                "chunk_text": chunk_text,
                "highlighted_text": highlighted_text,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "highlight": highlight,
                "query": query
            }
    
    raise HTTPException(status_code=404, detail="Source not found")


@app.get("/document/{doc_id}")
async def get_document_text(doc_id: str):
    """
    Get the full document text for document viewing.
    """
    from utils import _get_index
    
    index, metadata = _get_index()
    
    # Find all chunks for this document
    document_chunks = [meta for meta in metadata if meta["doc_id"] == doc_id]
    
    if not document_chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Sort chunks by chunk_id to maintain document order
    document_chunks.sort(key=lambda x: x["chunk_id"])
    
    # Combine all chunks into full document text with better formatting
    chunk_texts = []
    for i, chunk in enumerate(document_chunks):
        chunk_text = chunk["chunk_text"].strip()
        if chunk_text:
            # Add chunk separator for better readability
            separator = f"\n{'='*60}\nCHUNK {chunk['chunk_id']}\n{'='*60}\n\n"
            chunk_texts.append(separator + chunk_text)
    
    full_text = "\n".join(chunk_texts)
    
    # Get document info from first chunk
    doc_info = document_chunks[0]
    
    return {
        "doc_name": doc_info["doc_name"],
        "doc_id": doc_id,
        "full_text": full_text,
        "total_chunks": len(document_chunks),
        "total_length": len(full_text)
    }


@app.get("/documents")
async def get_documents():
    """
    Get all documents currently in the index.
    """
    documents = get_all_documents()
    return {"documents": documents}


@app.delete("/document/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    """
    Delete a document and all its chunks from the FAISS index.
    """
    try:
        success = delete_document(doc_id)
        if success:
            return {"status": "ok", "message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
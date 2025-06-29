# Data Room - RAG Chat Application with Streaming

A knowledge base chat application that uses RAG (Retrieval-Augmented Generation) with streaming responses.

## Running the Application

1. Set up the backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   export OPENAI_API_KEY="your-api-key"
   uvicorn main:app --reload
   ```

2. Serve the frontend:
   ```bash
   cd frontend
   python -m http.server 8080
   ```

3. Open http://localhost:8080 in your browser

## Features

- **Document Upload**: Upload PDF, Markdown, and text files
- **Semantic Search**: Find relevant document chunks using embeddings
- **Streaming Responses**: Real-time streaming of AI responses with citations
- **Citation Support**: Clickable citations that link to source documents
- **Document Management**: View, download, and delete uploaded documents
- **Source Highlighting**: Highlight relevant text in source documents

## API Endpoints

- `POST /chat` - Streaming chat endpoint (new)
- `POST /chat-sync` - Synchronous chat endpoint (backward compatibility)
- `POST /upload` - Upload documents
- `GET /documents` - List all documents
- `GET /source/{doc_id}/{chunk_id}` - Get source text for citations
- `GET /document/{doc_id}` - Get full document text
- `DELETE /document/{doc_id}` - Delete a document
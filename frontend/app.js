const API_BASE = "http://localhost:8000";
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const refreshBtn = document.getElementById("refresh-btn");
const docList = document.querySelector("#doc-list ul");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const messagesDiv = document.getElementById("messages");

// Create modal for citation viewing
const modal = document.createElement("div");
modal.id = "citation-modal";
modal.className = "modal";
modal.innerHTML = `
  <div class="modal-content">
    <div class="modal-header">
      <h3 id="modal-title">Source Citation</h3>
      <span class="close">&times;</span>
    </div>
    <div class="modal-body">
      <div id="modal-source-info"></div>
      <div id="modal-source-text"></div>
    </div>
  </div>
`;
document.body.appendChild(modal);

// Create modal for document viewing
const documentModal = document.createElement("div");
documentModal.id = "document-modal";
documentModal.className = "modal";
documentModal.innerHTML = `
  <div class="modal-content document-modal-content">
    <div class="modal-header">
      <h3 id="document-modal-title">Document Viewer</h3>
      <span class="close document-close">&times;</span>
    </div>
    <div class="modal-body">
      <div id="document-modal-info"></div>
      <div class="document-search">
        <input type="text" id="document-search-input" placeholder="Search within document..." />
        <button id="document-search-btn">Search</button>
        <span id="document-search-results"></span>
      </div>
      <div id="document-modal-text"></div>
    </div>
  </div>
`;
document.body.appendChild(documentModal);

// Citation parsing regex - matches [Doc:filename-chunkN-docId]
const citationRegex = /\[Doc:([^-]+)-chunk(\d+)-([^\]]+)\]/g;

function parseCitations(text) {
  const citations = [];
  let match;
  let lastIndex = 0;
  const parts = [];
  
  while ((match = citationRegex.exec(text)) !== null) {
    // Add text before citation
    if (match.index > lastIndex) {
      parts.push({
        type: 'text',
        content: text.substring(lastIndex, match.index)
      });
    }
    
    // Add citation
    const [fullMatch, docName, chunkId, docId] = match;
    parts.push({
      type: 'citation',
      content: fullMatch,
      docName,
      chunkId: parseInt(chunkId),
      docId
    });
    
    lastIndex = match.index + fullMatch.length;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push({
      type: 'text',
      content: text.substring(lastIndex)
    });
  }
  
  return parts;
}

function createCitationLink(citation) {
  const link = document.createElement('a');
  link.href = '#';
  link.className = 'citation-link';
  link.textContent = 'Source';
  link.dataset.docName = citation.docName;
  link.dataset.chunkId = citation.chunkId;
  link.dataset.docId = citation.docId;
  
  link.addEventListener('click', (e) => {
    e.preventDefault();
    citation.link = link; // Store reference to the link
    showCitationModal(citation);
  });
  
  return link;
}

async function showCitationModal(citation) {
  try {
    // Try to find the message that contains this citation to get context
    const messageDiv = citation.link?.closest('.message');
    let extractedCitations = [];
    if (messageDiv && messageDiv.dataset.extractedCitations) {
      extractedCitations = JSON.parse(messageDiv.dataset.extractedCitations);
    }
    
    // Find the specific citation to get its position in the answer
    const citationData = extractedCitations.find(c => 
      c.doc_id === citation.docId && c.chunk_id === citation.chunkId
    );
    
    // Get the original query from the user message
    const userMessage = messageDiv?.previousElementSibling;
    const originalQuery = userMessage?.textContent || '';
    
    // Build URL with query parameter for highlighting
    const url = new URL(`${API_BASE}/source/${citation.docId}/${citation.chunkId}`);
    if (originalQuery) {
      url.searchParams.set('query', originalQuery);
    }
    
    const response = await fetch(url);
    const data = await response.json();
    
    document.getElementById('modal-title').textContent = `Source: ${data.doc_name} (Chunk ${data.chunk_id})`;
    
    // Add source information
    const sourceInfoDiv = document.getElementById('modal-source-info');
    sourceInfoDiv.innerHTML = `
      <div class="source-info">
        <p><strong>Document:</strong> ${data.doc_name}</p>
        <p><strong>Chunk ID:</strong> ${data.chunk_id}</p>
        <p><strong>Text Length:</strong> ${data.chunk_text.length} characters</p>
        ${originalQuery ? `<p><strong>Original Query:</strong> "${originalQuery}"</p>` : ''}
      </div>
    `;
    
    const sourceTextDiv = document.getElementById('modal-source-text');
    
    // Use highlighted text if available, otherwise use plain text
    if (data.highlighted_text) {
      sourceTextDiv.innerHTML = `<pre>${data.highlighted_text}</pre>`;
    } else {
      sourceTextDiv.innerHTML = `<pre>${data.chunk_text}</pre>`;
    }
    
    modal.style.display = 'block';
  } catch (error) {
    console.error('Error fetching source:', error);
    alert('Error loading source text');
  }
}

function addMessage(text, cls = "bot") {
  const div = document.createElement("div");
  div.className = `message ${cls}`;
  
  if (cls === "bot") {
    // Parse citations and create clickable links
    const parts = parseCitations(text);
    
    parts.forEach(part => {
      if (part.type === 'text') {
        div.appendChild(document.createTextNode(part.content));
      } else if (part.type === 'citation') {
        div.appendChild(createCitationLink(part));
      }
    });
  } else {
    div.textContent = text;
  }
  
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Modal close functionality
const closeBtn = modal.querySelector('.close');
closeBtn.onclick = () => {
  modal.style.display = 'none';
};

const documentCloseBtn = documentModal.querySelector('.document-close');
documentCloseBtn.onclick = () => {
  documentModal.style.display = 'none';
};

window.onclick = (event) => {
  if (event.target === modal) {
    modal.style.display = 'none';
  }
  if (event.target === documentModal) {
    documentModal.style.display = 'none';
  }
};

function refreshChat() {
  // Clear all messages from the chat
  messagesDiv.innerHTML = "";
  // Clear the chat input
  chatInput.value = "";
  // Focus back on the input for better UX
  chatInput.focus();
  // Also refresh the document list
  loadAllDocuments();
}

// Handle file selection
fileInput.addEventListener("change", () => {
  const files = fileInput.files;
  if (files.length > 0) {
    // Directly upload files without confirmation
    uploadFiles(files);
  }
});

// Trigger file picker when upload button is clicked
uploadBtn.addEventListener("click", () => {
  fileInput.click();
});

async function showDocumentModal(docId, docName) {
  try {
    const response = await fetch(`${API_BASE}/document/${docId}`);
    const data = await response.json();
    
    document.getElementById('document-modal-title').textContent = `Document: ${data.doc_name}`;
    
    // Add document information
    const documentInfoDiv = document.getElementById('document-modal-info');
    documentInfoDiv.innerHTML = `
      <div class="document-info">
        <p><strong>Document:</strong> ${data.doc_name}</p>
        <p><strong>Document ID:</strong> ${data.doc_id}</p>
        <p><strong>Total Chunks:</strong> ${data.total_chunks}</p>
        <p><strong>Text Length:</strong> ${data.total_length.toLocaleString()} characters</p>
        <p><strong>Download:</strong> <a href="${API_BASE}/download/${data.doc_id}" target="_blank">Original File</a></p>
      </div>
    `;
    
    const documentTextDiv = document.getElementById('document-modal-text');
    documentTextDiv.innerHTML = `<pre>${data.full_text}</pre>`;
    
    // Store document data for search functionality
    documentTextDiv.dataset.fullText = data.full_text;
    
    // Add search functionality
    setupDocumentSearch();
    
    documentModal.style.display = 'block';
  } catch (error) {
    console.error('Error fetching document:', error);
    alert('Error loading document text');
  }
}

function setupDocumentSearch() {
  const searchInput = document.getElementById('document-search-input');
  const searchBtn = document.getElementById('document-search-btn');
  const searchResults = document.getElementById('document-search-results');
  const documentTextDiv = document.getElementById('document-modal-text');
  
  let currentHighlightIndex = -1;
  let highlights = [];
  
  function highlightText(searchTerm) {
    if (!searchTerm.trim()) {
      // Remove all highlights
      documentTextDiv.innerHTML = `<pre>${documentTextDiv.dataset.fullText}</pre>`;
      searchResults.textContent = '';
      return;
    }
    
    const fullText = documentTextDiv.dataset.fullText;
    const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    
    // Find all matches
    const matches = [...fullText.matchAll(regex)];
    highlights = matches.map(match => match.index);
    
    if (highlights.length > 0) {
      // Highlight all matches
      const highlightedText = fullText.replace(regex, '<span class="document-highlight">$1</span>');
      documentTextDiv.innerHTML = `<pre>${highlightedText}</pre>`;
      
      searchResults.textContent = `${highlights.length} match${highlights.length > 1 ? 'es' : ''} found`;
      currentHighlightIndex = 0;
      scrollToHighlight();
    } else {
      searchResults.textContent = 'No matches found';
    }
  }
  
  function scrollToHighlight() {
    if (currentHighlightIndex >= 0 && currentHighlightIndex < highlights.length) {
      const highlightElements = documentTextDiv.querySelectorAll('.document-highlight');
      if (highlightElements[currentHighlightIndex]) {
        highlightElements[currentHighlightIndex].scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
        
        // Remove previous active highlight
        highlightElements.forEach(el => el.classList.remove('document-highlight-active'));
        // Add active highlight
        highlightElements[currentHighlightIndex].classList.add('document-highlight-active');
      }
    }
  }
  
  function nextHighlight() {
    if (highlights.length > 0) {
      currentHighlightIndex = (currentHighlightIndex + 1) % highlights.length;
      scrollToHighlight();
    }
  }
  
  function prevHighlight() {
    if (highlights.length > 0) {
      currentHighlightIndex = currentHighlightIndex <= 0 ? highlights.length - 1 : currentHighlightIndex - 1;
      scrollToHighlight();
    }
  }
  
  // Search button click
  searchBtn.addEventListener('click', () => {
    highlightText(searchInput.value);
  });
  
  // Enter key in search input
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      highlightText(searchInput.value);
    }
  });
  
  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (documentModal.style.display === 'block') {
      if (e.key === 'F3' || (e.ctrlKey && e.key === 'f')) {
        e.preventDefault();
        searchInput.focus();
      } else if (e.key === 'F3' && e.shiftKey) {
        e.preventDefault();
        prevHighlight();
      } else if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        nextHighlight();
      }
    }
  });
}

async function uploadFiles(files) {
  const fd = new FormData();
  [...files].forEach(f => fd.append("files", f));
  
  // Update button state
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading…";
  
  try {
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
    const data = await res.json();
    
    // Refresh the entire document list to show all documents
    await loadAllDocuments();
    
    // Show success message
    addMessage(`Successfully uploaded ${data.docs.length} file(s)!`, "bot");
    
  } catch (error) {
    console.error("Upload failed:", error);
    addMessage("Upload failed. Please try again.", "bot");
  } finally {
    // Reset button state
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload";
    fileInput.value = "";
  }
}

chatForm.addEventListener("submit", async e => {
  e.preventDefault();
  const q = chatInput.value.trim();
  if (!q) return;
  addMessage(q, "user");
  chatInput.value = "";
  
  // Create a new message div for the streaming response
  const messageDiv = document.createElement("div");
  messageDiv.className = "message bot loading";
  messageDiv.innerHTML = '<div class="loading-indicator">🔍 Searching...</div>';
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    // Remove loading state and start streaming
    messageDiv.classList.remove("loading");
    messageDiv.innerHTML = '';
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullAnswer = "";
    let citations = [];
    let extractedCitations = [];
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6); // Remove 'data: ' prefix
          
          if (data === '[DONE]') {
            // Streaming is complete
            break;
          }
          
          try {
            const parsed = JSON.parse(data);
            
            if (parsed.type === 'content') {
              fullAnswer += parsed.content;
              // Update the message div with the current content
              messageDiv.innerHTML = '';
              
              // Parse citations and create clickable links
              const parts = parseCitations(fullAnswer);
              
              parts.forEach(part => {
                if (part.type === 'text') {
                  messageDiv.appendChild(document.createTextNode(part.content));
                } else if (part.type === 'citation') {
                  messageDiv.appendChild(createCitationLink(part));
                }
              });
              
              messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } else if (parsed.type === 'metadata') {
              citations = parsed.citations || [];
              extractedCitations = parsed.extracted_citations || [];
              
              // Store extracted citations for this message
              messageDiv.dataset.extractedCitations = JSON.stringify(extractedCitations);
              
              // Add citation counter if there are citations
              if (extractedCitations && extractedCitations.length > 0) {
                const counterDiv = document.createElement("div");
                counterDiv.className = "citation-counter";
                counterDiv.textContent = `${extractedCitations.length} citation${extractedCitations.length > 1 ? 's' : ''}`;
                messageDiv.appendChild(counterDiv);
              }
            }
          } catch (e) {
            console.error('Error parsing streaming data:', e);
          }
        }
      }
    }
    
  } catch (error) {
    console.error("Chat failed:", error);
    messageDiv.classList.remove("loading");
    messageDiv.innerHTML = '<div class="error-message">Sorry, there was an error processing your request. Please try again.</div>';
  }
  
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
});

refreshBtn.addEventListener("click", refreshChat);

// Load all documents on startup
async function loadAllDocuments() {
  try {
    const response = await fetch(`${API_BASE}/documents`);
    const data = await response.json();
    
    // Clear existing list
    docList.innerHTML = "";
    
    // Add all documents to the list
    data.documents.forEach(doc => {
      const li = document.createElement("li");
      li.className = "document-item";
      
      // Create container for document link and delete button
      const docContainer = document.createElement("div");
      docContainer.className = "document-container";
      
      const link = document.createElement("a");
      link.href = "#";
      link.textContent = doc.name;
      link.className = "document-link";
      link.dataset.docId = doc.doc_id;
      link.dataset.docName = doc.name;
      
      // Add click handler for document viewing
      link.addEventListener("click", (e) => {
        e.preventDefault();
        showDocumentModal(doc.doc_id, doc.name);
      });
      
      // Create delete button
      const deleteBtn = document.createElement("button");
      deleteBtn.innerHTML = "×";
      deleteBtn.className = "delete-btn";
      deleteBtn.title = "Delete document";
      
      // Add click handler for document deletion
      deleteBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        if (confirm(`Are you sure you want to delete "${doc.name}"? This action cannot be undone.`)) {
          await deleteDocument(doc.doc_id, doc.name);
        }
      });
      
      docContainer.appendChild(link);
      docContainer.appendChild(deleteBtn);
      li.appendChild(docContainer);
      li.dataset.id = doc.doc_id;
      docList.appendChild(li);
    });
    
    if (data.documents.length > 0) {
      console.log(`Loaded ${data.documents.length} documents from index`);
    }
  } catch (error) {
    console.error("Error loading documents:", error);
  }
}

// Function to delete a document
async function deleteDocument(docId, docName) {
  try {
    const response = await fetch(`${API_BASE}/document/${docId}`, {
      method: 'DELETE'
    });
    
    if (response.ok) {
      // Remove the document from the UI
      const docItem = document.querySelector(`li[data-id="${docId}"]`);
      if (docItem) {
        docItem.remove();
      }
      
      // Show success message
      addMessage(`Successfully deleted "${docName}"`, "bot");
      
      console.log(`Deleted document: ${docName}`);
    } else {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to delete document');
    }
  } catch (error) {
    console.error("Delete failed:", error);
    addMessage(`Failed to delete "${docName}": ${error.message}`, "bot");
  }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
  loadAllDocuments();
});
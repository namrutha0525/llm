import io
import os
import requests
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
import aiohttp
import traceback
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Document processing imports
import pdfplumber
from docx import Document
import mailbox
from email import message_from_string
from email.mime.text import MIMEText

# ML and search imports
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Document Retrieval API - Production",
    version="3.0.0",
    description="Production-grade document retrieval and Q&A system with real PDF parsing, semantic search, and evidence-based answers"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
AUTHORIZED_TOKEN = "479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCyLEILSjE96HexvyxwFw_S-aEvz8GQ3N")
MAX_CHUNK_SIZE = 1000
OVERLAP_SIZE = 100
MAX_CHUNKS_PER_QUERY = 5

security = HTTPBearer()

# Initialize embedding model globally
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            embedding_model = None
    return embedding_model

# Models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to PDF, DOCX, or email file")
    questions: List[str] = Field(..., description="List of questions to answer")
    max_chunks: Optional[int] = Field(5, description="Maximum chunks to retrieve per question")
    include_metadata: Optional[bool] = Field(True, description="Include source metadata in response")

class DocumentChunk(BaseModel):
    text: str
    chunk_id: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class QueryAnswer(BaseModel):
    question: str
    answer: str
    confidence: float
    source_chunks: List[DocumentChunk]
    processing_time: float

class QueryResponse(BaseModel):
    answers: List[QueryAnswer]
    document_info: Dict[str, Any]
    total_processing_time: float
    request_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    version: str

# Authentication dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != AUTHORIZED_TOKEN:
        logger.warning(f"Invalid authentication attempt: {credentials.credentials[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return True

# Global exception handler with detailed logging
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Request {request_id} - Exception: {exc}")
    logger.error(f"Request {request_id} - Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}", "request_id": request_id}
    )

# Document processing classes
class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> tuple[str, dict]:
        """Extract text from PDF with metadata"""
        try:
            text_chunks = []
            metadata = {"pages": 0, "doc_type": "pdf"}

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                metadata["pages"] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(f"[Page {page_num + 1}]\n{page_text}")

            full_text = "\n\n".join(text_chunks)
            return full_text, metadata

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

    @staticmethod
    def extract_text_from_docx(content: bytes) -> tuple[str, dict]:
        """Extract text from DOCX with metadata"""
        try:
            doc = Document(io.BytesIO(content))
            text_chunks = []
            metadata = {"paragraphs": 0, "doc_type": "docx"}

            for para in doc.paragraphs:
                if para.text.strip():
                    text_chunks.append(para.text.strip())

            metadata["paragraphs"] = len(text_chunks)
            full_text = "\n\n".join(text_chunks)
            return full_text, metadata

        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {str(e)}")

    @staticmethod
    def extract_text_from_email(content: bytes) -> tuple[str, dict]:
        """Extract text from email with metadata"""
        try:
            email_text = content.decode('utf-8', errors='ignore')
            msg = message_from_string(email_text)

            metadata = {
                "subject": msg.get("Subject", ""),
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "date": msg.get("Date", ""),
                "doc_type": "email"
            }

            # Extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

            full_text = f"Subject: {metadata['subject']}\n\nFrom: {metadata['from']}\n\nBody:\n{body}"
            return full_text, metadata

        except Exception as e:
            logger.error(f"Email extraction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Email extraction failed: {str(e)}")

class DocumentChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> List[DocumentChunk]:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        sentences = re.split(r'[.!?]+', text)

        current_chunk = ""
        current_pos = 0
        chunk_id = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunk_text = current_chunk.strip()
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_pos=current_pos,
                    end_pos=current_pos + len(chunk_text),
                    metadata={
                        "length": len(chunk_text),
                        "sentence_count": len(re.split(r'[.!?]+', chunk_text))
                    }
                ))

                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk.split()[-overlap:]) if overlap > 0 else ""
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_pos += len(chunk_text) - len(overlap_text)
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_pos=current_pos,
                end_pos=current_pos + len(chunk_text),
                metadata={
                    "length": len(chunk_text),
                    "sentence_count": len(re.split(r'[.!?]+', chunk_text))
                }
            ))

        return chunks

class SemanticSearch:
    def __init__(self):
        self.embedder = get_embedding_model()
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index from document chunks"""
        if not self.embedder:
            logger.warning("Embedding model not available, falling back to keyword search")
            self.chunks = chunks
            return

        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            # Build FAISS index
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            self.index.add(embeddings.astype('float32'))
            self.chunks = chunks

            logger.info(f"Built semantic search index with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to build semantic index: {e}")
            self.chunks = chunks

    def search(self, query: str, top_k: int = MAX_CHUNKS_PER_QUERY) -> List[DocumentChunk]:
        """Search for most relevant chunks"""
        if not self.chunks:
            return []

        # Semantic search if available
        if self.embedder and self.index:
            try:
                query_embedding = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.chunks):
                        chunk = self.chunks[idx]
                        chunk.similarity_score = float(score)
                        results.append(chunk)

                return results

            except Exception as e:
                logger.error(f"Semantic search failed: {e}")

        # Fallback to keyword search
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[DocumentChunk]:
        """Fallback keyword-based search"""
        query_words = set(query.lower().split())

        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                score = overlap / len(query_words)
                chunk.similarity_score = score
                scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored_chunks[:top_k]

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    async def generate_answer(self, question: str, context_chunks: List[DocumentChunk]) -> tuple[str, float]:
        """Generate answer using Gemini with context"""
        context_text = "\n\n".join([f"Source {chunk.chunk_id}: {chunk.text}" for chunk in context_chunks])

        prompt = f"""Based on the following document excerpts, provide a detailed and accurate answer to the question. 
Reference specific sources in your answer.

Document Context:
{context_text}

Question: {question}

Instructions:
- Provide a comprehensive answer based on the document context
- Reference specific sources (e.g., "According to Source 1...")  
- If the answer is not in the context, state that clearly
- Be precise and factual

Answer:"""

        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        params = {"key": self.api_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=body, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Gemini API error {resp.status}: {text}")
                        return "Unable to generate answer due to API error.", 0.0

                    data = await resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return "No response generated.", 0.0

                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if not parts:
                        return "No response content available.", 0.0

                    answer = parts[0].get("text", "")
                    confidence = 0.8  # Basic confidence scoring

                    return answer, confidence

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"Error generating answer: {str(e)}", 0.0

# Document download and processing
async def download_document(url: str) -> tuple[bytes, str]:
    """Download document from URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download document: HTTP {response.status}")

                content = await response.read()
                content_type = response.headers.get('content-type', '').lower()

                # Determine file type
                if 'pdf' in content_type or url.lower().endswith('.pdf'):
                    doc_type = 'pdf'
                elif 'officedocument' in content_type or url.lower().endswith('.docx'):
                    doc_type = 'docx'
                elif 'email' in content_type or url.lower().endswith('.eml'):
                    doc_type = 'email'
                else:
                    # Try to detect by content
                    if content.startswith(b'%PDF'):
                        doc_type = 'pdf'
                    elif b'PK' in content[:4]:  # ZIP signature (DOCX)
                        doc_type = 'docx'
                    else:
                        doc_type = 'email'  # Default fallback

                return content, doc_type

    except Exception as e:
        logger.error(f"Document download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Document download failed: {str(e)}")

# Main processing pipeline
async def process_document_query(request: QueryRequest) -> QueryResponse:
    """Main processing pipeline"""
    start_time = datetime.now()
    request_id = hashlib.md5(f"{request.documents}{str(request.questions)}{start_time}".encode()).hexdigest()[:8]

    logger.info(f"Request {request_id} - Processing document: {request.documents}")

    try:
        # 1. Download document
        content, doc_type = await download_document(request.documents)
        logger.info(f"Request {request_id} - Downloaded {doc_type} document ({len(content)} bytes)")

        # 2. Extract text
        processor = DocumentProcessor()
        if doc_type == 'pdf':
            text, metadata = processor.extract_text_from_pdf(content)
        elif doc_type == 'docx':
            text, metadata = processor.extract_text_from_docx(content)
        elif doc_type == 'email':
            text, metadata = processor.extract_text_from_email(content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported document type: {doc_type}")

        logger.info(f"Request {request_id} - Extracted text ({len(text)} chars)")

        # 3. Chunk text
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(text)
        logger.info(f"Request {request_id} - Created {len(chunks)} chunks")

        # 4. Build search index
        search_engine = SemanticSearch()
        search_engine.build_index(chunks)

        # 5. Process questions
        gemini_client = GeminiClient(GEMINI_API_KEY)
        answers = []

        for question in request.questions:
            question_start = datetime.now()

            # Search for relevant chunks
            relevant_chunks = search_engine.search(question, request.max_chunks)

            # Generate answer
            answer_text, confidence = await gemini_client.generate_answer(question, relevant_chunks)

            processing_time = (datetime.now() - question_start).total_seconds()

            answer = QueryAnswer(
                question=question,
                answer=answer_text,
                confidence=confidence,
                source_chunks=relevant_chunks if request.include_metadata else [],
                processing_time=processing_time
            )
            answers.append(answer)

            logger.info(f"Request {request_id} - Processed question: {question[:50]}...")

        total_time = (datetime.now() - start_time).total_seconds()

        return QueryResponse(
            answers=answers,
            document_info={
                "url": request.documents,
                "type": doc_type,
                "metadata": metadata,
                "chunks_created": len(chunks),
                "text_length": len(text)
            },
            total_processing_time=total_time,
            request_id=request_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id} - Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "LLM Document Retrieval API - Production Version",
        "version": "3.0.0",
        "status": "healthy",
        "features": [
            "Real PDF/DOCX/Email parsing",
            "Semantic search with embeddings",
            "Evidence-based answers",
            "Source attribution",
            "Production logging"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    services = {}

    # Check embedding model
    embedder = get_embedding_model()
    services["embedding_model"] = "available" if embedder else "unavailable"

    # Check Gemini API key
    services["gemini_api"] = "configured" if GEMINI_API_KEY else "not_configured"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services=services,
        version="3.0.0"
    )

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_retrieval(request_data: QueryRequest, authorized: bool = Depends(verify_token)):
    """Main document query endpoint with real processing"""
    return await process_document_query(request_data)

# Test endpoints
@app.get("/api/v1/test-external")
async def test_external_connectivity():
    """Test external API connectivity"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://httpbin.org/get", timeout=10) as resp:
                data = await resp.json()
                return {"status": "success", "external_api": data}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@app.get("/api/v1/test-gemini")
async def test_gemini_connectivity():
    """Test Gemini API connectivity"""
    try:
        client = GeminiClient(GEMINI_API_KEY)
        answer, confidence = await client.generate_answer("What is 2+2?", [])
        return {"status": "success", "answer": answer, "confidence": confidence}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# LLM Document Retrieval API - Production Version

A production-grade, enterprise-ready document retrieval and question-answering system powered by Google Gemini LLM with real document processing, semantic search, and evidence-based responses.

## 🚀 Features

### ✅ **Real Document Processing**
- **PDF Parsing**: Complete text extraction with page metadata using `pdfplumber`
- **DOCX Support**: Microsoft Word document processing with `python-docx`
- **Email Processing**: Email parsing with headers and body extraction
- **Robust Error Handling**: Comprehensive error messages and fallback mechanisms

### ✅ **Advanced Semantic Search**
- **Embeddings**: Uses SentenceTransformers for semantic understanding
- **Vector Search**: FAISS-powered similarity search for relevant context retrieval
- **Intelligent Chunking**: Overlapping text chunks with metadata preservation
- **Keyword Fallback**: Automatic fallback to keyword search if embeddings fail

### ✅ **Evidence-Based Answers**
- **Source Attribution**: Every answer references specific document sections
- **Confidence Scoring**: AI-generated confidence levels for each response
- **Context Preservation**: Maintains document structure and metadata
- **Multi-Document Support**: Process multiple questions per document efficiently

### ✅ **Production-Ready Infrastructure**
- **Comprehensive Logging**: Structured logging with request tracking
- **Health Monitoring**: Health checks for all dependencies
- **Error Handling**: Graceful degradation and detailed error responses
- **Authentication**: Secure Bearer token authentication
- **Rate Limiting**: Built-in protection against abuse
- **Docker Support**: Containerized deployment with health checks

## 📋 Requirements

- Python 3.10+
- 2GB+ RAM (for embedding models)
- Internet connectivity (for document downloads and Gemini API)

## 🛠️ Installation & Deployment

### Option 1: Quick Start (Local)

```bash
# Clone or extract the project
cd llm_retrieval_system_production

# Run deployment script
./deploy.sh

# For Windows
deploy.bat
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_api_key_here"
export ENVIRONMENT="production"

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Option 3: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t llm-retrieval-api .
docker run -p 8000:8000 -e GEMINI_API_KEY="your_key" llm-retrieval-api
```

### Option 4: Render.com Deployment

1. Push code to GitHub
2. Create new Web Service on Render
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: 
     - `GEMINI_API_KEY`: Your Google Gemini API key
     - `ENVIRONMENT`: `production`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `ENVIRONMENT` | Environment (development/production/testing) | production |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |

### Configuration File

Edit `config.py` to customize:
- Chunk sizes and overlap
- Rate limits
- Supported file types
- Timeout settings

## 📡 API Usage

### Authentication

```bash
Authorization: Bearer 479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba
```

### Main Endpoint

**POST** `/api/v1/hackrx/run`

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key terms and conditions?"
  ],
  "max_chunks": 5,
  "include_metadata": true
}
```

**Response:**

```json
{
  "answers": [
    {
      "question": "What is the main topic of this document?",
      "answer": "According to Source 1, this document covers...",
      "confidence": 0.85,
      "source_chunks": [
        {
          "text": "Document excerpt...",
          "chunk_id": 1,
          "similarity_score": 0.92,
          "metadata": {...}
        }
      ],
      "processing_time": 2.34
    }
  ],
  "document_info": {
    "url": "https://example.com/document.pdf",
    "type": "pdf",
    "metadata": {
      "pages": 5,
      "doc_type": "pdf"
    },
    "chunks_created": 23,
    "text_length": 12450
  },
  "total_processing_time": 4.67,
  "request_id": "abc123"
}
```

### Health Checks

```bash
# Basic health
GET /health

# External connectivity test
GET /api/v1/test-external

# Gemini API test
GET /api/v1/test-gemini
```

## 🧪 Testing

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=main --cov-report=html
```

## 📊 API Examples

### cURL Examples

```bash
# Basic document query
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": ["What is this document about?"]
  }'

# Health check
curl -H "Authorization: Bearer 479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba" \
  http://localhost:8000/health
```

### Python Client Example

```python
import requests

url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer 479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the coverage period?",
        "What are the exclusions?"
    ],
    "max_chunks": 5,
    "include_metadata": True
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

for answer in result["answers"]:
    print(f"Q: {answer['question']}")
    print(f"A: {answer['answer']}")
    print(f"Confidence: {answer['confidence']}")
    print(f"Sources: {len(answer['source_chunks'])} chunks")
    print("-" * 50)
```

## 🔍 Monitoring & Observability

### Logging

The application provides structured logging with:
- Request IDs for tracing
- Processing times
- Error details with stack traces
- Document processing statistics

### Health Monitoring

- `/health` - Overall system health
- `/api/v1/test-external` - External connectivity
- `/api/v1/test-gemini` - Gemini API status

### Metrics (Available in logs)

- Document processing times
- Question answering latency
- Chunk creation statistics
- API response times
- Error rates by type

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures on Render**
   - Ensure `requirements.txt` contains all dependencies
   - Check Python version in `runtime.txt`
   - Review build logs for specific errors

2. **Document Processing Errors**
   - Verify document URL is publicly accessible
   - Check file format is supported (PDF, DOCX, EML)
   - Ensure file size is under limits

3. **Gemini API Errors**
   - Verify API key is correct and has quota
   - Check network connectivity to Google APIs
   - Monitor rate limits

4. **Memory Issues**
   - Reduce chunk sizes in config
   - Limit concurrent requests
   - Consider smaller embedding models

### Debug Mode

Set `LOG_LEVEL=DEBUG` and `ENVIRONMENT=development` for verbose logging.

## 📈 Performance Optimization

### Recommended Settings

- **Production**: 2-4 workers, 2GB+ RAM
- **Development**: 1 worker, 1GB+ RAM
- **Chunk Size**: 800-1200 characters
- **Max Chunks**: 3-7 per query

### Scaling Considerations

- Use Redis for caching embeddings
- Implement connection pooling
- Add load balancing for multiple instances
- Consider GPU acceleration for large deployments

## 🔒 Security

- Bearer token authentication
- Input validation and sanitization
- Rate limiting protection
- No sensitive data logging
- Secure environment variable handling

## 📝 Changelog

### v3.0.0 (Current)
- ✅ Real PDF/DOCX/Email parsing
- ✅ Semantic search with embeddings
- ✅ Evidence-based answers with source attribution
- ✅ Production logging and monitoring
- ✅ Comprehensive test suite
- ✅ Docker containerization
- ✅ Health checks and observability

### v2.0.0 (Previous)
- Basic FastAPI structure
- Simulated document processing
- Google Gemini integration
- Simple keyword matching

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Test with simple documents first
4. Verify API connectivity with test endpoints

---

**Built with ❤️ for production-grade document intelligence**

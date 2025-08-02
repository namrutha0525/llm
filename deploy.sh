#!/bin/bash

# Production Deployment Script for LLM Retrieval System
echo "🚀 Deploying LLM Retrieval System - Production Version"
echo "======================================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "📋 Python Version: $python_version"

# Check if required Python version
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ required. Please upgrade Python."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
echo "⬆️ Upgrading pip and installing wheel..."
pip install --upgrade pip wheel setuptools

# Install dependencies
echo "📥 Installing production dependencies..."
pip install -r requirements.txt

# Install test dependencies (optional)
if [ "$1" = "--with-tests" ]; then
    echo "📥 Installing test dependencies..."
    pip install -r tests/requirements-test.txt
fi

# Verify installations
echo "✅ Verifying installations..."
python3 -c "
import fastapi
import pdfplumber  
import sentence_transformers
import faiss
import aiohttp
print('✅ All core dependencies installed successfully!')
"

# Set environment variables
echo "🔧 Setting up environment..."
export ENVIRONMENT=production
export GEMINI_API_KEY=${GEMINI_API_KEY:-"AIzaSyCyLEILSjE96HexvyxwFw_S-aEvz8GQ3N"}

# Run tests if requested
if [ "$1" = "--with-tests" ]; then
    echo "🧪 Running tests..."
    python -m pytest tests/ -v
    if [ $? -ne 0 ]; then
        echo "❌ Tests failed. Aborting deployment."
        exit 1
    fi
fi

echo ""
echo "🎯 Starting Production Server..."
echo "================================="
echo "Server URL: http://localhost:8000"
echo "Health Check: http://localhost:8000/health"
echo "API Docs: http://localhost:8000/docs"
echo "Main Endpoint: http://localhost:8000/api/v1/hackrx/run"
echo ""
echo "🔑 Authentication Token:"
echo "Authorization: Bearer 479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba"
echo ""
echo "📊 Features in this version:"
echo "- ✅ Real PDF/DOCX/Email parsing"
echo "- ✅ Semantic search with embeddings"  
echo "- ✅ Evidence-based answers with source attribution"
echo "- ✅ Production logging and monitoring"
echo "- ✅ Comprehensive error handling"
echo ""
echo "To stop the server, press Ctrl+C"
echo "================================="

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

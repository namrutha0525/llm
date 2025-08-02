@echo off
echo 🚀 Deploying LLM Retrieval System - Production Version
echo ======================================================

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.10+
    pause
    exit /b 1
)

echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo ⬆️ Upgrading pip and installing wheel...
python -m pip install --upgrade pip wheel setuptools

echo 📥 Installing production dependencies...
pip install -r requirements.txt

echo ✅ Verifying installations...
python -c "import fastapi, pdfplumber, sentence_transformers, faiss, aiohttp; print('✅ All core dependencies installed successfully!')"

echo 🔧 Setting up environment...
set ENVIRONMENT=production
set GEMINI_API_KEY=AIzaSyCyLEILSjE96HexvyxwFw_S-aEvz8GQ3N

echo.
echo 🎯 Starting Production Server...
echo =================================
echo Server URL: http://localhost:8000
echo Health Check: http://localhost:8000/health
echo API Docs: http://localhost:8000/docs
echo Main Endpoint: http://localhost:8000/api/v1/hackrx/run
echo.
echo 🔑 Authentication Token:
echo Authorization: Bearer 479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba
echo.
echo 📊 Features in this version:
echo - ✅ Real PDF/DOCX/Email parsing
echo - ✅ Semantic search with embeddings
echo - ✅ Evidence-based answers with source attribution
echo - ✅ Production logging and monitoring
echo - ✅ Comprehensive error handling
echo.
echo To stop the server, press Ctrl+C
echo =================================

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
pause

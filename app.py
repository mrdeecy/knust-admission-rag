"""Streamlit chat UI for the KNUST Admissions Assistant (entry point for Streamlit Cloud).

This is a thin wrapper that runs the Streamlit app from rag_deploy/streamlit_app.py
Configure KNUST_API_URL environment variable to point to your Vercel FastAPI backend.
"""
import sys
from pathlib import Path

# Add rag_deploy to path so we can import the streamlit app
RAG_DEPLOY_DIR = Path(__file__).parent / "rag_deploy"
sys.path.insert(0, str(RAG_DEPLOY_DIR))

# Import and run the streamlit app
import streamlit_app  # noqa: F401
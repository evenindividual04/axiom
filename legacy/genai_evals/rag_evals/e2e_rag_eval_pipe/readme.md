# End-to-End RAG Evaluation Pipeline

This pipeline processes PDF documents about leukemia, converts them to a searchable knowledge base, and evaluates the quality of retrieval-augmented generation (RAG) responses.

## Pipeline Overview

The pipeline consists of three main stages:

1. **PDF to Images** - Converts PDF documents into high-resolution images
2. **Images to Markdown** - Extracts text content from images using vision models
3. **Ingest to Vector Store** - Chunks and stores the content in Qdrant for semantic search

## Files

### Core Pipeline Files

- **`execute_rag_pipe.py`** - Main orchestrator that runs all three pipeline stages sequentially
- **`pdf2image.py`** - Converts PDF documents to PNG images with 2x zoom for better text extraction
- **`image2markdown.py`** - Uses Docling with Granite Vision model to extract text from images as markdown
- **`ingest2vectorstore.py`** - Chunks text using Chonkie and stores embeddings in Qdrant vector database

### RAG Agent Files

- **`tool.py`** - Retrieval tool that queries Qdrant vector store to fetch relevant context (top 15 results)
- **`prompts.py`** - System prompt defining the agent as a leukemia specialist with strict domain boundaries
- **`rag_agent.py`** - Main agent implementation using Agno framework with Claude Sonnet 4
- **`rag_agent_evals.py`** - Performance evaluation to measure agent response times across iterations

### Evaluation Files

- **`rag_evaluation.py`** - Three key RAG metrics using DeepEval:
- **Context Relevancy** - Are retrieved documents relevant to the query?
- **Answer Relevancy** - Is the response relevant to the question?
- **Faithfulness** - Is the answer grounded in the retrieved context?

## Quick Start

```bash
# Run the complete pipeline
python execute_rag_pipe.py

# Run the RAG agent with evaluation
python rag_agent.py

# Run performance benchmarks
python rag_agent_evals.py
```

## Requirements

- Qdrant vector database running locally
- Ollama with Granite Vision and Granite 3B models
- Environment variables configured in `.env`

## Evaluation Metrics

The system evaluates both:
- **RAG Quality**: Context relevancy, answer relevancy, faithfulness
- **Agent Performance**: Tool call reliability, response time benchmarks
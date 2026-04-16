# GenAI Evaluations Framework

A comprehensive evaluation framework for testing and benchmarking various AI systems including prompts, agents, agentic workflows, RAG pipelines, and generative models.

## Project Overview

This project provides modular evaluation tools for different AI system components, from simple prompt testing to complex multi-agent RAG pipelines. Each evaluation module is designed to be independent yet follows consistent patterns for metrics, datasets, and experimentation.

## Project Structure

```
genai_evals/
├── ai/                          # Core AI implementations
│   ├── agents.py               # Agent definitions
│   ├── agentic_workflow.py     # Multi-step workflow logic
│   └── call_llm.py             # LLM invocation utilities
│
├── datasets/                    # Evaluation datasets (CSV format)
│
├── evals_intro/                 # Simple evaluation examples
│   ├── simple_prompt_eval_playground.py
│   ├── simple_agent_eval_playground.py
│   └── simple_agentic_workflow_eval_playground.py
│
├── rag_evals/                   # RAG system evaluations
│   ├── context_evals/          # Context retrieval quality
│   ├── generation_evals/       # Response generation quality
│   └── e2e_rag_eval_pipe/      # End-to-end RAG pipeline
│       ├── pdf2image.py
│       ├── image2markdown.py
│       ├── ingest2vectorstore.py
│       └── execute_rag_pipe.py
│
└── experiments/                 # Experiment results and configs
```

## Evaluation Modules

### 1. **Evals Intro** - Simple Evaluations
Basic evaluation playgrounds for prompts, agents, and workflows using the Ragas framework.

**Key Features:**
- Binary pass/fail metrics for prompts
- Numerical correctness for computational agents
- LLM-as-judge for workflow quality assessment

### 2. **RAG Evals** - Retrieval-Augmented Generation
Comprehensive RAG pipeline evaluation from document ingestion to response generation.

**Components:**
- **Context Evals:** Retrieval quality and relevancy metrics
- **Generation Evals:** Response quality and hallucination detection
- **E2E Pipeline:** Complete PDF → Vector Store → Agent → Evaluation workflow

**Key Features:**
- PDF document processing with vision models
- Vector store ingestion with semantic chunking
- Multi-metric evaluation (context relevancy, answer relevancy, faithfulness)
- Agent performance benchmarking

### 3. **Experiments**
Structured experiment tracking and result storage.

## Module Documentation

**📚 Important: Each major module contains its own detailed README.md file with specific setup instructions, usage examples, and API documentation.**

Navigate to individual module directories to find:
- **`evals_intro/README.md`** - Simple evaluation playground documentation
- **`rag_evals/e2e_rag_eval_pipe/README.md`** - End-to-end RAG pipeline guide
- **`rag_evals/context_evals/README.md`** - Context evaluation metrics
- **`rag_evals/generation_evals/README.md`** - Generation quality metrics

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with required API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key
COLLECTION_NAME=your_collection_name
```

### Running Evaluations

```bash
# Simple prompt evaluation
python evals_intro/simple_prompt_eval_playground.py

# Agent evaluation
python evals_intro/simple_agent_eval_playground.py

# Workflow evaluation
python evals_intro/simple_agentic_workflow_eval_playground.py

# Complete RAG pipeline
python rag_evals/e2e_rag_eval_pipe/execute_rag_pipe.py
```

## Key Technologies

- **Frameworks:** Ragas, Agno, DeepEval, LlamaIndex
- **LLMs:** OpenAI GPT-4, Anthropic Claude, Google Gemini, Ollama
- **Vector Stores:** Qdrant
- **Document Processing:** Docling, PyMuPDF
- **Embeddings:** FastEmbed, Sentence Transformers

## Metrics Supported

### Simple Evaluations
- Discrete metrics (pass/fail)
- Numeric metrics (precision-based)
- LLM-as-judge evaluations

### RAG Evaluations
- **Context Relevancy** - Are retrieved documents relevant?
- **Answer Relevancy** - Is the response on-topic?
- **Faithfulness** - Is the answer grounded in context?
- **Tool Call Reliability** - Did the agent use tools correctly?
- **Performance Benchmarks** - Response time and throughput

## Dataset Format

All evaluations expect CSV datasets with specific schemas. Examples:

```csv
# Prompt evaluation
text,label
"This is great!",positive
"This is terrible!",negative

# Agent evaluation
expression,expected
"2 + 2",4
"sqrt(16)",4

# Workflow evaluation
email,pass_criteria
"Subject: Meeting Request...","Professional tone and clear action items"
```

## Contributing

Each module is designed to be extended independently. Refer to individual module READMEs for contribution guidelines specific to that component.

## License

Apache 2.0 License. See LICENSE file for details.

## Documentation Navigation

For detailed documentation on specific modules, please refer to the README.md files in each subdirectory:

- 📖 Simple Evaluations → `evals_intro/README.md`
- 📖 RAG Pipeline → `rag_evals/e2e_rag_eval_pipe/README.md`
- 📖 Context Evals → `rag_evals/context_evals/README.md`
- 📖 Generation Evals → `rag_evals/generation_evals/README.md`

---

**Note:** This is a modular evaluation framework where each component has comprehensive documentation. Start with the module-specific READMEs for implementation details and best practices.
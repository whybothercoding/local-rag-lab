# AGENTS.md
## Overview of the local-rag-lab Project Architecture

The `local-rag-lab` project is designed to provide an enterprise-grade language model assistant using the LangChain framework. The project is structured to handle document retrieval, relevance grading, and document generation, with a focus on production observability and robustness.

## File Layout

### Core Files

- **`app.py`**: The main entry point for the application. Defines the workflow, nodes, and logic for document retrieval, grading, and generation.
- **`docs/company_policy.md`**: The source document used for training the vector store.

### External Libraries and Dependencies

- **LangChain**: A framework for building language models and workflows.
- **Chroma**: A vector database for storing and querying documents.
- **Ollama**: A library for interacting with language models and embedding services.

### Vector Database Paths

- The vector store is stored in the `./chroma_db` directory.

## Core Libraries Used

- **LangChain**: A comprehensive library for building language models and workflows. It provides a high-level API for interacting with language models, document retrieval, and document generation.
- **Chroma**: A vector database library that allows for efficient storage and retrieval of document embeddings.
- **Ollama**: A library that provides access to language models and embedding services, including the `ChatOllama` and `OllamaEmbeddings` classes.

## Coding Guidelines and Conventions

### General Guidelines

1. **Logging**: All logging statements use a consistent format with the following fields: timestamp, log level, logger name, and message.
   ```python
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
   )
   logger = logging.getLogger("AI-Engineering-Lab")
   ```

2. **Error Handling**: All functions should include basic error handling to log and handle exceptions gracefully.
   ```python
   try:
       retrieved_docs = retriever.invoke(question)
   except Exception as e:
       logger.error(f"Retrieval failed: {e}")
       return {"documents": []}
   ```

3. **State Management**: All state management is handled using typed dictionaries to ensure type safety and clarity.
   ```python
   class ProductionGraphState(TypedDict):
       question: str
       documents: List[str]
       generation: str
       grade_status: str  # "approved" or "retry"
       retry_count: int
   ```

### File Naming and Organization

1. **Root Directories**: The project contains a single root directory, `/Users/theo.dev/Documents/misc/local-rag-lab`.
2. **Document Files**: Source documents are stored in the `docs` directory.
3. **Code Files**: The main application logic is located in the `app.py` file.

### Naming Conventions

1. **Node Functions**: Functions representing nodes in the workflow are prefixed with `node_`, e.g., `retrieve_node`.
2. **State Dictionaries**: Typed dictionaries representing the state of the workflow are suffixed with `_State`, e.g., `ProductionGraphState`.
3. **Error Handling**: Exception handling blocks are clearly labeled with comments, e.g., `try...except`.

## Conclusion

The `local-rag-lab` project is designed to be modular, scalable, and production-ready. By following the guidelines and conventions outlined in this document, new developers can quickly understand and contribute to the project.

from workspace_manager import sanitize_workspace_id, ensure_base_dirs, CHROMA_PATH, BM25_DIR

def ingest_workspace(workspace_id: str, source_dir: str, chunk_size: int = 600, chunk_overlap: int = 100):
    ws_id = sanitize_workspace_id(workspace_id)
    if not ws_id:
        print("Error: Invalid workspace name provided.")
        return

    ensure_base_dirs()
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Directory not found: {source_dir}")
        return

    # Discover and read .md and .txt
    raw_docs = []
    for file_path in source_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in [".md", ".txt"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        raw_docs.append({"content": content, "source": file_path.name})
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

    if not raw_docs:
        print(f"No text documents found in {source_dir}")
        return

    print(f"Indexing {len(raw_docs)} document(s) into workspace '{ws_id}'...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadatas = []
    chunk_ids = []

    for doc_idx, doc in enumerate(raw_docs):
        sub_chunks = splitter.split_text(doc["content"])
        for chunk_idx, chunk_text in enumerate(sub_chunks):
            cid = f"{ws_id}_{doc_idx}_{chunk_idx}"
            chunks.append(chunk_text)
            metadatas.append({"source": doc["source"], "workspace": ws_id, "id": cid})
            chunk_ids.append(cid)

    # Dense Indexing (ChromaDB)
    embeddings = OllamaEmbeddings(model="qwen2.5-coder:7b", base_url=OLLAMA_URL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    collection_name = f"ws_{ws_id}"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=collection_name)

    vectors = embeddings.embed_documents(chunks)
    collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas, embeddings=vectors)

    # Sparse Indexing (BM25)
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "chunks": chunks,
        "metadatas": metadatas,
        "ids": chunk_ids,
    }
    bm25_path = os.path.join(BM25_DIR, f"{ws_id}.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"Workspace '{ws_id}' indexed successfully: {len(chunks)} chunks.")

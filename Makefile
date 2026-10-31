.PHONY: setup models verify-models up down ollama-pull test test-offline sync ingest ui audit-verify audit-tail users eval clean

# 1. Bootstrap: create the venv and install pure-Python dependencies.
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Run 'make models' once (needs network), add documents under knowledge_base/<workspace>/, create accounts with 'python access.py add-user <name> --admin', then run 'make ui'."

# 2. One-time, network-enabled: download the pinned embedding model and verify its SHA-256
models:
	./venv/bin/python fetch_models.py

# 3. Check the cached model against its pins without any network access
verify-models:
	./venv/bin/python fetch_models.py --verify-only

# 4. Optional PostgreSQL checkpointer (set DB_URI=postgresql://... to use it)
up:
	docker compose up -d

down:
	docker compose down

# 5. Optional: pull the Ollama chat model when using the Ollama backend
ollama-pull:
	ollama pull qwen2.5-coder:7b

# 6. Run the test suite
test:
	./venv/bin/pytest -v

# 7. Same suite with every outbound connection blocked, proving the app needs no network
test-offline:
	RAG_OFFLINE=1 ./venv/bin/pytest -v

# 8. Discover every folder under knowledge_base/ and index it.
#    Idempotent: unchanged workspaces are skipped.
sync:
	./venv/bin/python ingest.py --all

ingest: sync

# 9. Sync, then launch the UI. A failed sync is reported but does not block the UI.
ui:
	-./venv/bin/python ingest.py --all
	./venv/bin/streamlit run ui.py

# 10. Verify the audit hash chain (exits non-zero if any record was altered or removed)
audit-verify:
	./venv/bin/python audit.py verify

# 11. Show the 20 most recent audit events
audit-tail:
	./venv/bin/python audit.py tail -n 20

# 12. List accounts and workspace grants (never password hashes)
users:
	./venv/bin/python access.py list

# 13. Offline evaluation gate: retrieval, answer, and access-control metrics on eval/corpus
eval:
	./venv/bin/python eval.py

# 14. Clean caches and test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage

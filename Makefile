.PHONY: setup up down ollama-pull test sync ingest ui clean

# 1. Bootstrap: create the venv and install pure-Python dependencies.
#    No Docker, Ollama, or model download is required.
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Add documents under knowledge_base/<workspace>/ then run 'make ui'."

# 2. Optional PostgreSQL checkpointer (set DB_URI=postgresql://... to use it)
up:
	docker compose up -d

down:
	docker compose down

# 3. Optional: pull the Ollama chat model when using the Ollama backend
ollama-pull:
	ollama pull qwen2.5-coder:7b

# 4. Run the test suite
test:
	./venv/bin/pytest -v

# 5. Discover every folder under knowledge_base/ and index it.
#    Idempotent: unchanged workspaces are skipped.
sync:
	./venv/bin/python ingest.py --all

ingest: sync

# 6. Sync, then launch the UI. A failed sync is reported but does not block the UI.
ui:
	-./venv/bin/python ingest.py --all
	./venv/bin/streamlit run ui.py

# 7. Clean caches and test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage

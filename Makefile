.PHONY: setup up down test ingest ui clean

# 1. Full bootstrap: installs libraries, spins up DB, and pulls Ollama weights
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	docker compose up -d
	ollama pull qwen2.5-coder:7b
	@echo "Setup complete. Run 'make test' or 'make ui'."

# 2. Start PostgreSQL container
up:
	docker compose up -d

# 3. Stop PostgreSQL container
down:
	docker compose down

# 4. Run full test suite (unit + postgres integration)
test:
	./venv/bin/pytest -v

# 5. Ingest bundled documents
ingest:
	./venv/bin/python ingest.py --workspace accounting --dir ./knowledge_base/accounting
	./venv/bin/python ingest.py --workspace strategy --dir ./knowledge_base/strategy

# 6. Run interactive UI
ui:
	./venv/bin/streamlit run ui.py

# 7. Clean caches and test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage

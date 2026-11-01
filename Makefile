.PHONY: setup rust-build models verify-models up down ollama-pull test test-offline sync ingest ui audit-verify audit-tail audit-feedback audit-anchor retention backup backup-verify users reset-users factory-reset eval eval-judge clean

# 1. Bootstrap: create the venv and install pure-Python dependencies.
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Run 'make models' once (needs network), add documents under knowledge_base/<workspace>/, create accounts with 'python access.py add-user <name> --admin', then run 'make ui'."

# 1b. Build optional native Rust accelerator module (rag_accel) for high-throughput context scanning
rust-build:
	PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 VIRTUAL_ENV=$$PWD/venv ./venv/bin/maturin develop --release -m crates/rag_accel/Cargo.toml

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
	@test -n "$(ADMIN)" || (echo "Indexing is an administrative action. Run: make sync ADMIN=<admin account>" && exit 1)
	./venv/bin/python ingest.py --all --as $(ADMIN)

ingest: sync

# 9. Sync, then launch the UI. A failed sync is reported but does not block the UI.
ui:
	@if [ -n "$(ADMIN)" ]; then ./venv/bin/python ingest.py --all --as $(ADMIN) || echo "Indexing failed; launching the UI anyway."; else echo "Skipping workspace sync. To index knowledge_base/ on launch: make ui ADMIN=<admin account>"; fi
	./venv/bin/streamlit run ui.py

# 10. Verify the audit hash chain (exits non-zero if any record was altered or removed)
audit-verify:
	./venv/bin/python audit.py verify

# 11. Show the 20 most recent audit events
audit-tail:
	./venv/bin/python audit.py tail -n 20

# 15. Anchor the audit head hash. Run on a schedule and copy data/audit_anchors.jsonl off the host.
audit-anchor:
	./venv/bin/python audit.py anchor

# 16. Apply the retention policy (183 days by default) to audit records and idle checkpoint threads
retention:
	./venv/bin/python retention.py

# 17. Consistent, checksummed backup of the audit log, checkpoints, anchors, and policy (keys excluded)
backup:
	./venv/bin/python backup.py create

# 18. Verify a backup: make backup-verify DIR=backups/backup-<timestamp>
backup-verify:
	@test -n "$(DIR)" || (echo "Set DIR=<backup directory>" && exit 1)
	./venv/bin/python backup.py verify $(DIR)

# 12. List accounts and workspace grants (never password hashes)
users:
	./venv/bin/python access.py list

# 13. Offline evaluation gate: retrieval, answer, and access-control metrics on eval/corpus
eval:
	./venv/bin/python eval.py

# Same gates, plus an advisory answer-support check by the local Ollama model
eval-judge:
	./venv/bin/python eval.py --judge

# List answers reviewers rejected, with their reasons
audit-feedback:
	./venv/bin/python audit.py feedback -n 20

# 14. Clean caches and test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .coverage htmlcov eval/out *.log

# 19. Delete all users and reset the access policy
reset-users:
	./venv/bin/python access.py reset-users $(if $(CONFIRM),-y,)

# 20. Complete zero-trace factory reset (wipes policy, data, audit, checkpoints, keys, vectors, uploads, and caches)
factory-reset:
	@if [ "$(CONFIRM)" != "yes" ]; then \
		printf "WARNING: This permanently deletes all user accounts, audit logs, checkpoints, encryption keys, and indexed vectors.\nType 'yes' to proceed: "; \
		read ans; \
		if [ "$$ans" != "yes" ]; then \
			echo "Factory reset cancelled."; \
			exit 1; \
		fi; \
	fi
	rm -f access_policy.json
	rm -rf data
	rm -rf chroma_data
	rm -rf bm25_indices
	rm -rf uploads
	rm -rf backups
	rm -rf eval/out
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .coverage htmlcov *.log
	@echo "Factory reset complete. System restored to clean initial state."

.DEFAULT_GOAL := help
.PHONY: help status setup rust-build models verify-models up down ollama-pull test test-offline workspaces create-workspace delete-workspace sync ingest ui audit-verify audit-tail audit-feedback audit-anchor audit-export retention backup backups backup-verify add-user grant revoke set-clearance set-admin set-default team-mode set-password revoke-sessions delete-user users reset-users factory-reset eval eval-judge eval-calibrate graph clean reviews review approve reject

# 0. Command Reference & System Readiness
help:
	@echo "Local Enterprise RAG Lab — Available Make Commands"
	@echo ""
	@echo "Setup & Health:"
	@echo "  make setup                                     Create venv and install Python dependencies"
	@echo "  make models                                    Download & verify pinned ONNX models (one-time)"
	@echo "  make verify-models                             Verify cached ONNX model SHA-256 hashes offline"
	@echo "  make rust-build                                Compile optional Rust PyO3 accelerator (rag_accel)"
	@echo "  make ollama-pull                               Pull optional Ollama chat + embedding models"
	@echo "  make status                                    Check models, accelerator, LLM, workspaces, and audit health"
	@echo ""
	@echo "RBAC & User Management (Admin-Only after first admin is created):"
	@echo "  make users                                     List accounts, clearance levels, defaults, and grants"
	@echo "  make add-user ACCOUNT=<u> [ADMIN=1|<admin>] [MAKE_ADMIN=1] [CLEARANCE=public|internal|confidential|secret]"
	@echo "  make set-clearance ACCOUNT=<u> CLEARANCE=<lvl> ADMIN=<admin>"
	@echo "  make set-admin ACCOUNT=<u> ADMIN=<admin> [REVOKE=1]"
	@echo "  make set-default PERMS=query,auto_approve ADMIN=<admin>"
	@echo "  make team-mode ADMIN=<admin>                   Require human review by default (needs a reviewer)"
	@echo "  make grant ACCOUNT=<u> WS=<ws|w1,w2|all> [PERMS=query,review,auto_approve] ADMIN=<admin>"
	@echo "  make revoke ACCOUNT=<u> WS=<ws|w1,w2|all> ADMIN=<admin>"
	@echo "  make set-password ACCOUNT=<u> ADMIN=<admin>    Reset a user's password and revoke active tokens"
	@echo "  make revoke-sessions ACCOUNT=<u> ADMIN=<admin> Sign a user out everywhere"
	@echo "  make delete-user ACCOUNT=<u> ADMIN=<admin> [YES=1]"
	@echo "  make reset-users ADMIN=<admin> [CONFIRM=yes]"
	@echo ""
	@echo "Workspaces, Ingestion & UI:"
	@echo "  make workspaces                                List workspaces with file and chunk counts"
	@echo "  make create-workspace WS=<ws> [ADMIN=<admin>]  Initialize a new workspace bundle"
	@echo "  make delete-workspace WS=<ws> [ADMIN=<admin>] [YES=1]"
	@echo "  make sync ADMIN=<u> [WS=<ws>] [DIR=<path>] [FORCE=1]  Index all or one workspace"
	@echo "  make ui [ADMIN=<u>]                            Launch Streamlit web interface"
	@echo ""
	@echo "Human-in-the-Loop (HITL) Compliance Review:"
	@echo "  make reviews                                   List paused threads awaiting review"
	@echo "  make review [THREAD=<id>] [ACCOUNT=<u>]        Interactive review of a paused thread"
	@echo "  make approve THREAD=<id> [ACCOUNT=<u>] [REASON=\"...\"]"
	@echo "  make reject THREAD=<id> [ACCOUNT=<u>] [REASON=\"...\"]"
	@echo ""
	@echo "Audit, Backups & Governance:"
	@echo "  make audit-verify                              Verify HMAC audit hash chain and anchors"
	@echo "  make audit-tail [N=20] [WS=<ws>]               Show most recent audit events"
	@echo "  make audit-feedback [N=20] [WS=<ws>]           Show rejected answers and reviewer reasons"
	@echo "  make audit-anchor                              Record signed audit head hash anchor"
	@echo "  make audit-export [OUT=audit-export.jsonl]     Export decrypted audit trail as JSONL"
	@echo "  make retention [DAYS=183]                      Prune expired audit records and idle checkpoints"
	@echo "  make backup                                    Create checksummed snapshot in backups/"
	@echo "  make backups                                   List existing backup snapshots"
	@echo "  make backup-verify DIR=<backup_dir>            Verify backup checksums, chain, and anchors"
	@echo ""
	@echo "Testing, Evaluation & Maintenance:"
	@echo "  make test                                      Run test suite (pytest -v)"
	@echo "  make test-offline                              Run test suite with all network sockets blocked"
	@echo "  make eval                                      Run offline RAG evaluation gate (eval/golden.json)"
	@echo "  make eval-judge                                Run evaluation with local Ollama LLM-as-a-judge"
	@echo "  make eval-calibrate                            Print retrieval calibration split signals"
	@echo "  make graph                                     Export LangGraph architecture diagram to docs/"
	@echo "  make clean                                     Remove caches and temporary artifacts"
	@echo "  make factory-reset [CONFIRM=yes]               Wipe all data, keys, users, indexes, and logs"

status:
	./venv/bin/python workspace_manager.py status

# 1. Bootstrap: create the venv and install pure-Python dependencies.
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -e . --no-build-isolation
	@echo "Setup complete. Next steps:"
	@echo "  1. 'make models' (one-time download; requires network)"
	@echo "  2. (Optional) 'make rust-build' to compile high-throughput Rust accelerator"
	@echo "  3. 'make add-user ACCOUNT=<name> ADMIN=1' to create your admin account"
	@echo "  4. 'make grant ACCOUNT=<name> WS=all PERMS=query,review' to assign workspace access"
	@echo "  5. 'make sync ADMIN=<name>' to index knowledge_base/ workspaces"
	@echo "  6. 'make ui' to launch the application"

# 1b. Build optional native Rust accelerator module (rag_accel) for high-throughput context scanning
rust-build:
	@./venv/bin/pip show maturin >/dev/null 2>&1 || ./venv/bin/pip install "maturin>=1.7.0"
	PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 VIRTUAL_ENV=$$PWD/venv ./venv/bin/maturin develop --release -m crates/rag_accel/Cargo.toml

# 2. One-time, network-enabled: download the pinned embedding model and verify its SHA-256
models:
	./venv/bin/python fetch_models.py

# 3. Check the cached model against its pins without any network access
verify-models:
	./venv/bin/python fetch_models.py --verify-only

# 4. Optional PostgreSQL checkpointer (set DB_URI=postgresql://... to use it)
up:
	@command -v docker >/dev/null 2>&1 || { echo "Error: Docker is not installed. (Note: Docker is optional; SQLite is used by default with zero setup)."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "Error: Docker daemon is not running. Start Docker or use the default embedded SQLite checkpointer."; exit 1; }
	docker compose up -d

down:
	@command -v docker >/dev/null 2>&1 || exit 0
	docker compose down

# 5. Optional: pull the Ollama chat and embedding models (names resolved by config.py, so .env applies)
ollama-pull:
	ollama pull "$$(./venv/bin/python -c 'import config; print(config.CHAT_MODEL)')"
	ollama pull "$$(./venv/bin/python -c 'import config; print(config.EMBED_MODEL)')"

# 6. Run the test suite
test:
	./venv/bin/pytest -v

# 7. Same suite with every outbound connection blocked, proving the app needs no network
test-offline:
	RAG_OFFLINE=1 ./venv/bin/pytest -v

# 8. Workspace management & document ingestion
workspaces:
	./venv/bin/python workspace_manager.py list

create-workspace:
	@test -n "$(WS)" || (echo "Usage: make create-workspace WS=<workspace> [ADMIN=<admin>]" && exit 1)
	./venv/bin/python workspace_manager.py $(if $(ADMIN),--as $(ADMIN),) create $(WS)

delete-workspace:
	@test -n "$(WS)" || (echo "Usage: make delete-workspace WS=<workspace> [ADMIN=<admin>] [YES=1]" && exit 1)
	./venv/bin/python workspace_manager.py $(if $(ADMIN),--as $(ADMIN),) delete $(WS) $(if $(or $(YES),$(CONFIRM)),-y,)

# Discover every folder under knowledge_base/ and index it, or index a single WS.
# Idempotent: unchanged workspaces are skipped unless FORCE=1.
sync:
	@test -n "$(ADMIN)" || (echo "Indexing is an administrative action. Run: make sync ADMIN=<admin account> [WS=<ws>] [DIR=<path>] [FORCE=1]" && exit 1)
	./venv/bin/python ingest.py $(if $(WS),--workspace $(WS) $(if $(DIR),--dir $(DIR),),--all) $(if $(FORCE),--force,) --as $(ADMIN)

ingest: sync

# 9. Sync, then launch the UI. A failed sync is reported but does not block the UI.
ui:
	@if [ -n "$(ADMIN)" ]; then ./venv/bin/python ingest.py --all --as $(ADMIN) || echo "Indexing failed; launching the UI anyway."; else echo "Skipping workspace sync. To index knowledge_base/ on launch: make ui ADMIN=<admin account>"; fi
	./venv/bin/streamlit run ui.py

# 10. Verify the audit hash chain (exits non-zero if any record was altered or removed)
audit-verify:
	./venv/bin/python audit.py verify

# 11. Show recent audit events (defaults to N=20, optional WS=<workspace>)
audit-tail:
	./venv/bin/python audit.py tail -n $(or $(N),20) $(if $(WS),--workspace $(WS),)

# 15. Anchor the audit head hash. Run on a schedule and copy data/audit_anchors.jsonl off the host.
audit-anchor:
	./venv/bin/python audit.py anchor

# Export decrypted audit trail as JSON lines (to stdout or OUT=<file>)
audit-export:
	./venv/bin/python audit.py export $(if $(OUT),-o $(OUT),)

# 16. Apply the retention policy (183 days by default) to audit records and idle checkpoint threads
retention:
	./venv/bin/python retention.py $(if $(DAYS),--days $(DAYS),)

# 17. Consistent, checksummed backup of the audit log, checkpoints, anchors, and policy (keys excluded)
backup:
	./venv/bin/python backup.py create

backups:
	./venv/bin/python backup.py list

# 18. Verify a backup: make backup-verify DIR=backups/backup-<timestamp>
backup-verify:
	@test -n "$(DIR)" || (echo "Set DIR=<backup directory>" && exit 1)
	./venv/bin/python backup.py verify $(DIR)

# 12. Account & Role-Based Access Control (RBAC) Management
users:
	./venv/bin/python access.py list

add-user:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make add-user ACCOUNT=<username> [ADMIN=1|<admin>] [MAKE_ADMIN=1] [CLEARANCE=public|internal|confidential|secret]" && exit 1)
	@if [ "$(ADMIN)" = "1" ] || [ "$(ADMIN)" = "true" ] || [ "$(ADMIN)" = "yes" ]; then \
		./venv/bin/python access.py $(if $(AS),--as $(AS),) add-user $(or $(ACCOUNT),$(USERNAME)) --admin $(if $(CLEARANCE),--clearance $(CLEARANCE),); \
	else \
		./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) add-user $(or $(ACCOUNT),$(USERNAME)) $(if $(or $(MAKE_ADMIN),$(IS_ADMIN)),--admin,) $(if $(CLEARANCE),--clearance $(CLEARANCE),); \
	fi

grant:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" && test -n "$(WS)" || (echo "Usage: make grant ACCOUNT=<username> WS=<workspace|ws1,ws2|all> [PERMS=query,review,auto_approve] ADMIN=<admin>" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) grant $(or $(ACCOUNT),$(USERNAME)) $(WS) $(if $(PERMS),--perms $(PERMS),)

revoke:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" && test -n "$(WS)" || (echo "Usage: make revoke ACCOUNT=<username> WS=<workspace|ws1,ws2|all> ADMIN=<admin>" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) revoke $(or $(ACCOUNT),$(USERNAME)) $(WS)

set-clearance:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" && test -n "$(CLEARANCE)" || (echo "Usage: make set-clearance ACCOUNT=<username> CLEARANCE=public|internal|confidential|secret ADMIN=<admin>" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) set-clearance $(or $(ACCOUNT),$(USERNAME)) $(CLEARANCE)

set-admin:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make set-admin ACCOUNT=<username> ADMIN=<admin> [REVOKE=1]" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) set-admin $(or $(ACCOUNT),$(USERNAME)) $(if $(REVOKE),--revoke,)

set-default:
	@test -n "$(PERMS)" || (echo "Usage: make set-default PERMS=query,auto_approve ADMIN=<admin>" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) set-default --perms $(PERMS)

team-mode:
	@test -n "$(ADMIN)" || (echo "Usage: make team-mode ADMIN=<admin>" && exit 1)
	./venv/bin/python access.py --as $(ADMIN) team-mode

set-password:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make set-password ACCOUNT=<username> ADMIN=<admin>" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) set-password $(or $(ACCOUNT),$(USERNAME))

revoke-sessions:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make revoke-sessions ACCOUNT=<username> ADMIN=<admin>" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) revoke-sessions $(or $(ACCOUNT),$(USERNAME))

delete-user:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make delete-user ACCOUNT=<username> ADMIN=<admin> [YES=1]" && exit 1)
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) delete-user $(or $(ACCOUNT),$(USERNAME)) $(if $(or $(YES),$(CONFIRM)),-y,)

# 12. Human-in-the-Loop compliance review CLI targets
# List all threads awaiting human review
reviews:
	./venv/bin/python approve.py --list

# Interactive review of a paused thread (auto-selects first pending thread if THREAD not set)
review:
	./venv/bin/python approve.py $(if $(THREAD),--thread $(THREAD),) $(if $(ACCOUNT),--approver $(ACCOUNT),)

# Direct approval of a candidate answer
approve:
	@test -n "$(THREAD)" || (echo "Usage: make approve THREAD=<thread_id> [ACCOUNT=<reviewer>] [REASON=\"reason\"]" && exit 1)
	./venv/bin/python approve.py --thread $(THREAD) --action approved $(if $(ACCOUNT),--approver $(ACCOUNT),) $(if $(REASON),--reason "$(REASON)",)

# Direct rejection of a candidate answer
reject:
	@test -n "$(THREAD)" || (echo "Usage: make reject THREAD=<thread_id> [ACCOUNT=<reviewer>] [REASON=\"reason\"]" && exit 1)
	./venv/bin/python approve.py --thread $(THREAD) --action rejected $(if $(ACCOUNT),--approver $(ACCOUNT),) $(if $(REASON),--reason "$(REASON)",)

# 13. Offline evaluation gate: retrieval, answer, and access-control metrics on eval/corpus
eval:
	./venv/bin/python eval.py

# Same gates, plus an advisory answer-support check by the local Ollama model
eval-judge:
	./venv/bin/python eval.py --judge

# Print retrieval similarity and coverage signals for the calibration split
eval-calibrate:
	./venv/bin/python eval.py --calibrate

# Export LangGraph topology to docs/architecture.mmd (and .png if mmdc is installed)
graph:
	./venv/bin/python export_graph.py

# List answers reviewers rejected, with their reasons (defaults to N=20, optional WS=<workspace>)
audit-feedback:
	./venv/bin/python audit.py feedback -n $(or $(N),20) $(if $(WS),--workspace $(WS),)

# 14. Clean caches and test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .coverage htmlcov eval/out *.log *.egg-info

# 19. Delete all users and reset the access policy
reset-users:
	./venv/bin/python access.py $(if $(ADMIN),--as $(ADMIN),) reset-users $(if $(or $(CONFIRM),$(YES)),-y,)

# 20. Complete zero-trace factory reset (wipes policy, data, audit, checkpoints, keys, vectors, uploads, and caches)
factory-reset:
	@if [ "$(CONFIRM)" != "yes" ] && [ "$(YES)" != "1" ] && [ "$(YES)" != "yes" ]; then \
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
	rm -rf .pytest_cache .coverage htmlcov *.log *.egg-info
	@echo "Factory reset complete. System restored to clean initial state."

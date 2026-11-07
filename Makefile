.DEFAULT_GOAL := help
.PHONY: help status setup rust-build models verify-models calibrate-models up down ollama-pull test test-offline lint typecheck coverage workspaces create-workspace delete-workspace sync ingest ui audit-verify audit-tail audit-feedback audit-harvest audit-anchor audit-export retention backup backups backup-verify add-user grant revoke set-clearance set-admin set-role set-default team-mode groups create-group delete-group add-to-group remove-from-group grant-group set-password revoke-sessions delete-user users reset-users factory-reset recover db-password scale-up scale-down lan-tls lan-tls-down lan-tls-ca migrate-keys totp-enroll totp-recovery totp-disable eval eval-judge eval-llm eval-calibrate graph clean reviews review approve reject

# 0. Command Reference & System Readiness
help:
	@echo "Local Enterprise RAG Lab — Available Make Commands"
	@echo ""
	@echo "Setup & Health:"
	@echo "  make setup                                     Create venv and install Python dependencies"
	@echo "  make models [PROFILE=multilingual|multilingual-large|en]  Download & verify the profile's pinned models"
	@echo "  make verify-models [PROFILE=...]               Verify cached ONNX model SHA-256 hashes offline"
	@echo "  make calibrate-models [PROFILE=...]            Suggest the profile's thresholds from the eval calibration split"
	@echo "  make rust-build                                Compile the native Rust accelerator (rag_accel) from source"
	@echo "  make ollama-pull                               Pull the optional Ollama chat model"
	@echo "  make up / make down                            Start / stop the optional PostgreSQL checkpointer (Docker); 'up' creates .env with a random password"
	@echo "  make scale-up SERVICE=<qdrant|pgvector|vllm>   Start an optional scaling service (docs/scaling.md); make up never does"
	@echo "  make scale-down SERVICE=<qdrant|pgvector|vllm> Stop it (its container and data volume are kept)"
	@echo "  make lan-tls                                   Optional HTTPS for the UI on the LAN (Caddy profile, docs/lan-tls.md)"
	@echo "  make lan-tls-down                              Stop the LAN TLS proxy"
	@echo "  make lan-tls-ca                                Export Caddy's local root certificate for client devices"
	@echo "  make db-password                               Give an existing PostgreSQL volume the password in .env (once, after upgrading)"
	@echo "  make status                                    Check models, accelerator, LLM, keys, workspaces, and audit health"
	@echo "  make migrate-keys [REVERSE=1]                  Move the keys into the OS keychain (REVERSE=1: back into their files)"
	@echo ""
	@echo "RBAC & User Management (Admin-Only after first admin is created):"
	@echo "  Roles: owner, admin (+administration) | reviewer (query,review) | trusted (query,auto_approve) | member (query) | none"
	@echo "  Resolution: account grant > group grants (any denial wins) > account role > default role"
	@echo "  make users                                     List accounts, roles, groups, clearance levels, and grants"
	@echo "  make add-user ACCOUNT=<u> [ADMIN=1|<admin>] [MAKE_ADMIN=1] [ROLE=<role>] [CLEARANCE=public|internal|confidential|secret]"
	@echo "  make set-role ACCOUNT=<u> ROLE=<role|default> ADMIN=<admin>  Only an owner assigns or removes 'owner'"
	@echo "  make set-clearance ACCOUNT=<u> CLEARANCE=<lvl> ADMIN=<admin>"
	@echo "  make set-admin ACCOUNT=<u> ADMIN=<admin> [REVOKE=1]  Shortcut for ROLE=admin / ROLE=default"
	@echo "  make set-default ROLE=<member|trusted|reviewer|none> ADMIN=<admin>  (or PERMS=<list>)"
	@echo "  make team-mode ADMIN=<admin>                   Default role trusted -> member (needs a reviewer)"
	@echo "  make totp-enroll ADMIN=<me>                    Turn on two-factor sign-in for yourself (owners and admins)"
	@echo "  make totp-recovery ADMIN=<me>                  Replace your recovery codes"
	@echo "  make totp-disable ADMIN=<me> [ACCOUNT=<u>]     Turn it off (yours; another account's: owner only)"
	@echo "  make recover ACCOUNT=<u> [RESET_TOTP=1] [RESET_PASSWORD=1]  Operator break-glass at this machine for a locked-out account"
	@echo "  make groups                                    List groups, members, and workspace roles"
	@echo "  make create-group GROUP=<g> ADMIN=<admin>"
	@echo "  make delete-group GROUP=<g> ADMIN=<admin>      Members keep their own access"
	@echo "  make add-to-group ACCOUNT=<u> GROUP=<g> ADMIN=<admin>"
	@echo "  make remove-from-group ACCOUNT=<u> GROUP=<g> ADMIN=<admin>"
	@echo "  make grant-group GROUP=<g> WS=<ws|w1,w2|all> ROLE=<reviewer|trusted|member|none|default> ADMIN=<admin>"
	@echo "  make grant ACCOUNT=<u> WS=<ws|w1,w2|all> [PERMS=query,review,auto_approve | ROLE=<role>] ADMIN=<admin>"
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
	@echo "  make audit-harvest [N=50] [WS=<ws>] [OUT=path] Export rejected answers as candidate eval cases"
	@echo "  make audit-anchor                              Record signed audit head hash anchor (+ AUDIT_ANCHOR_COPY_DIR copy)"
	@echo "  make audit-export [OUT=audit-export.jsonl]     Export decrypted audit trail as JSONL"
	@echo "  make retention [DAYS=183]                      Prune expired audit records and idle checkpoints"
	@echo "  make backup                                    Create checksummed snapshot in backups/"
	@echo "  make backups                                   List existing backup snapshots"
	@echo "  make backup-verify DIR=<backup_dir>            Verify backup checksums, chain, and anchors"
	@echo ""
	@echo "Testing, Evaluation & Maintenance:"
	@echo "  make test                                      Run test suite (pytest -v)"
	@echo "  make test-offline                              Run test suite with all network sockets blocked"
	@echo "  make lint                                      Lint with ruff (pinned rule set)"
	@echo "  make typecheck                                 Type-check the rag_lab package with mypy"
	@echo "  make coverage                                  Run the suite with a per-module coverage report"
	@echo "  make eval                                      Run offline RAG evaluation gate (eval/golden.json)"
	@echo "  make eval-judge                                Run evaluation with local Ollama LLM-as-a-judge"
	@echo "  make eval-llm [SUITE=en]                       Evaluate answers from the local Ollama model, judge gated"
	@echo "  make eval-calibrate                            Print retrieval calibration split signals"
	@echo "  make graph                                     Export LangGraph architecture diagram to docs/"
	@echo "  make clean                                     Remove caches and temporary artifacts"
	@echo "  make factory-reset [CONFIRM=yes]               Wipe all data, keys, users, indexes, and logs"

status:
	./venv/bin/python -m rag_lab.storage.workspace_manager status

# 1. Bootstrap: create the venv and install pure-Python dependencies.
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -e . --no-build-isolation
	./venv/bin/python -m rag_lab.ops.install_accel
	@echo "Setup complete. Next steps:"
	@echo "  1. 'make models' (one-time download; requires network)"
	@echo "  2. 'make status' shows whether the native accelerator is active (wheelhouse/ wheel, else cargo, else Python fallback)"
	@echo "  3. 'make add-user ACCOUNT=<name> ADMIN=1' to create your admin account"
	@echo "  4. (optional) 'make add-user ACCOUNT=<colleague> ADMIN=<name>' adds accounts; you choose a role for each"
	@echo "  5. 'make sync ADMIN=<name>' to index knowledge_base/ workspaces"
	@echo "  6. 'make ui' to launch the application"

# 1b. Build the native Rust accelerator module (rag_accel) from source. 'make setup' installs a prebuilt wheel from
# wheelhouse/ first, then falls back to this when cargo exists, then to pure Python (see rag_lab/ops/install_accel.py).
rust-build:
	@./venv/bin/pip show maturin >/dev/null 2>&1 || ./venv/bin/pip install "maturin>=1.7.0"
	VIRTUAL_ENV=$$PWD/venv ./venv/bin/maturin develop --release --locked -m crates/rag_accel/Cargo.toml

# 2. One-time, network-enabled: download the pinned embedding model and verify its SHA-256
models:
	$(if $(PROFILE),MODEL_PROFILE=$(PROFILE) ,)./venv/bin/python -m rag_lab.ops.fetch_models

# 3. Check the cached model against its pins without any network access
verify-models:
	$(if $(PROFILE),MODEL_PROFILE=$(PROFILE) ,)./venv/bin/python -m rag_lab.ops.fetch_models --verify-only

# Suggest the model-dependent thresholds for a profile from the eval calibration split (runs make eval first)
calibrate-models:
	$(if $(PROFILE),MODEL_PROFILE=$(PROFILE) ,)./venv/bin/python -m rag_lab.ops.eval > /dev/null || true
	$(if $(PROFILE),MODEL_PROFILE=$(PROFILE) ,)./venv/bin/python -m rag_lab.ops.calibrate

# 4. Optional PostgreSQL checkpointer (set DB_URI=postgresql://... to use it)
up:
	@command -v docker >/dev/null 2>&1 || { echo "Error: Docker is not installed. (Note: Docker is optional; SQLite is used by default with zero setup)."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "Error: Docker daemon is not running. Start Docker or use the default embedded SQLite checkpointer."; exit 1; }
	@$(if $(wildcard venv/bin/python),./venv/bin/python,python3) -m rag_lab.ops.init_env
	docker compose up -d

# Optional scaling services (compose profiles qdrant, pgvector, vllm). Nothing here is needed on one laptop. The
# service only starts: turn it on in .env (VECTOR_STORE=qdrant, VECTOR_STORE=pgvector, LLM_PROVIDER=openai).
scale-up:
	@case "$(SERVICE)" in qdrant|pgvector|vllm) ;; *) echo "Usage: make scale-up SERVICE=qdrant|pgvector|vllm"; exit 1;; esac
	@docker info >/dev/null 2>&1 || { echo "Error: Docker daemon is not running."; exit 1; }
	@$(if $(wildcard venv/bin/python),./venv/bin/python,python3) -m rag_lab.ops.init_env
	docker compose --profile $(SERVICE) up -d $(SERVICE)

scale-down:
	@case "$(SERVICE)" in qdrant|pgvector|vllm) ;; *) echo "Usage: make scale-down SERVICE=qdrant|pgvector|vllm"; exit 1;; esac
	docker compose --profile $(SERVICE) stop $(SERVICE)

# Optional LAN TLS (compose profile "caddy"): Caddy terminates HTTPS with its own local CA and proxies to the
# Streamlit UI on the host. Never started by `make up`. Set RAG_LAN_HOST and RAG_HTTPS_PORT in .env.
lan-tls:
	@docker info >/dev/null 2>&1 || { echo "Error: Docker daemon is not running."; exit 1; }
	docker compose --profile caddy up -d caddy

lan-tls-down:
	docker compose --profile caddy rm -sf caddy

# Copy Caddy's local root certificate out of the container, to install on the devices that open the UI.
lan-tls-ca:
	@mkdir -p data && docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt data/caddy-root.crt && echo "Wrote data/caddy-root.crt (public certificate; install it as a trusted root on your devices)."

# Rotate the database role's password to the one in .env. POSTGRES_PASSWORD only applies when the volume is
# first created, so a volume from an earlier version keeps its old password until this runs. It connects over
# the container's local socket (no old password needed) and the password never appears on the host.
db-password:
	@docker compose exec -T postgres sh -c 'printf "ALTER USER \"%s\" PASSWORD '"'"'%s'"'"';\n" "$$POSTGRES_USER" "$$POSTGRES_PASSWORD" | psql -q -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' && echo "The database role now uses the password in .env."

down:
	@command -v docker >/dev/null 2>&1 || exit 0
	docker compose down

# 5. Optional: pull the Ollama chat model (name resolved by config.py, so .env applies). Embeddings are in-process ONNX.
ollama-pull:
	ollama pull "$$(./venv/bin/python -c 'from rag_lab import config; print(config.settings.chat_model)')"

# 6. Run the test suite
test:
	./venv/bin/pytest -v

# 7. Same suite with every outbound connection blocked, proving the app needs no network
test-offline:
	RAG_OFFLINE=1 ./venv/bin/pytest -v

# Lint gate (rule set pinned in pyproject.toml) and the coverage report CI prints
lint:
	./venv/bin/ruff check .

typecheck:
	./venv/bin/mypy rag_lab

coverage:
	./venv/bin/pytest -q --cov=. --cov-report=term-missing:skip-covered

# 8. Workspace management & document ingestion
workspaces:
	./venv/bin/python -m rag_lab.storage.workspace_manager list

create-workspace:
	@test -n "$(WS)" || (echo "Usage: make create-workspace WS=<workspace> [ADMIN=<admin>]" && exit 1)
	./venv/bin/python -m rag_lab.storage.workspace_manager $(if $(ADMIN),--as $(ADMIN),) create $(WS)

delete-workspace:
	@test -n "$(WS)" || (echo "Usage: make delete-workspace WS=<workspace> [ADMIN=<admin>] [YES=1]" && exit 1)
	./venv/bin/python -m rag_lab.storage.workspace_manager $(if $(ADMIN),--as $(ADMIN),) delete $(WS) $(if $(or $(YES),$(CONFIRM)),-y,)

# Discover every folder under knowledge_base/ and index it, or index a single WS.
# Idempotent: unchanged workspaces are skipped unless FORCE=1.
sync:
	@test -n "$(ADMIN)" || (echo "Indexing is an administrative action. Run: make sync ADMIN=<admin account> [WS=<ws>] [DIR=<path>] [FORCE=1]" && exit 1)
	./venv/bin/python -m rag_lab.ingestion.ingest $(if $(WS),--workspace $(WS) $(if $(DIR),--dir $(DIR),),--all) $(if $(FORCE),--force,) --as $(ADMIN)

ingest: sync

# 9. Sync, then launch the UI headlessly. Binds to 127.0.0.1 by default; use LAN=1 to expose to the network.
ui:
	@if [ -n "$(ADMIN)" ]; then ./venv/bin/python -m rag_lab.ingestion.ingest --all --as $(ADMIN) || echo "Indexing failed; launching the UI anyway."; else echo "Skipping workspace sync. To index knowledge_base/ on launch: make ui ADMIN=<admin account>"; fi
	STREAMLIT_SERVER_HEADLESS=$${STREAMLIT_SERVER_HEADLESS:-true} STREAMLIT_SERVER_ADDRESS=$${STREAMLIT_SERVER_ADDRESS:-$(if $(filter 1 true yes,$(LAN)),0.0.0.0,127.0.0.1)} ./venv/bin/streamlit run ui.py

# 10. Verify the audit hash chain (exits non-zero if any record was altered or removed)
audit-verify:
	./venv/bin/python -m rag_lab.security.audit verify

# 11. Show recent audit events (defaults to N=20, optional WS=<workspace>)
audit-tail:
	./venv/bin/python -m rag_lab.security.audit tail -n $(or $(N),20) $(if $(WS),--workspace $(WS),)

# 15. Anchor the audit head hash. Run on a schedule. With AUDIT_ANCHOR_COPY_DIR the anchors are also written to a
# second location (another disk or share); the command exits non-zero when that copy cannot be updated.
audit-anchor:
	./venv/bin/python -m rag_lab.security.audit anchor

# Export decrypted audit trail as JSON lines (to stdout or OUT=<file>)
audit-export:
	./venv/bin/python -m rag_lab.security.audit export $(if $(OUT),-o $(OUT),)

# 16. Apply the retention policy (183 days by default) to audit records and idle checkpoint threads
retention:
	./venv/bin/python -m rag_lab.storage.retention $(if $(DAYS),--days $(DAYS),)

# 17. Consistent, checksummed backup of the audit log, checkpoints, anchors, and policy (keys excluded)
backup:
	./venv/bin/python -m rag_lab.storage.backup create

backups:
	./venv/bin/python -m rag_lab.storage.backup list

# 18. Verify a backup: make backup-verify DIR=backups/backup-<timestamp>
backup-verify:
	@test -n "$(DIR)" || (echo "Set DIR=<backup directory>" && exit 1)
	./venv/bin/python -m rag_lab.storage.backup verify $(DIR)

# 11b. Keys: move the access and data keys into the OS keychain, or back into their owner-only files.
# Keys in the keychain are not covered by `make backup`: keep a copy in custody (REVERSE=1 restores the files).
migrate-keys:
	./venv/bin/python -m rag_lab.security.keystore migrate $(if $(REVERSE),--reverse,)

# 12. Account & Role-Based Access Control (RBAC) Management
users:
	./venv/bin/python -m rag_lab.security.access list

add-user:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make add-user ACCOUNT=<username> [ADMIN=1|<admin>] [MAKE_ADMIN=1] [CLEARANCE=public|internal|confidential|secret]" && exit 1)
	@if [ "$(ADMIN)" = "1" ] || [ "$(ADMIN)" = "true" ] || [ "$(ADMIN)" = "yes" ]; then \
		./venv/bin/python -m rag_lab.security.access $(if $(AS),--as $(AS),) add-user $(or $(ACCOUNT),$(USERNAME)) --admin $(if $(ROLE),--role $(ROLE),) $(if $(CLEARANCE),--clearance $(CLEARANCE),); \
	else \
		./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) add-user $(or $(ACCOUNT),$(USERNAME)) $(if $(or $(MAKE_ADMIN),$(IS_ADMIN)),--admin,) $(if $(ROLE),--role $(ROLE),) $(if $(CLEARANCE),--clearance $(CLEARANCE),); \
	fi

grant:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" && test -n "$(WS)" || (echo "Usage: make grant ACCOUNT=<username> WS=<workspace|ws1,ws2|all> [PERMS=query,review,auto_approve] ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) grant $(or $(ACCOUNT),$(USERNAME)) $(WS) $(if $(PERMS),--perms $(PERMS),) $(if $(ROLE),--role $(ROLE),)

revoke:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" && test -n "$(WS)" || (echo "Usage: make revoke ACCOUNT=<username> WS=<workspace|ws1,ws2|all> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) revoke $(or $(ACCOUNT),$(USERNAME)) $(WS)

set-clearance:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" && test -n "$(CLEARANCE)" || (echo "Usage: make set-clearance ACCOUNT=<username> CLEARANCE=public|internal|confidential|secret ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) set-clearance $(or $(ACCOUNT),$(USERNAME)) $(CLEARANCE)

set-admin:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make set-admin ACCOUNT=<username> ADMIN=<admin> [REVOKE=1]" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) set-admin $(or $(ACCOUNT),$(USERNAME)) $(if $(REVOKE),--revoke,)

set-role:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" && test -n "$(ROLE)" || (echo "Usage: make set-role ACCOUNT=<username> ROLE=<owner|admin|reviewer|trusted|member|none|default> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) set-role $(or $(ACCOUNT),$(USERNAME)) $(ROLE)

set-default:
	@test -n "$(or $(ROLE),$(PERMS))" || (echo "Usage: make set-default ROLE=<member|trusted|reviewer|none> ADMIN=<admin>  (or PERMS=query,auto_approve)" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) set-default $(if $(ROLE),--role $(ROLE),--perms $(PERMS))

# Break-glass for an account that cannot sign in (lost authenticator and codes, forgotten password), including a sole
# owner. No sign-in: run it at the machine, from a terminal. See README "When someone is locked out".
recover:
	@test -n "$(ACCOUNT)" && { test -n "$(RESET_TOTP)" || test -n "$(RESET_PASSWORD)"; } || (echo "Usage: make recover ACCOUNT=<account> [RESET_TOTP=1] [RESET_PASSWORD=1]" && exit 1)
	./venv/bin/python -m rag_lab.security.access recover $(ACCOUNT) $(if $(RESET_TOTP),--totp,) $(if $(RESET_PASSWORD),--password,)

totp-enroll:
	@test -n "$(ADMIN)" || (echo "Usage: make totp-enroll ADMIN=<your account>" && exit 1)
	./venv/bin/python -m rag_lab.security.access --as $(ADMIN) totp-enroll

totp-recovery:
	@test -n "$(ADMIN)" || (echo "Usage: make totp-recovery ADMIN=<your account>" && exit 1)
	./venv/bin/python -m rag_lab.security.access --as $(ADMIN) totp-recovery

totp-disable:
	@test -n "$(ADMIN)" || (echo "Usage: make totp-disable ADMIN=<your account> [ACCOUNT=<account to reset, owner only>]" && exit 1)
	./venv/bin/python -m rag_lab.security.access --as $(ADMIN) totp-disable $(ACCOUNT)

groups:
	./venv/bin/python -m rag_lab.security.access groups

create-group:
	@test -n "$(GROUP)" || (echo "Usage: make create-group GROUP=<group> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) create-group $(GROUP)

delete-group:
	@test -n "$(GROUP)" || (echo "Usage: make delete-group GROUP=<group> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) delete-group $(GROUP)

add-to-group:
	@test -n "$(ACCOUNT)" && test -n "$(GROUP)" || (echo "Usage: make add-to-group ACCOUNT=<username> GROUP=<group> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) add-to-group $(ACCOUNT) $(GROUP)

remove-from-group:
	@test -n "$(ACCOUNT)" && test -n "$(GROUP)" || (echo "Usage: make remove-from-group ACCOUNT=<username> GROUP=<group> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) remove-from-group $(ACCOUNT) $(GROUP)

grant-group:
	@test -n "$(GROUP)" && test -n "$(WS)" && test -n "$(ROLE)" || (echo "Usage: make grant-group GROUP=<group> WS=<ws|w1,w2|all> ROLE=<reviewer|trusted|member|none|default> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) grant-group $(GROUP) $(WS) --role $(ROLE)

team-mode:
	@test -n "$(ADMIN)" || (echo "Usage: make team-mode ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access --as $(ADMIN) team-mode

set-password:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make set-password ACCOUNT=<username> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) set-password $(or $(ACCOUNT),$(USERNAME))

revoke-sessions:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make revoke-sessions ACCOUNT=<username> ADMIN=<admin>" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) revoke-sessions $(or $(ACCOUNT),$(USERNAME))

delete-user:
	@test -n "$(or $(ACCOUNT),$(USERNAME))" || (echo "Usage: make delete-user ACCOUNT=<username> ADMIN=<admin> [YES=1]" && exit 1)
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) delete-user $(or $(ACCOUNT),$(USERNAME)) $(if $(or $(YES),$(CONFIRM)),-y,)

# 12. Human-in-the-Loop compliance review CLI targets
# List all threads awaiting human review
reviews:
	./venv/bin/python -m rag_lab.ops.approve --list

# Interactive review of a paused thread (auto-selects first pending thread if THREAD not set)
review:
	./venv/bin/python -m rag_lab.ops.approve $(if $(THREAD),--thread $(THREAD),) $(if $(ACCOUNT),--approver $(ACCOUNT),)

# Direct approval of a candidate answer
approve:
	@test -n "$(THREAD)" || (echo "Usage: make approve THREAD=<thread_id> [ACCOUNT=<reviewer>] [REASON=\"reason\"]" && exit 1)
	./venv/bin/python -m rag_lab.ops.approve --thread $(THREAD) --action approved $(if $(ACCOUNT),--approver $(ACCOUNT),) $(if $(REASON),--reason "$(REASON)",)

# Direct rejection of a candidate answer
reject:
	@test -n "$(THREAD)" || (echo "Usage: make reject THREAD=<thread_id> [ACCOUNT=<reviewer>] [REASON=\"reason\"]" && exit 1)
	./venv/bin/python -m rag_lab.ops.approve --thread $(THREAD) --action rejected $(if $(ACCOUNT),--approver $(ACCOUNT),) $(if $(REASON),--reason "$(REASON)",)

# 13. Offline evaluation gate: retrieval, answer, and access-control metrics on eval/corpus
eval:
	./venv/bin/python -m rag_lab.ops.eval

# Same gates, plus an advisory answer-support check by the local Ollama model
eval-judge:
	./venv/bin/python -m rag_lab.ops.eval --judge

# Answer with the local Ollama model and gate the LLM judge too (needs 'ollama serve'; SUITE=en limits it)
eval-llm:
	./venv/bin/python -m rag_lab.ops.eval --llm $(if $(SUITE),--suite $(SUITE),)

# Print retrieval similarity and coverage signals for the calibration split
eval-calibrate:
	./venv/bin/python -m rag_lab.ops.eval --calibrate

# Export LangGraph topology to docs/architecture.mmd (and .png if mmdc is installed)
graph:
	./venv/bin/python -m rag_lab.ops.export_graph

# List answers reviewers rejected, with their reasons (defaults to N=20, optional WS=<workspace>)
audit-feedback:
	./venv/bin/python -m rag_lab.security.audit feedback -n $(or $(N),20) $(if $(WS),--workspace $(WS),)

# Export reviewer rejections as candidate golden eval cases (feedback loop)
audit-harvest:
	./venv/bin/python -m rag_lab.security.audit harvest -n $(or $(N),50) $(if $(WS),--workspace $(WS),) $(if $(OUT),--output $(OUT),)

# 14. Clean caches and test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .coverage htmlcov eval/out *.log *.egg-info

# 19. Delete all users and reset the access policy
reset-users:
	./venv/bin/python -m rag_lab.security.access $(if $(ADMIN),--as $(ADMIN),) reset-users $(if $(or $(CONFIRM),$(YES)),-y,)

# 20. Complete zero-trace factory reset (wipes policy, data, audit, checkpoints, keys, vectors, uploads, and caches)
factory-reset:
	@if [ "$(CONFIRM)" != "yes" ] && [ "$(YES)" != "1" ] && [ "$(YES)" != "yes" ]; then \
		printf "WARNING: This permanently deletes all user accounts, audit logs, checkpoints, encryption keys, and indexed vectors, including whatever the configured Qdrant, pgvector or shared-cache server holds for this install.\nType 'yes' to proceed: "; \
		read ans; \
		if [ "$$ans" != "yes" ]; then \
			echo "Factory reset cancelled."; \
			exit 1; \
		fi; \
	fi
	./venv/bin/python -m rag_lab.ops.purge_external || echo "Warning: a configured external store was NOT purged (see above); the documents or cached answers are still on that server."
	./venv/bin/python -m rag_lab.security.keystore forget || echo "Warning: keychain entries could not be removed; delete the local-rag-lab entries in your keychain app by hand."
	rm -f access_policy.json access_policy.json.*.bak access_policy.json.lock
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

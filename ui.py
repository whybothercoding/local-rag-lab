import atexit
import streamlit as st
import json
import time
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from access import (
    CLEARANCE_LEVELS,
    PolicyIntegrityError,
    add_user,
    allowed_workspaces,
    authorize,
    clearance_covers,
    describe_review_mode,
    effective_permissions,
    get_default_permissions,
    get_user_clearance,
    is_admin,
    is_idle_expired,
    load_policy,
    reset_grant,
    review_mode_report,
    revoke_sessions,
    set_admin,
    set_default_permissions,
    set_grant,
    set_user_clearance,
    set_user_permissions_everywhere,
    sign_in,
    verify_token,
)
from app import workflow, GraphState
from audit import export_events, list_events, record, verify_anchors, verify_chain
from approve import list_pending_threads
import checkpointer as checkpointer_module
from checkpointer import close_checkpointer, fetch_checkpoint_threads, open_checkpointer, shared_pool
from config import MAX_SOURCE_BYTES, UPLOADS_DIR
from ingest import ingest_workspace
from parsers import SUPPORTED_EXTENSIONS
from redaction import redact_pii
from workspace_manager import (
    list_workspaces,
    create_workspace,
    delete_workspace,
    get_workspace_stats,
    sanitize_workspace_id,
    staged_path,
)
from pathlib import Path

st.set_page_config(page_title="Local Enterprise Knowledge Hub", layout="wide")


def render_citations(citations):
    """Renders visual expandable cards showing retrieved document sources."""
    if not citations:
        return
    with st.expander("Retrieved Source Citations", expanded=False):
        for idx, citation in enumerate(citations, 1):
            st.markdown(f"**[{idx}]** `{citation}`")


def render_degraded(notes) -> None:
    """Shows what went wrong when an answer was produced in a reduced mode, so no one reads it as a full answer."""
    for note in notes or []:
        st.warning(f"Degraded answer: {note}")


def _sign_out(notice: str = "", revoke: bool = False) -> None:
    current_user = st.session_state.get("user")
    if revoke and current_user:
        try:
            revoke_sessions(current_user, actor=current_user)
        except Exception:
            pass
    for key in ("user", "token", "last_seen", "messages", "thread_id", "current_ws", "queue_pick"):
        st.session_state.pop(key, None)
    if notice:
        st.session_state.notice = notice


def login_gate() -> None:
    """Stops the script until a user holds a valid, unexpired session."""
    if st.session_state.get("user"):
        if not verify_token(st.session_state.get("token", "")):
            _sign_out("Your session has expired. Please sign in again.")
        elif is_idle_expired(st.session_state.last_seen, time.time()):
            _sign_out("You were signed out after 30 minutes of inactivity.")
        else:
            st.session_state.last_seen = time.time()
            return

    st.title("Local Enterprise Knowledge Hub")
    if notice := st.session_state.pop("notice", ""):
        st.info(notice)
    try:
        has_accounts = bool(load_policy()["users"])
    except PolicyIntegrityError as exc:
        st.error(str(exc))
        st.stop()
    if not has_accounts:
        st.error("No accounts exist yet. Create one with: make add-user ACCOUNT=<name> ADMIN=1")
        st.stop()

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        result = sign_in(username, password)
        if result.ok:
            st.session_state.user = result.username
            st.session_state.token = result.token
            st.session_state.last_seen = time.time()
            st.rerun()
        message = result.describe()
        if result.reason == "invalid":
            message += " Forgot the password? Ask an administrator to reset it (make set-password ACCOUNT=<name>)."
        st.error(message)
    st.stop()


def pending_review_panel(engine, thread_id: str, ws_id: str, user: str) -> bool:
    """Show the compliance gate for one paused thread. Returns True while that thread awaits a decision."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = engine.get_state(config)
    if state.next != ("human_approval",):
        return False

    requester = state.values.get("principal", "")
    req_clearance = state.values.get("principal_clearance") or get_user_clearance(requester)
    user_clearance = get_user_clearance(user)
    if user != requester and not clearance_covers(user_clearance, req_clearance):
        st.error(
            f"Clearance restriction: this thread requires '{req_clearance}' clearance "
            f"(your account holds '{user_clearance}')."
        )
        return True

    interrupt_payload = {}
    for task in state.tasks or []:
        if getattr(task, "interrupts", None):
            interrupt_payload = task.interrupts[0].value or {}
            break
    review_reasons = interrupt_payload.get("review_reasons") or []

    st.warning("Candidate answer paused at the compliance approval gate.")
    if review_reasons:
        st.caption(f"Gate triggered by: `{', '.join(review_reasons)}`")
    display_q = state.values.get("original_question") or state.values.get("sanitized_question", "")
    st.caption(f"Question ({req_clearance}): {display_q}")
    st.text_area("Candidate Output", value=state.values.get("generation", ""), height=130,
                 disabled=True, key=f"candidate-{thread_id}")
    render_citations(state.values.get("citations", []))
    render_degraded(state.values.get("degraded", []))

    if not authorize(user, ws_id, "review") or user == requester or not clearance_covers(user_clearance, req_clearance):
        if user == requester:
            st.info("Awaiting approval from another reviewer (Separation of Duties). An authorized reviewer can approve here or via CLI: `make review`.")
        else:
            st.info("Awaiting a decision from a reviewer with access to this workspace.")
        return True

    reason = st.text_input("Reason (optional, recorded in the audit log)", key=f"reason-{thread_id}").strip()
    st.caption(f"Your decision will be recorded under **{user}**.")
    col1, col2 = st.columns(2)
    approve_clicked = col1.button("Approve Output", key=f"approve-{thread_id}")
    reject_clicked = col2.button("Reject Output", key=f"reject-{thread_id}")

    if approve_clicked or reject_clicked:
        decision = "approved" if approve_clicked else "rejected"
        resume = {"decision": decision, "approver_token": st.session_state.token, "reason": reason}
        for _ in engine.stream(Command(resume=resume), config=config):
            pass
        if thread_id == st.session_state.thread_id:
            final_state = engine.get_state(config)
            if decision == "approved":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_state.values.get("generation", ""),
                    "citations": final_state.values.get("citations", []),
                    "degraded": final_state.values.get("degraded", []),
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "[Rejected by reviewer]",
                    "citations": [],
                })
        st.rerun()
    return True


def review_queue(engine, ws_id: str, user: str, own_thread: str) -> None:
    """Reviewers see other users' paused queries for the active workspace within their clearance."""
    if not authorize(user, ws_id, "review"):
        return
    viewer_clr = get_user_clearance(user)
    pending = [
        tid for tid in list_pending_threads(engine, viewer_clearance=viewer_clr, prefix=f"ui-{ws_id}-")
        if tid.startswith(f"ui-{ws_id}-") and tid != own_thread
    ]
    if not pending:
        return
    with st.expander(f"Review queue: {len(pending)} awaiting decision", expanded=True):
        choice = st.selectbox("Select a query to review", pending, key="queue_pick")
        pending_review_panel(engine, choice, ws_id, user)


def load_user_threads(engine, username: str, ws_id: str) -> list[dict]:
    """Retrieve existing conversation threads for this user in this workspace."""
    prefix = f"ui-{ws_id}-"
    threads = []
    for tid in fetch_checkpoint_threads(prefix=prefix, limit=50, pool=shared_pool(engine.checkpointer)):
        if not tid.startswith(prefix):
            continue
        try:
            state = engine.get_state({"configurable": {"thread_id": tid}})
            if state.values and state.values.get("principal") == username:
                q = (
                    state.values.get("original_question")
                    or state.values.get("sanitized_question")
                    or state.values.get("question")
                    or ""
                ).strip()
                if not q:
                    continue
                is_paused = state.next == ("human_approval",)
                status_label = "⏳ Awaiting Review" if is_paused else "✅ Answered"
                short_q = (q[:32] + "...") if len(q) > 32 else q
                threads.append({
                    "thread_id": tid,
                    "label": f"{short_q} ({status_label})",
                    "question": q,
                    "generation": state.values.get("generation", ""),
                    "citations": state.values.get("citations", []),
                    "degraded": state.values.get("degraded", []),
                    "paused": is_paused,
                })
        except Exception:
            continue
    return list(reversed(threads))


@st.cache_resource(show_spinner=False)
def compiled_engine(db_uri: str):
    """One checkpointer and compiled graph per server process and database, shared by every session.

    Reruns (every click) reuse them instead of reconnecting and recompiling. With PostgreSQL this keeps
    a single pool (CHECKPOINT_POOL_SIZE connections) open instead of opening one per rerun.
    """
    saver = open_checkpointer(db_uri)
    atexit.register(close_checkpointer, saver)
    return workflow.compile(checkpointer=saver)


login_gate()
user = st.session_state.user
admin = is_admin(user)

# --- SIDEBAR: ACCOUNT & WORKSPACE CONTROLS ---
st.sidebar.title("Document Bundles")
st.sidebar.caption(f"Signed in as **{user}**{' (admin)' if admin else ''}")
if st.sidebar.button("Sign out"):
    _sign_out(revoke=True)
    st.rerun()

# The audit panel comes before the workspace check, so an administrator without grants can still reach it.
if admin:
    with st.sidebar.expander("Audit log (administrators)"):
        if st.button("Verify chain and anchors"):
            chain_ok, bad_seq, count = verify_chain()
            if chain_ok:
                st.success(f"Chain intact: {count} records.")
            else:
                st.error(f"Chain broken at seq {bad_seq}.")
            anchors_ok, problems, anchor_count = verify_anchors()
            if anchor_count == 0:
                st.info("No anchors yet. Run make audit-anchor, then copy data/audit_anchors.jsonl off the host.")
            elif anchors_ok:
                st.success(f"{anchor_count} anchors match the log.")
            for problem in problems:
                st.error(problem)
        st.caption("Most recent events (payloads decrypted for administrators):")
        for event in list_events(limit=10):
            st.caption(f"{event['ts'][:19]} · {event['event_type']} · {event['workspace_id'] or '-'}")
        if st.button("Prepare export"):
            export = export_events()
            # Exporting decrypted records is itself recorded in the chain.
            record("audit_export", workspace_id="", session_id="", data={"actor": user, "records": len(export)})
            payload = "\n".join(json.dumps(event, sort_keys=True) for event in export)
            st.download_button("Download audit export (JSON lines)", payload, file_name="audit-export.jsonl")

all_workspaces = list_workspaces()
if admin and not all_workspaces:
    create_workspace("general", actor=user)
    all_workspaces = ["general"]

visible = allowed_workspaces(user, all_workspaces)
if not visible:
    st.info("No workspaces are granted to your account. Ask an administrator for access.")
    st.stop()

active_ws = st.sidebar.selectbox("Active Workspace", visible)

stats = get_workspace_stats(active_ws)
st.sidebar.caption(f"Status: **{stats['files']}** files | **{stats['chunks']}** indexed chunks")

engine = compiled_engine(checkpointer_module.DB_URI)

user_threads = load_user_threads(engine, user, active_ws)

# Reset session memory when switching workspace
if "current_ws" not in st.session_state or st.session_state.current_ws != active_ws:
    st.session_state.current_ws = active_ws
    st.session_state.pop("thread_id", None)
    st.session_state.pop("messages", None)

# Auto-restore latest thread if messages not yet initialized
if not st.session_state.get("thread_id") or "messages" not in st.session_state:
    if user_threads:
        latest = user_threads[0]
        st.session_state.thread_id = latest["thread_id"]
        msgs = [{"role": "user", "content": latest["question"]}]
        if not latest["paused"] and latest["generation"]:
            msgs.append({
                "role": "assistant",
                "content": latest["generation"],
                "citations": latest["citations"],
                "degraded": latest["degraded"],
            })
        st.session_state.messages = msgs
    else:
        st.session_state.thread_id = f"ui-{active_ws}-{uuid.uuid4().hex}"
        st.session_state.messages = []

# Sidebar: Conversation History
with st.sidebar.expander(f"Conversations ({len(user_threads)})", expanded=True):
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.thread_id = f"ui-{active_ws}-{uuid.uuid4().hex}"
        st.session_state.messages = []
        st.rerun()

    if user_threads:
        opts = {t["thread_id"]: t["label"] for t in user_threads}
        curr_tid = st.session_state.get("thread_id", "")
        if curr_tid and curr_tid not in opts:
            opts[curr_tid] = "➕ New Chat (in progress)"

        opt_keys = list(opts.keys())
        curr_idx = opt_keys.index(curr_tid) if curr_tid in opt_keys else 0
        selected_tid = st.selectbox(
            "Select conversation",
            options=opt_keys,
            format_func=lambda tid: opts[tid],
            index=curr_idx,
            key="thread_selector",
        )
        if selected_tid != curr_tid:
            st.session_state.thread_id = selected_tid
            target = next((t for t in user_threads if t["thread_id"] == selected_tid), None)
            if target:
                msgs = [{"role": "user", "content": target["question"]}]
                if not target["paused"] and target["generation"]:
                    msgs.append({
                        "role": "assistant",
                        "content": target["generation"],
                        "citations": target["citations"],
                        "degraded": target["degraded"],
                    })
                st.session_state.messages = msgs
            else:
                st.session_state.messages = []
            st.rerun()

if admin:
    # 1. User Grants & Auto-Approval
    with st.sidebar.expander("User Grants & Auto-Approval (Admin)"):
        policy_data = load_policy()
        global_defaults = get_default_permissions()
        report = review_mode_report(list_workspaces())
        for line in describe_review_mode(report):
            st.caption(line)
        for warning in report.warnings:
            st.warning(warning)
        st.caption(f"Global default permissions: `{', '.join(global_defaults) or 'none'}`")
        def_q = st.checkbox("Default: query", value="query" in global_defaults, key="def_perm_q")
        def_auto = st.checkbox("Default: auto_approve", value="auto_approve" in global_defaults, key="def_perm_auto")
        if st.button("Save Global Defaults", use_container_width=True, key="save_global_defaults"):
            next_defaults = []
            if def_q:
                next_defaults.append("query")
            if def_auto:
                next_defaults.append("auto_approve")
            set_default_permissions(next_defaults, actor=user)
            st.success(f"Global defaults updated to: {', '.join(next_defaults) or 'none'}")
            st.rerun()

        st.divider()
        user_list = sorted(policy_data["users"].keys())
        if user_list:
            selected_user = st.selectbox("Account", user_list, key="grant_user_select")
            user_info = policy_data["users"][selected_user]
            curr_clr = user_info.get("clearance", "public")
            curr_admin = bool(user_info.get("admin", False))
            st.caption(f"Clearance: `{curr_clr}` | Admin: `{curr_admin}`")

            clr_idx = CLEARANCE_LEVELS.index(curr_clr) if curr_clr in CLEARANCE_LEVELS else 0
            new_clr = st.selectbox(
                "Clearance Level",
                list(CLEARANCE_LEVELS),
                index=clr_idx,
                key=f"clr_select_{selected_user}",
            )
            new_admin = st.checkbox(
                "Administrator privilege",
                value=curr_admin,
                key=f"admin_chk_{selected_user}",
            )
            if (new_clr != curr_clr or new_admin != curr_admin) and st.button(
                "Update Role / Clearance",
                use_container_width=True,
                key=f"save_role_{selected_user}",
            ):
                try:
                    if new_clr != curr_clr:
                        set_user_clearance(selected_user, new_clr, actor=user)
                    if new_admin != curr_admin:
                        set_admin(selected_user, new_admin, actor=user)
                    st.success(f"Updated role/clearance for '{selected_user}'.")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))

            ws_options = ["all"] + all_workspaces
            target_ws = st.selectbox("Workspace", ws_options, key="grant_ws_select")

            effective_ws = active_ws if target_ws == "all" else target_ws
            curr_grants = effective_permissions(policy_data, policy_data["users"][selected_user], effective_ws)
            q_checked = "query" in curr_grants
            r_checked = "review" in curr_grants
            auto_checked = "auto_approve" in curr_grants

            grant_q = st.checkbox("query (ask questions)", value=q_checked, key=f"g_q_{selected_user}_{target_ws}")
            grant_r = st.checkbox("review (approve answers)", value=r_checked, key=f"g_r_{selected_user}_{target_ws}")
            grant_auto = st.checkbox("auto_approve (bypass compliance gate)", value=auto_checked, key=f"g_a_{selected_user}_{target_ws}")

            if st.button("Save Grants", use_container_width=True, key=f"save_grants_{selected_user}_{target_ws}"):
                new_perms = []
                if grant_q:
                    new_perms.append("query")
                if grant_r:
                    new_perms.append("review")
                if grant_auto:
                    new_perms.append("auto_approve")

                if target_ws == "all":
                    set_user_permissions_everywhere(selected_user, new_perms, actor=user)
                else:
                    set_grant(selected_user, target_ws, new_perms, actor=user)
                st.success(f"Updated grants for '{selected_user}' on {target_ws}: {', '.join(new_perms) or 'none'}")
                st.rerun()

            if st.button("Reset to Defaults", use_container_width=True, key=f"reset_grants_{selected_user}_{target_ws}"):
                if target_ws == "all":
                    from access import clear_user_permissions_override
                    clear_user_permissions_override(selected_user, actor=user)
                else:
                    reset_grant(selected_user, target_ws, actor=user)
                st.success(f"Reset '{selected_user}' on {target_ws} to follow default permissions.")
                st.rerun()

    # 1b. Add New User Account
    with st.sidebar.expander("Add User Account (Admin)"):
        new_u_name = st.text_input("New Username", key="new_user_name").strip()
        new_u_pass = st.text_input("Initial Password (min 12 chars)", type="password", key="new_user_pass")
        new_u_clr = st.selectbox("Initial Clearance", list(CLEARANCE_LEVELS), index=0, key="new_user_clr")
        new_u_adm = st.checkbox("Grant Admin Role", value=False, key="new_user_adm")
        if st.button("Create User", use_container_width=True, key="create_user_btn"):
            try:
                add_user(new_u_name, new_u_pass, admin=new_u_adm, clearance=new_u_clr, actor=user)
                st.success(f"Created account '{new_u_name.lower()}' with default workspace access.")
                st.rerun()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))

    # 2. Add New Workspace
    with st.sidebar.expander("Create Workspace Bundle"):
        new_name = st.text_input("Name (e.g., accounting, strategy, marketing)").strip()
        if st.button("Create Bundle", use_container_width=True):
            clean_name = sanitize_workspace_id(new_name)
            if not clean_name:
                st.error("Enter a valid alphanumeric workspace name.")
            elif clean_name not in all_workspaces:
                create_workspace(clean_name, actor=user)
                st.success(f"Workspace '{clean_name}' created.")
                st.rerun()
            else:
                st.warning("A workspace with that name already exists.")

    # 3. Delete Workspace
    with st.sidebar.expander("Delete Bundle"):
        st.write(f"Active: **{active_ws}**")
        confirm = st.checkbox(f"Confirm deletion of '{active_ws}'")
        if st.button("Delete Permanently", type="primary", disabled=not confirm, use_container_width=True):
            delete_workspace(active_ws, actor=user)
            st.success(f"Removed '{active_ws}'.")
            st.rerun()

    st.sidebar.divider()

    # 4. Multi-Format Document Ingestion into Active Workspace
    st.sidebar.subheader(f"Add Files to [{active_ws}]")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
    )

    if st.sidebar.button(f"Index into '{active_ws}'", use_container_width=True) and uploaded_files:
        accepted = 0
        for uf in uploaded_files:
            if uf.size > MAX_SOURCE_BYTES:
                st.sidebar.error(f"'{uf.name}' is larger than {MAX_SOURCE_BYTES // (1024 * 1024)} MB and was not saved.")
                continue
            try:
                target = staged_path(active_ws, uf.name)
            except ValueError as exc:
                st.sidebar.error(f"'{uf.name}' was not saved: {exc}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(uf.getbuffer())
            accepted += 1

        if accepted:
            with st.spinner(f"Parsing, chunking, and embedding into '{active_ws}'..."):
                res = ingest_workspace(active_ws, str(Path(UPLOADS_DIR) / active_ws), actor=user)
            for skipped_note in res.skipped:
                st.sidebar.warning(f"Skipped {skipped_note}")
            if not res.ok:
                st.sidebar.error(f"Indexing failed: {res.error}")
            else:
                st.sidebar.success(f"Successfully indexed {res.documents} file(s) ({res.chunks} chunks).")
                st.rerun()

# --- MAIN CHAT AREA ---
st.title(f"Workspace: `{active_ws}`")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        render_degraded(msg.get("degraded"))
        if msg.get("citations"):
            render_citations(msg["citations"])

config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}

review_queue(engine, active_ws, user, own_thread=st.session_state.thread_id)
paused = pending_review_panel(engine, st.session_state.thread_id, active_ws, user)

# Sync completed answer back into chat if approved/rejected externally (e.g. via CLI or another reviewer tab)
if not paused and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    completed_state = engine.get_state(config)
    if completed_state.values and not completed_state.next and completed_state.values.get("generation"):
        gen_text = completed_state.values.get("generation", "")
        rejected = "REJECTED BY HUMAN COMPLIANCE" in gen_text
        cites = [] if rejected else completed_state.values.get("citations", [])
        deg = [] if rejected else completed_state.values.get("degraded", [])
        st.session_state.messages.append({
            "role": "assistant",
            "content": gen_text,
            "citations": cites,
            "degraded": deg,
        })
        st.rerun()

can_query = authorize(user, active_ws, "query")
user_query = st.chat_input(
    f"Ask a question grounded in '{active_ws}'..." if can_query else f"Your account does not have query permission on '{active_ws}'.",
    disabled=paused or not can_query,
)

if user_query:
    # If active thread was already completed, start fresh thread
    prior_state = engine.get_state(config)
    if prior_state.values and not prior_state.next and prior_state.values.get("generation"):
        st.session_state.thread_id = f"ui-{active_ws}-{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        st.session_state.messages = []

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    inputs: GraphState = {
        "workspace_id": active_ws,
        "session_id": st.session_state.thread_id,
        # The pipeline verifies this token itself. The plain name is never sent.
        "principal_token": st.session_state.token,
        # Redacted before submission: LangGraph persists the input in the checkpoint before any node runs.
        "question": redact_pii(user_query),
        "original_question": redact_pii(user_query),
        "sanitized_question": "",
        "documents": [],
        "citations": [],
        "generation": "",
        "intent": "",
        "cache_hit": False,
        "is_safe": True,
        "grade_status": "",
        "hallucination_status": "",
        "relevance_status": "",
        "retry_count": 0,
        "generation_retry_count": 0,
        "grader_feedback": "",
        "history": [],
        "telemetry": {}
    }

    with st.spinner("Processing query via guardrails, hybrid search, and Self-RAG..."):
        for _ in engine.stream(inputs, config=config):
            pass

    state = engine.get_state(config)

    if state.next == ("human_approval",):
        # The review panel renders on the next run.
        st.rerun()

    ans = state.values.get("generation", "No answer generated.")
    citations = state.values.get("citations", [])
    degraded = state.values.get("degraded", [])
    st.session_state.messages.append({
        "role": "assistant",
        "content": ans,
        "citations": citations,
        "degraded": degraded,
    })
    with st.chat_message("assistant"):
        st.write(ans)
        render_degraded(degraded)
        render_citations(citations)
    st.rerun()

import atexit
import streamlit as st
import json
import time
import uuid
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from rag_lab.security.access import (
    CLEARANCE_LEVELS,
    ROLES,
    WORKSPACE_ROLES,
    PolicyIntegrityError,
    add_to_group,
    add_user,
    begin_totp_enrollment,
    allowed_workspaces,
    authorize,
    clearance_covers,
    confirm_totp_enrollment,
    create_group,
    describe_review_mode,
    disable_totp,
    effective_permissions,
    get_default_permissions,
    get_default_role,
    grant_group,
    get_user_clearance,
    is_admin,
    is_idle_expired,
    is_owner,
    list_groups,
    load_policy,
    recovery_codes_left,
    regenerate_recovery_codes,
    remove_from_group,
    reset_grant,
    review_mode_report,
    revoke_sessions,
    role_of,
    set_default_role,
    set_grant,
    set_role,
    set_user_clearance,
    set_user_permissions_everywhere,
    sign_in,
    totp_enabled,
    verify_token,
)
from rag_lab.pipeline.graph import workflow
from rag_lab.pipeline.entry import prepare_pipeline_input
from rag_lab.pipeline.state import REJECTION_NOTICE
from rag_lab.security.audit import export_events, list_events, record, verify_anchors, verify_chain
from rag_lab.ops.approve import list_pending_threads
from rag_lab.storage.checkpointer import close_checkpointer, fetch_checkpoint_threads, open_checkpointer, shared_pool
from rag_lab.config import settings
from rag_lab.ingestion.ingest import ingest_workspace
from rag_lab.ingestion.parsers import SUPPORTED_EXTENSIONS
from rag_lab.storage.workspace_manager import (
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
    st.query_params.clear()
    if notice:
        st.session_state.notice = notice


def login_gate() -> None:
    """Stops the script until a user holds a valid, unexpired session."""
    token = st.session_state.get("token") or st.query_params.get("session", "")
    if token:
        username = verify_token(token)
        if not username:
            _sign_out("Your session has expired. Please sign in again.")
        elif "last_seen" in st.session_state and is_idle_expired(st.session_state.last_seen, time.time()):
            _sign_out("You were signed out after 30 minutes of inactivity.")
        else:
            st.session_state.user = username
            st.session_state.token = token
            st.session_state.last_seen = time.time()
            st.query_params["session"] = token
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
        code = st.text_input("Authentication code (only if two-factor sign-in is on)", key="login_code")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        result = sign_in(username, password, code=code)
        if result.ok:
            st.session_state.user = result.username
            st.session_state.token = result.token
            st.session_state.last_seen = time.time()
            st.query_params["session"] = result.token
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

    interrupt_payload: dict[str, Any] = {}
    for task in state.tasks or []:
        if getattr(task, "interrupts", None):
            interrupt_payload = task.interrupts[0].value or {}
            break
    review_reasons = interrupt_payload.get("review_reasons") or []

    st.warning("Candidate answer paused at the compliance approval gate.")
    if review_reasons:
        st.caption(f"Gate triggered by: `{', '.join(review_reasons)}`")
    display_q = state.values.get("original_question") or state.values.get("sanitized_question", "")
    asked_q = state.values.get("asked_question") or display_q
    st.caption(f"Question ({req_clearance}): {asked_q}")
    if asked_q != display_q:
        st.caption(f"Read in the conversation as: {display_q}")
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
                    "content": REJECTION_NOTICE,
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


def thread_messages(thread: dict) -> list[dict]:
    """Chat messages for a conversation: every finished turn from its history, then the current one."""
    msgs: list[dict] = []
    turns = [h for h in thread.get("history", []) if isinstance(h, dict)]
    for turn in turns:
        msgs.append({"role": "user", "content": turn.get("asked") or turn.get("question", "")})
        msgs.append({"role": "assistant", "content": turn.get("answer") or REJECTION_NOTICE})
    last_asked = turns[-1].get("asked") if turns else None
    if thread["paused"] or thread["question"] != last_asked:
        msgs.append({"role": "user", "content": thread["question"]})
        if not thread["paused"] and thread["generation"] and not turns:
            msgs.append({"role": "assistant", "content": thread["generation"],
                         "citations": thread["citations"], "degraded": thread["degraded"]})
    if msgs and msgs[-1]["role"] == "assistant" and not thread["paused"]:
        msgs[-1].update({"citations": thread["citations"], "degraded": thread["degraded"]})
    return msgs


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
                    state.values.get("asked_question")
                    or state.values.get("original_question")
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
                    "history": list(state.values.get("history", [])),
                    "paused": is_paused,
                })
        except Exception:
            continue
    return threads


def _release_engine(engine) -> None:
    close_checkpointer(engine.checkpointer)


@st.cache_resource(show_spinner=False, on_release=_release_engine)
def compiled_engine(db_uri: str):
    """One checkpointer and compiled graph per server process and database, shared by every session.

    Reruns (every click) reuse them instead of reconnecting and recompiling. With PostgreSQL this keeps
    a single pool (CHECKPOINT_POOL_SIZE connections) open instead of opening one per rerun. The
    connection is closed when the cache entry is released, or at process exit.
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
                st.info("No anchors yet. Run make audit-anchor (set AUDIT_ANCHOR_COPY_DIR to keep a second copy elsewhere).")
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

target_ws = st.session_state.get("current_ws") or st.query_params.get("ws")
default_ws_idx = visible.index(target_ws) if target_ws in visible else 0
active_ws = st.sidebar.selectbox("Active Workspace", visible, index=default_ws_idx)
st.query_params["ws"] = active_ws

stats = get_workspace_stats(active_ws)
st.sidebar.caption(f"Status: **{stats['files']}** files | **{stats['chunks']}** indexed chunks")

engine = compiled_engine(settings.db_uri)

user_threads = load_user_threads(engine, user, active_ws)

# Reset session memory when switching workspace
if "current_ws" not in st.session_state or st.session_state.current_ws != active_ws:
    st.session_state.current_ws = active_ws
    st.session_state.pop("thread_id", None)
    st.session_state.pop("messages", None)
    st.query_params["ws"] = active_ws
    st.query_params.pop("thread", None)

# Auto-restore thread: first check query param/state, then newest thread
target_tid = st.session_state.get("thread_id") or st.query_params.get("thread")
target_thread = next((t for t in user_threads if t["thread_id"] == target_tid), None) if target_tid else None

if not st.session_state.get("thread_id") or "messages" not in st.session_state:
    if target_thread:
        st.session_state.thread_id = target_thread["thread_id"]
        st.session_state.messages = thread_messages(target_thread)
    elif user_threads:
        latest = user_threads[0]
        st.session_state.thread_id = latest["thread_id"]
        st.session_state.messages = thread_messages(latest)
    else:
        st.session_state.thread_id = f"ui-{active_ws}-{uuid.uuid4().hex}"
        st.session_state.messages = []

if st.session_state.get("thread_id"):
    st.query_params["thread"] = st.session_state.thread_id

# Sidebar: Conversation History
with st.sidebar.expander(f"Conversations ({len(user_threads)})", expanded=True):
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.thread_id = f"ui-{active_ws}-{uuid.uuid4().hex}"
        st.session_state.messages = []
        st.query_params["thread"] = st.session_state.thread_id
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
            st.query_params["thread"] = selected_tid
            target = next((t for t in user_threads if t["thread_id"] == selected_tid), None)
            if target:
                st.session_state.messages = thread_messages(target)
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
        current_default = get_default_role()
        st.caption(
            f"Default role (accounts without a role): `{current_default or 'custom'}` "
            f"({', '.join(global_defaults) or 'none'})"
        )
        default_choice = st.selectbox(
            "Default role",
            list(WORKSPACE_ROLES),
            index=WORKSPACE_ROLES.index(current_default) if current_default in WORKSPACE_ROLES else 0,
            key="default_role_select",
        )
        if default_choice != current_default and st.button(
            "Save Default Role", use_container_width=True, key="save_default_role"
        ):
            set_default_role(default_choice, actor=user)
            st.success(f"Default role is now '{default_choice}'.")
            st.rerun()

        st.divider()
        user_list = sorted(policy_data["users"].keys())
        if user_list:
            selected_user = st.selectbox("Account", user_list, key="grant_user_select")
            user_info = policy_data["users"][selected_user]
            curr_clr = user_info.get("clearance", "public")
            curr_role = role_of(user_info) or "default"
            st.caption(
                f"Clearance: `{curr_clr}` | Role: `{curr_role}` | "
                f"Groups: `{', '.join(user_info.get('groups') or []) or '-'}`"
            )

            clr_idx = CLEARANCE_LEVELS.index(curr_clr) if curr_clr in CLEARANCE_LEVELS else 0
            new_clr = st.selectbox(
                "Clearance Level",
                list(CLEARANCE_LEVELS),
                index=clr_idx,
                key=f"clr_select_{selected_user}",
            )
            role_choices = ["default", *ROLES]
            new_role = st.selectbox(
                "Role",
                role_choices,
                index=role_choices.index(curr_role),
                key=f"role_select_{selected_user}",
            )
            if (new_clr != curr_clr or new_role != curr_role) and st.button(
                "Update Role / Clearance",
                use_container_width=True,
                key=f"save_role_{selected_user}",
            ):
                try:
                    if new_clr != curr_clr:
                        set_user_clearance(selected_user, new_clr, actor=user)
                    if new_role != curr_role:
                        set_role(selected_user, new_role, actor=user)
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
                    from rag_lab.security.access import clear_user_permissions_override
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
        new_u_role = st.selectbox(
            "Role", ["default", "member", "trusted", "reviewer", "admin", "owner", "none"], key="new_user_role",
            help=f"'default' follows the default role ({get_default_role() or 'custom'}).",
        )
        if st.button("Create User", use_container_width=True, key="create_user_btn"):
            try:
                add_user(new_u_name, new_u_pass, clearance=new_u_clr, actor=user,
                         role=None if new_u_role == "default" else new_u_role)
                st.session_state.account_notice = (
                    f"Created account '{new_u_name.lower()}' with role '{new_u_role}'.",
                    review_mode_report(list_workspaces()).gaps,
                )
                st.rerun()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
        notice = st.session_state.pop("account_notice", None)
        if notice:
            st.success(notice[0])
            for gap in notice[1]:
                st.warning(f"Answers may wait with nobody able to approve them: {gap}")

    # 1c. Groups
    with st.sidebar.expander("Groups (Admin)"):
        groups = list_groups()
        for g in groups:
            grants = ", ".join(f"{'all' if ws == '*' else ws}: {r}" for ws, r in sorted(g["grants"].items()))
            st.caption(f"**{g['group']}**: {', '.join(g['members']) or 'no members'} | {grants or 'no grants'}")
        new_group = st.text_input("New group", key="new_group_name").strip()
        if st.button("Create Group", use_container_width=True, key="create_group_btn"):
            try:
                create_group(new_group, actor=user)
                st.rerun()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
        group_names = [g["group"] for g in groups]
        if group_names:
            target_group = st.selectbox("Group", group_names, key="group_select")
            member = st.selectbox("Account", sorted(load_policy()["users"]), key="group_member_select")
            in_group = member in next(g["members"] for g in groups if g["group"] == target_group)
            if st.button("Remove from group" if in_group else "Add to group", use_container_width=True,
                         key="group_member_btn"):
                try:
                    (remove_from_group if in_group else add_to_group)(member, target_group, actor=user)
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
            group_ws = st.selectbox("Workspace", ["all", *all_workspaces], key="group_ws_select")
            group_role = st.selectbox("Role on workspace", ["default", *WORKSPACE_ROLES], key="group_role_select")
            if st.button("Save Group Grant", use_container_width=True, key="group_grant_btn"):
                try:
                    grant_group(target_group, group_ws, group_role, actor=user)
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))

    # 1d. Two-factor sign-in for the signed-in owner or administrator
    with st.sidebar.expander("Two-factor sign-in (Admin)"):
        if totp_enabled(user):
            st.caption(f"On for **{user}**. Recovery codes left: {recovery_codes_left(user)}.")
            current = st.text_input("Authentication or recovery code", key="totp_current_code")
            off_col, new_col = st.columns(2)
            try:
                if off_col.button("Turn off", key="totp_off_btn"):
                    disable_totp(user, current, actor=user)
                    st.rerun()
                if new_col.button("New recovery codes", key="totp_recovery_btn"):
                    st.session_state.recovery_codes = regenerate_recovery_codes(user, current, actor=user)
                    st.rerun()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
        elif "totp_enrolment" in st.session_state:
            enrolment = st.session_state.totp_enrolment
            st.caption("Add this secret to your authenticator app (or open the link on the device), then enter a code.")
            st.code(enrolment.secret)
            st.code(enrolment.uri)
            confirm_code = st.text_input("Code from the app", key="totp_confirm_code")
            if st.button("Confirm", key="totp_confirm_btn"):
                try:
                    st.session_state.recovery_codes = confirm_totp_enrollment(user, confirm_code, actor=user)
                    del st.session_state["totp_enrolment"]
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
        elif st.button("Set up two-factor sign-in", key="totp_setup_btn"):
            try:
                st.session_state.totp_enrolment = begin_totp_enrollment(user, actor=user)
                st.rerun()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
        if "recovery_codes" in st.session_state:
            st.warning("Recovery codes: each works once and they are not shown again. Store them somewhere safe.")
            st.code("\n".join(st.session_state.recovery_codes))
            if st.button("I saved them", key="totp_codes_saved_btn"):
                del st.session_state["recovery_codes"]
                st.rerun()
        if is_owner(user):
            enrolled = sorted(
                name for name, info in load_policy()["users"].items() if info.get("totp") and name != user
            )
            if enrolled:
                st.divider()
                reset_target = st.selectbox("Reset another account's two-factor sign-in", enrolled, key="totp_reset_select")
                if st.button("Reset (signs them out)", key="totp_reset_btn"):
                    disable_totp(reset_target, actor=user)
                    st.rerun()

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
            if uf.size > settings.max_source_bytes:
                st.sidebar.error(f"'{uf.name}' is larger than {settings.max_source_bytes // (1024 * 1024)} MB and was not saved.")
                continue
            try:
                staged = staged_path(active_ws, uf.name)
            except ValueError as exc:
                st.sidebar.error(f"'{uf.name}' was not saved: {exc}")
                continue
            staged.parent.mkdir(parents=True, exist_ok=True)
            with open(staged, "wb") as handle:
                handle.write(uf.getbuffer())
            accepted += 1

        if accepted:
            with st.spinner(f"Parsing, chunking, and embedding into '{active_ws}'..."):
                res = ingest_workspace(active_ws, str(Path(settings.uploads_dir) / active_ws), actor=user)
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
        rejected = gen_text == REJECTION_NOTICE
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
    # A conversation keeps its thread, so follow-up questions are read with the earlier turns ("New Chat"
    # starts a fresh one). The input box is disabled while the thread waits for a review decision.

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    inputs = prepare_pipeline_input(
        workspace_id=active_ws,
        session_id=st.session_state.thread_id,
        principal_token=st.session_state.token,
        question=user_query,
    )

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

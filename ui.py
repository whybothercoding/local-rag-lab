import streamlit as st
import uuid
import os
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from access import allowed_workspaces, authenticate, authorize, is_admin, load_policy
from app import workflow, GraphState
from approve import list_pending_threads
from checkpointer import get_checkpointer
from ingest import ingest_workspace
from redaction import redact_pii
from workspace_manager import (
    list_workspaces,
    create_workspace,
    delete_workspace,
    get_workspace_stats,
    UPLOADS_DIR,
)

st.set_page_config(page_title="Local Enterprise Knowledge Hub", layout="wide")


def render_citations(citations):
    """Renders visual expandable cards showing retrieved document sources."""
    if not citations:
        return
    with st.expander("Retrieved Source Citations", expanded=False):
        for idx, citation in enumerate(citations, 1):
            st.markdown(f"**[{idx}]** `{citation}`")


def login_gate() -> None:
    """Stops the script until a user is signed in."""
    if st.session_state.get("user"):
        return
    st.title("Local Enterprise Knowledge Hub")
    if not load_policy()["users"]:
        st.error("No accounts exist yet. Create one with: python access.py add-user <name> --admin")
        st.stop()
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        name = username.strip().lower()
        if authenticate(name, password):
            st.session_state.user = name
            st.rerun()
        st.error("Invalid username or password.")
    st.stop()


def pending_review_panel(engine, thread_id: str, ws_id: str, user: str) -> bool:
    """Show the compliance gate for one paused thread. Returns True while that thread awaits a decision."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = engine.get_state(config)
    if state.next != ("human_approval",):
        return False

    st.warning("Candidate answer paused at the compliance approval gate.")
    st.caption(f"Question: {state.values.get('sanitized_question', '')}")
    st.text_area("Candidate Output", value=state.values.get("generation", ""), height=130,
                 disabled=True, key=f"candidate-{thread_id}")
    render_citations(state.values.get("citations", []))

    requester = state.values.get("principal", "")
    if not authorize(user, ws_id, "review") or user == requester:
        st.info("Awaiting a decision from another reviewer with access to this workspace.")
        return True

    reason = st.text_input("Reason (optional, recorded in the audit log)", key=f"reason-{thread_id}").strip()
    st.caption(f"Your decision will be recorded under **{user}**.")
    col1, col2 = st.columns(2)
    approve_clicked = col1.button("Approve Output", key=f"approve-{thread_id}")
    reject_clicked = col2.button("Reject Output", key=f"reject-{thread_id}")

    if approve_clicked or reject_clicked:
        decision = "approved" if approve_clicked else "rejected"
        resume = {"decision": decision, "approver": user, "reason": reason}
        for _ in engine.stream(Command(resume=resume), config=config):
            pass
        if thread_id == st.session_state.thread_id:
            final_state = engine.get_state(config)
            if decision == "approved":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_state.values.get("generation", ""),
                    "citations": final_state.values.get("citations", []),
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
    """Reviewers see other users' paused queries for the active workspace."""
    if not authorize(user, ws_id, "review"):
        return
    pending = [
        tid for tid in list_pending_threads(engine)
        if tid.startswith(f"ui-{ws_id}-") and tid != own_thread
    ]
    if not pending:
        return
    with st.expander(f"Review queue: {len(pending)} awaiting decision", expanded=True):
        choice = st.selectbox("Select a query to review", pending, key="queue_pick")
        pending_review_panel(engine, choice, ws_id, user)


login_gate()
user = st.session_state.user
admin = is_admin(user)

# --- SIDEBAR: ACCOUNT & WORKSPACE CONTROLS ---
st.sidebar.title("Document Bundles")
st.sidebar.caption(f"Signed in as **{user}**{' (admin)' if admin else ''}")
if st.sidebar.button("Sign out"):
    del st.session_state["user"]
    st.rerun()

all_workspaces = list_workspaces()
if admin and not all_workspaces:
    create_workspace("general")
    all_workspaces = ["general"]

visible = allowed_workspaces(user, all_workspaces)
if not visible:
    st.info("No workspaces are granted to your account. Ask an administrator for access.")
    st.stop()

active_ws = st.sidebar.selectbox("Active Workspace", visible)

stats = get_workspace_stats(active_ws)
st.sidebar.caption(f"Status: **{stats['files']}** files | **{stats['chunks']}** indexed chunks")

if admin:
    # 2. Add New Workspace
    with st.sidebar.expander("Create Workspace Bundle"):
        new_name = st.text_input("Name (e.g., accounting, strategy, marketing)").strip()
        if st.button("Create Bundle", use_container_width=True):
            clean_name = new_name.lower().replace(" ", "_")
            if clean_name and clean_name not in all_workspaces:
                create_workspace(clean_name)
                st.success(f"Workspace '{clean_name}' created. Grant users access with access.py.")
                st.rerun()
            elif clean_name in all_workspaces:
                st.warning("A workspace with that name already exists.")

    # 3. Delete Workspace
    with st.sidebar.expander("Delete Bundle"):
        st.write(f"Active: **{active_ws}**")
        confirm = st.checkbox(f"Confirm deletion of '{active_ws}'")
        if st.button("Delete Permanently", type="primary", disabled=not confirm, use_container_width=True):
            delete_workspace(active_ws)
            st.success(f"Removed '{active_ws}'.")
            st.rerun()

    st.sidebar.divider()

    # 4. Multi-Format Document Ingestion into Active Workspace
    st.sidebar.subheader(f"Add Files to [{active_ws}]")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents (.md, .txt, .pdf, .docx)",
        accept_multiple_files=True,
        type=["md", "txt", "pdf", "docx"]
    )

    if st.sidebar.button(f"Index into '{active_ws}'", use_container_width=True) and uploaded_files:
        target_dir = os.path.join(UPLOADS_DIR, active_ws)
        os.makedirs(target_dir, exist_ok=True)

        for uf in uploaded_files:
            with open(os.path.join(target_dir, uf.name), "wb") as f:
                f.write(uf.getbuffer())

        with st.spinner(f"Parsing, chunking, and embedding into '{active_ws}'..."):
            ingest_workspace(active_ws, target_dir)
        st.sidebar.success(f"Successfully indexed {len(uploaded_files)} file(s).")
        st.rerun()

# --- MAIN CHAT AREA ---
st.title(f"Workspace: `{active_ws}`")

# Reset session memory when switching workspace
if "current_ws" not in st.session_state or st.session_state.current_ws != active_ws:
    st.session_state.current_ws = active_ws
    st.session_state.thread_id = f"ui-{active_ws}-{uuid.uuid4().hex[:6]}"
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])

config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}

with get_checkpointer() as checkpointer:
    engine = workflow.compile(checkpointer=checkpointer)
    review_queue(engine, active_ws, user, own_thread=st.session_state.thread_id)
    paused = pending_review_panel(engine, st.session_state.thread_id, active_ws, user)
    user_query = st.chat_input(f"Ask a question grounded in '{active_ws}'...", disabled=paused)

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        inputs: GraphState = {
            "workspace_id": active_ws,
            "session_id": st.session_state.thread_id,
            "principal": user,
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
        st.session_state.messages.append({
            "role": "assistant",
            "content": ans,
            "citations": citations
        })
        with st.chat_message("assistant"):
            st.write(ans)
            render_citations(citations)

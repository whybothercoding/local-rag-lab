import streamlit as st
import uuid
import os
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from app import workflow, DB_URI
from ingest import ingest_workspace
from workspace_manager import (
    list_workspaces,
    create_workspace,
    delete_workspace,
    get_workspace_stats,
    UPLOADS_DIR,
)

st.set_page_config(page_title="Local Enterprise Knowledge Hub", layout="wide")

# --- SIDEBAR: WORKSPACE CONTROLS ---
st.sidebar.title("Document Bundles")

# 1. Discover Active Workspaces
workspaces = list_workspaces()
if not workspaces:
    create_workspace("general")
    workspaces = ["general"]

active_ws = st.sidebar.selectbox("Active Workspace", workspaces)

# Workspace Metrics
stats = get_workspace_stats(active_ws)
st.sidebar.caption(f"Status: **{stats['files']}** files | **{stats['chunks']}** indexed chunks")

# 2. Add New Workspace
with st.sidebar.expander("Create Workspace Bundle"):
    new_name = st.text_input("Name (e.g., accounting, strategy, marketing)").strip()
    if st.button("Create Bundle", use_container_width=True):
        clean_name = new_name.lower().replace(" ", "_")
        if clean_name and clean_name not in workspaces:
            create_workspace(clean_name)
            st.success(f"Workspace '{clean_name}' created.")
            st.rerun()
        elif clean_name in workspaces:
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

user_query = st.chat_input(f"Ask a question grounded in '{active_ws}'...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with ConnectionPool(conninfo=DB_URI, max_size=5, open=True, kwargs={"autocommit": True}) as pool:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        engine = workflow.compile(checkpointer=checkpointer)

        inputs = {
            "workspace_id": active_ws,
            "question": user_query,
            "original_question": user_query,
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

        # HITL Interruption Checkpoint
        if state.next == ("human_approval",):
            st.warning("Action Required: Candidate answer paused at compliance approval gate.")
            candidate = state.values.get("generation", "")
            st.text_area("Candidate Output", value=candidate, height=130, disabled=True)

            col1, col2 = st.columns(2)
            if col1.button("Approve Output"):
                for _ in engine.stream(Command(resume="approved"), config=config):
                    pass
                final_state = engine.get_state(config)
                ans = final_state.values.get("generation", "")
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.rerun()
            if col2.button("Reject Output"):
                for _ in engine.stream(Command(resume="rejected"), config=config):
                    pass
                st.session_state.messages.append({"role": "assistant", "content": "[Rejected by reviewer]"})
                st.rerun()
        else:
            ans = state.values.get("generation", "No answer generated.")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.write(ans)

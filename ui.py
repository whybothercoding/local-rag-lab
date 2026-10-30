import streamlit as st
import uuid
import os
import shutil
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from app import workflow, DB_URI
from ingest import ingest_workspace

st.set_page_config(page_title="Local Knowledge Engine", layout="wide")

st.sidebar.title("Workspaces")
workspace = st.sidebar.selectbox("Active Domain", ["law_firm", "eshop", "personal_wiki", "devops"])

# File Upload / Ingestion in Sidebar
st.sidebar.subheader("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload .md or .txt files", accept_multiple_files=True)
if st.sidebar.button("Index Documents") and uploaded_files:
    upload_dir = f"./uploads/{workspace}"
    os.makedirs(upload_dir, exist_ok=True)
    for uf in uploaded_files:
        with open(os.path.join(upload_dir, uf.name), "wb") as f:
            f.write(uf.getbuffer())
    with st.spinner("Indexing into Chroma + BM25..."):
        ingest_workspace(workspace, upload_dir)
    st.sidebar.success("Ingestion complete!")

st.title(f"Knowledge Explorer: `{workspace}`")

# Session state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Query Input
user_query = st.chat_input("Ask a question about the active workspace...")

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
            "workspace_id": workspace,
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

        with st.spinner("Processing through guardrails, retrieval, and Self-RAG..."):
            for _ in engine.stream(inputs, config=config):
                pass

        state = engine.get_state(config)

        # Check for HITL pause
        if state.next == ("human_approval",):
            st.warning("Action Required: Generation paused for compliance approval.")
            candidate = state.values.get("generation", "")
            st.text_area("Candidate Response", value=candidate, height=150, disabled=True)
            
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
                st.rerun()
        else:
            ans = state.values.get("generation", "No answer produced.")
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.write(ans)

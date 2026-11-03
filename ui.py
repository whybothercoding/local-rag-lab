"""Streamlit entry point: `streamlit run ui.py` (or `make ui`).

The page lives in rag_lab/ui/streamlit_app.py. It is executed afresh on every Streamlit rerun, as a
script would be, and imported from the installed package so each module exists exactly once.
"""
import runpy

runpy.run_module("rag_lab.ui.streamlit_app", run_name="__main__")

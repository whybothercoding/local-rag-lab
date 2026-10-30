#!/usr/bin/env python3
import os
import logging
from langgraph.checkpoint.memory import MemorySaver

# Import the workflow state graph from app.py
from app import workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("GraphExporter")

OUTPUT_DIR = "./docs"
MERMAID_FILE = os.path.join(OUTPUT_DIR, "architecture.mmd")
PNG_FILE = os.path.join(OUTPUT_DIR, "architecture.png")


def export_mermaid_diagram():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Compile the graph with a dummy in-memory checkpointer for static analysis
    logger.info("Compiling StateGraph for topology extraction...")
    compiled_app = workflow.compile(checkpointer=MemorySaver())

    # 2. Extract Mermaid syntax using LangGraph's visual graph helper
    drawable_graph = compiled_app.get_graph()
    mermaid_code = drawable_graph.draw_mermaid()

    # 3. Write Mermaid syntax (.mmd) to disk
    logger.info(f"Writing Mermaid definition to {MERMAID_FILE}...")
    with open(MERMAID_FILE, "w", encoding="utf-8") as f:
        f.write(mermaid_code)

    print("\n" + "=" * 60)
    print(" MERMAID GRAPH EXPORTED SUCCESSFULLY")
    print("=" * 60)
    print(f"File: {MERMAID_FILE}\n")

    # 4. Attempt PNG export via Mermaid rendering engine
    try:
        logger.info(f"Rendering binary PNG to {PNG_FILE}...")
        png_bytes = drawable_graph.draw_mermaid_png()
        with open(PNG_FILE, "wb") as f:
            f.write(png_bytes)
        print(f"Rendered image saved: {PNG_FILE}")
    except Exception as exc:
        logger.warning(
            f"PNG rendering skipped: {exc}. "
            f"The .mmd file is available to paste directly into https://mermaid.live or GitHub Markdown."
        )


if __name__ == "__main__":
    export_mermaid_diagram()

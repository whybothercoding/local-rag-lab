#!/usr/bin/env python3
"""Export the LangGraph topology to docs/architecture.mmd and, when possible, docs/architecture.png.

The .mmd file is the source of truth and is written without network access.
PNG rendering uses a locally installed Mermaid CLI (`mmdc`, or the path in MERMAID_CLI).
No diagram is ever sent to a remote renderer.
"""
import logging
import os
import shutil
import subprocess

from langgraph.checkpoint.memory import MemorySaver

from app import workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("GraphExporter")

OUTPUT_DIR = "./docs"
MERMAID_FILE = os.path.join(OUTPUT_DIR, "architecture.mmd")
PNG_FILE = os.path.join(OUTPUT_DIR, "architecture.png")
MMDC_TIMEOUT_SEC = 120


def render_png_locally(source: str, target: str) -> bool:
    """Render the Mermaid file to PNG with a local CLI. Returns False if no CLI is installed."""
    mmdc = shutil.which(os.getenv("MERMAID_CLI", "mmdc"))
    if mmdc is None:
        logger.info(
            "PNG skipped: Mermaid CLI not found. Install with 'npm install -g @mermaid-js/mermaid-cli'. "
            "The .mmd file renders natively on GitHub without it."
        )
        return False
    subprocess.run([mmdc, "-i", source, "-o", target], check=True, timeout=MMDC_TIMEOUT_SEC)
    return True


def export_mermaid_diagram():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Compiling StateGraph for topology extraction...")
    compiled_app = workflow.compile(checkpointer=MemorySaver())
    mermaid_code = compiled_app.get_graph().draw_mermaid()

    logger.info(f"Writing Mermaid definition to {MERMAID_FILE}...")
    with open(MERMAID_FILE, "w", encoding="utf-8") as f:
        f.write(mermaid_code)

    print("\n" + "=" * 60)
    print(" MERMAID GRAPH EXPORTED SUCCESSFULLY")
    print("=" * 60)
    print(f"File: {MERMAID_FILE}\n")

    try:
        if render_png_locally(MERMAID_FILE, PNG_FILE):
            print(f"Rendered image saved: {PNG_FILE}")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"PNG rendering failed: {exc}. The .mmd file is still up to date.")


main = export_mermaid_diagram

if __name__ == "__main__":
    main()

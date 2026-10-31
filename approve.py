#!/usr/bin/env python3
import sys
import argparse
import logging
from langgraph.types import Command

# Import workflow and checkpointer factory
from app import workflow
from checkpointer import get_checkpointer, fetch_checkpoint_threads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HITL-Operator-CLI")


def list_pending_threads(app) -> list[str]:
    """Inspects checkpoints to find threads paused at an interrupt."""
    thread_ids = fetch_checkpoint_threads()
    pending = []

    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = app.get_state(config)
            # Check if the thread is halted at the human_approval node with an active interrupt
            if state.next == ("human_approval",) and state.tasks and any(t.interrupts for t in state.tasks):
                pending.append(thread_id)
        except Exception as exc:
            logger.debug("Failed inspecting thread %s: %s", thread_id, exc)

    return pending


def inspect_thread(app, thread_id: str):
    """Retrieves and prints the candidate payload held in Postgres."""
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)

    if state.next != ("human_approval",):
        print(f"\n[!] Thread '{thread_id}' is not awaiting approval. Current next node: {state.next}")
        return None

    # Extract pending interrupt payload
    interrupt_payload = {}
    for task in state.tasks:
        if task.interrupts:
            interrupt_payload = task.interrupts[0].value
            break

    print("\n" + "=" * 65)
    print(f" PENDING AUDIT FOR THREAD: {thread_id}")
    print("=" * 65)
    print(f"Action Type       : {interrupt_payload.get('action', 'N/A')}")
    print(f"Original Question : {interrupt_payload.get('question', state.values.get('original_question'))}")
    print(f"Citations         : {', '.join(interrupt_payload.get('citations', []))}")
    print("-" * 65)
    print("Candidate Generation to Review:\n")
    print(interrupt_payload.get('candidate_answer', state.values.get('generation')))
    print("=" * 65 + "\n")

    return state


def process_decision(app, thread_id: str, decision: str):
    """Resumes the graph from Postgres using the operator's decision."""
    config = {"configurable": {"thread_id": thread_id}}
    resume_cmd = Command(resume=decision.strip().lower())

    logger.info(f"Submitting '{decision}' decision to thread: {thread_id}...")
    for event in app.stream(resume_cmd, config=config):
        for node_name, state_update in event.items():
            logger.info(f"Post-approval Node Completed: [{node_name}]")

    final_state = app.get_state(config)
    print("\n" + "=" * 65)
    print(" EXECUTION RESUMED & COMPLETED")
    print("=" * 65)
    print(f"Final Emitted Output:\n{final_state.values.get('generation')}\n")


def main():
    parser = argparse.ArgumentParser(description="Human-in-the-Loop CLI Operator for Postgres Checkpoints")
    parser.add_argument("--list", action="store_true", help="List all threads awaiting human review")
    parser.add_argument("--thread", type=str, help="Thread ID to inspect or approve/reject")
    parser.add_argument("--action", type=str, choices=["approved", "rejected"], help="Decision action to apply")
    args = parser.parse_args()

    with get_checkpointer() as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)

        # Mode 1: List all pending threads
        if args.list:
            pending = list_pending_threads(app)
            print("\n--- THREADS AWAITING HUMAN APPROVAL ---")
            if not pending:
                print("No threads currently paused at [human_approval].")
            else:
                for idx, tid in enumerate(pending, 1):
                    print(f"{idx}. {tid}")
            print()
            return

        # Mode 2: Direct approval/rejection via CLI args
        if args.thread and args.action:
            state = inspect_thread(app, args.thread)
            if state:
                process_decision(app, args.thread, args.action)
            return

        # Mode 3: Interactive approval
        target_thread = args.thread
        if not target_thread:
            pending = list_pending_threads(app)
            if not pending:
                print("\nNo pending threads found in checkpoint database.")
                sys.exit(0)
            target_thread = pending[0]
            print(f"Auto-selected pending thread: {target_thread}")

        state = inspect_thread(app, target_thread)
        if not state:
            sys.exit(1)

        choice = input("Enter decision ('a' for approve / 'r' for reject / 'q' to abort): ").strip().lower()
        if choice in ["a", "approve", "approved"]:
            process_decision(app, target_thread, "approved")
        elif choice in ["r", "reject", "rejected"]:
            process_decision(app, target_thread, "rejected")
        else:
            print("Aborted without changes.")


if __name__ == "__main__":
    main()

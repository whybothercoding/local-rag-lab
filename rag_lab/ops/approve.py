#!/usr/bin/env python3
import sys
import argparse
import getpass
import logging
from langgraph.types import Command

# Import workflow and checkpointer factory
from rag_lab.security.access import (
    authorize,
    clearance_covers,
    get_user_clearance,
    issue_token,
    sign_in,
    verify_token,
)
from rag_lab.pipeline.graph import workflow
from rag_lab.storage.checkpointer import get_checkpointer, fetch_checkpoint_threads, shared_pool

logger = logging.getLogger("HITL-Operator-CLI")

MAX_SIGN_IN_ATTEMPTS = 3


def list_pending_threads(
    app,
    viewer_clearance: str | None = None,
    prefix: str | None = None,
) -> list[str]:
    """Inspects checkpoints to find threads paused at an interrupt.

    When `viewer_clearance` is given, threads whose requester holds a higher clearance
    than the viewer are excluded so lower-clearance reviewers cannot browse them.
    """
    pool = shared_pool(getattr(app, "checkpointer", None))
    thread_ids = fetch_checkpoint_threads(prefix=prefix, pool=pool) if prefix is not None else fetch_checkpoint_threads(pool=pool)
    pending = []

    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = app.get_state(config)
            # Check if the thread is halted at the human_approval node with an active interrupt
            if state.next == ("human_approval",) and state.tasks and any(t.interrupts for t in state.tasks):
                if viewer_clearance is not None:
                    requester = state.values.get("principal", "")
                    req_clr = state.values.get("principal_clearance") or get_user_clearance(requester)
                    if not clearance_covers(viewer_clearance, req_clr):
                        continue
                pending.append(thread_id)
        except Exception as exc:
            logger.debug("Failed inspecting thread %s: %s", thread_id, exc)

    return pending


def inspect_thread(app, thread_id: str, viewer_clearance: str | None = None):
    """Retrieves and prints the candidate payload held in the checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)

    if state.next != ("human_approval",):
        print(f"\n[!] Thread '{thread_id}' is not awaiting approval. Current next node: {state.next}")
        return None

    requester = state.values.get("principal", "")
    req_clearance = state.values.get("principal_clearance") or get_user_clearance(requester)
    if viewer_clearance is not None and not clearance_covers(viewer_clearance, req_clearance):
        print(
            f"\n[!] Clearance violation: viewer clearance '{viewer_clearance}' cannot inspect "
            f"thread '{thread_id}' (requires '{req_clearance}')."
        )
        return None

    # Extract pending interrupt payload
    interrupt_payload = {}
    for task in state.tasks:
        if task.interrupts:
            interrupt_payload = task.interrupts[0].value
            break

    citations_list = interrupt_payload.get("citations") or state.values.get("citations") or []
    degraded_list = state.values.get("degraded") or []
    reasons_list = interrupt_payload.get("review_reasons") or []

    print("\n" + "=" * 65)
    print(f" PENDING AUDIT FOR THREAD: {thread_id}")
    print("=" * 65)
    print(f"Action Type       : {interrupt_payload.get('action', 'N/A')}")
    print(f"Workspace         : {state.values.get('workspace_id', 'N/A')}")
    print(f"Requested By      : {requester or 'N/A'} (clearance: {req_clearance})")
    print(f"Original Question : {interrupt_payload.get('question', state.values.get('original_question'))}")
    print(f"Citations         : {', '.join(citations_list) if citations_list else 'none'}")
    if reasons_list:
        print(f"Gate Reasons      : {', '.join(reasons_list)}")
    if degraded_list:
        print(f"Degraded Warnings : {'; '.join(degraded_list)}")
    print("-" * 65)
    print("Candidate Generation to Review:\n")
    print(interrupt_payload.get('candidate_answer', state.values.get('generation')))
    print("=" * 65 + "\n")

    return state


def process_decision(app, thread_id: str, decision: str, approver_token: str, reason: str = ""):
    """Resumes the graph with the operator's decision. The reviewer is identified by a signed session token."""
    config = {"configurable": {"thread_id": thread_id}}
    payload = {
        "decision": decision.strip().lower(),
        "approver_token": approver_token,
        "reason": reason.strip(),
    }
    resume_cmd = Command(resume=payload)

    reviewer = verify_token(approver_token) or "unknown"
    logger.info(f"Submitting '{payload['decision']}' by {reviewer} to thread: {thread_id}...")
    for event in app.stream(resume_cmd, config=config):
        for node_name, state_update in event.items():
            logger.info(f"Post-approval Node Completed: [{node_name}]")

    final_state = app.get_state(config)
    print("\n" + "=" * 65)
    print(" EXECUTION RESUMED & COMPLETED")
    print("=" * 65)
    print(f"Final Emitted Output:\n{final_state.values.get('generation')}\n")


def confirm_reviewer(state, preset: str | None) -> str:
    """Sign the reviewer in, check their review grant on this thread's workspace, and return a session token."""
    workspace = state.values.get("workspace_id", "")
    print("\nTo record this decision, sign in with the reviewer account.")
    print("Enter the account name and the password set for that account.")
    print("Forgot the password? An administrator can reset it with: make set-password ACCOUNT=<name>")
    print("Unsure of the account name? List accounts with: make users")
    print("Press Ctrl-C at any time to cancel. Nothing is recorded and the query stays in the queue.\n")

    try:
        name = (preset or input("Account name: ")).strip().lower()
        if not name:
            sys.exit("Account name is required. No decision recorded.")
        for attempt in range(1, MAX_SIGN_IN_ATTEMPTS + 1):
            password = getpass.getpass(f"Password for '{name}': ")
            result = sign_in(name, password)
            if result.ok:
                break
            if result.reason == "locked":
                sys.exit(f"{result.describe()} No decision recorded.")
            remaining = MAX_SIGN_IN_ATTEMPTS - attempt
            if remaining == 0:
                sys.exit("Sign-in failed too many times. No decision recorded.")
            print(f"{result.describe()} Attempts left: {remaining}.")
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nCancelled. No decision recorded; the query is still waiting for review.")

    if not authorize(result.username, workspace, "review"):
        sys.exit(
            f"'{result.username}' signed in but has no review permission on workspace '{workspace}'. "
            "No decision recorded."
        )
    requester = state.values.get("principal", "")
    if requester and result.username == requester:
        sys.exit(
            f"Separation of Duties: '{result.username}' submitted this query and cannot review their own request. "
            "No decision recorded; the query remains in the queue for another reviewer."
        )
    req_clearance = state.values.get("principal_clearance") or get_user_clearance(requester)
    reviewer_clearance = get_user_clearance(result.username)
    if not clearance_covers(reviewer_clearance, req_clearance):
        sys.exit(
            f"Clearance Violation: '{result.username}' [{reviewer_clearance}] cannot review a "
            f"'{req_clearance}' query. No decision recorded."
        )
    return issue_token(result.username)


def main():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    parser = argparse.ArgumentParser(description="Human-in-the-Loop CLI Operator for Checkpoints")
    parser.add_argument("--list", action="store_true", help="List all threads awaiting human review")
    parser.add_argument("--thread", type=str, help="Thread ID to inspect or approve/reject")
    parser.add_argument("--action", type=str, choices=["approved", "rejected"], help="Decision action to apply")
    parser.add_argument("--approver", type=str, help="Reviewer account; you will be asked for its password")
    parser.add_argument("--reason", type=str, default="", help="Optional reason recorded in the audit log")
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
            if not state:
                sys.exit(1)
            approver_token = confirm_reviewer(state, args.approver)
            process_decision(app, args.thread, args.action, approver_token, args.reason)
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

        approver_token = confirm_reviewer(state, args.approver)

        if args.action:
            process_decision(app, target_thread, args.action, approver_token, args.reason)
            return

        choice = input("Enter decision ('a' for approve / 'r' for reject / 'q' to abort): ").strip().lower()
        if choice in ["a", "approve", "approved"]:
            process_decision(app, target_thread, "approved", approver_token, args.reason)
        elif choice in ["r", "reject", "rejected"]:
            process_decision(app, target_thread, "rejected", approver_token, args.reason)
        else:
            print("Aborted without changes.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from src.core.context import CheckpointStore
from src.core.loop import HarnessLoop
from src.core.promotion import PromotionService

LoopFactory = Callable[[Path, Path], HarnessLoop]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataops-harness",
        description="CLI for inspecting Run checkpoints and executing HiTL resume approvals/rejections.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show latest checkpoint status of a Run.")
    status_parser.add_argument("run_id", help="Run identifier.")
    status_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))

    resume_parser = subparsers.add_parser(
        "resume", help="Approve (promote) or reject (re-run) a Run awaiting approval."
    )
    resume_parser.add_argument("run_id", help="Run identifier.")
    resume_parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    resume_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))

    action_group = resume_parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--approve",
        action="store_true",
        help="Promote approved shadow fix into sandbox git and production lakehouse.",
    )
    action_group.add_argument(
        "--reject",
        action="store_true",
        help="Reject proposed fix, inject feedback, and grant turn extension.",
    )
    resume_parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Rejection feedback reason (required with --reject).",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    loop_factory: LoopFactory | None = None,
) -> int:
    """Entrypoint for HiTL status and resume CLI operations."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "status":
            store = CheckpointStore(args.runs_dir, args.run_id)
            state = store.load_latest()
            payload = {
                "run_id": state.run_id,
                "status": state.status,
                "turn": state.turn,
                "max_turns": state.max_turns,
                "rejection_count": state.rejection_count,
                "patched_files": state.patched_files,
                "has_approval_packet": "approval_packet" in state.metadata,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "resume":
            service = PromotionService(args.workspace, args.runs_dir)
            if args.approve:
                result = service.approve(args.run_id)
                print(result.model_dump_json(indent=2))
                return 0

            if not args.reason.strip():
                raise ValueError("--reason must be provided with --reject.")
            loop = loop_factory(args.workspace, args.runs_dir) if loop_factory else None
            updated = service.reject(args.run_id, args.reason, loop=loop)
            print(
                json.dumps(
                    {
                        "run_id": updated.run_id,
                        "status": updated.status,
                        "turn": updated.turn,
                        "max_turns": updated.max_turns,
                        "rejection_count": updated.rejection_count,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

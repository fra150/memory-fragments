"""Command-line interface for Memory Fragments V2.

Allows exercising the full pipeline (ingest -> query -> appeal -> governance)
without external consumers::

    python -m memory_fragments init
    python -m memory_fragments ingest --content "..." --topic "physics"
    python -m memory_fragments query --text "sunlight"
    python -m memory_fragments appeal --appeal-id A1 --sources f1 --content "..."
    python -m memory_fragments approve --appeal-id A1
    python -m memory_fragments status

State can be persisted to a JSON file with ``--state PATH`` so that the
workflow can span multiple invocations.  Without ``--state`` every command
starts from an empty in-memory store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from memory_fragments import __version__
from memory_fragments.config import default_config
from memory_fragments.governance.api import GovernanceAPI
from memory_fragments.library.cassetto import Cassetto, CassettoConfig
from memory_fragments.models import (
    Appeal,
    AppealMetrics,
    AppealOperation,
    AppealStatus,
    Fragment,
    FragmentMetadata,
    FragmentStatus,
    OperationType,
)
from memory_fragments.models.graph import GenealogyGraph
from memory_fragments.models.quality import QualitySource


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _new_cassetto(name: str) -> Cassetto:
    """Build a fresh Cassetto with a deterministic name."""
    return Cassetto(CassettoConfig(name=name, topic=""))


def _load_cassetto(state_path: Optional[str]) -> Cassetto:
    """Restore a Cassetto from a JSON state file, or build a new one."""
    if state_path and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        config = data.get("config", {})
        cassetto = _new_cassetto(config.get("name", "main"))

        cassetto.archive = __import__(
            "memory_fragments.archive.static", fromlist=["StaticArchive"]
        ).StaticArchive.from_dict(data.get("archive", {"fragments": []}))
        cassetto.appeal_space = __import__(
            "memory_fragments.archive.appeal_space", fromlist=["AppealTrialSpace"]
        ).AppealTrialSpace.from_dict(data.get("appeal_space", {"appeals": []}))
        cassetto.retriever.rebuild(cassetto.archive.list_all())

        graph = GenealogyGraph.from_dict(data.get("genealogy", {"nodes": {}}))
        cassetto._genealogy = graph  # noqa: SLF001 — CLI internals
        return cassetto

    return _new_cassetto("main")


def _save_state(cassetto: Cassetto, state_path: Optional[str]) -> None:
    """Persist the Cassetto state to a JSON file."""
    if not state_path:
        return
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(cassetto.to_dict(), fh, indent=2, default=str)
    print(f"State saved to {state_path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_version(_args: argparse.Namespace, _cassetto: Cassetto) -> None:
    print(f"memory-fragments {__version__}")


def cmd_init(args: argparse.Namespace, cassetto: Cassetto) -> None:
    print(f"Cassetto '{cassetto.config.name}' initialised (empty).")
    _save_state(cassetto, args.state)


def cmd_ingest(args: argparse.Namespace, cassetto: Cassetto) -> None:
    fragment = Fragment(
        fragment_id=args.id or f"f-{cassetto.count() + 1}",
        content=args.content,
        metadata=FragmentMetadata(
            topic=args.topic or cassetto.config.topic,
            source=args.source or "cli",
            quality=args.quality,
            author=args.author or "cli-user",
            tags=args.tags or [],
        ),
    )
    accepted = cassetto.add(fragment)
    if accepted:
        print(f"Fragment '{fragment.fragment_id}' accepted (quality={fragment.metadata.quality}).")
    else:
        print(f"Fragment '{fragment.fragment_id}' REJECTED (quality={fragment.metadata.quality}).")
    _save_state(cassetto, args.state)


def cmd_query(args: argparse.Namespace, cassetto: Cassetto) -> None:
    results = cassetto.search(args.text, top_k=args.top_k)
    if not results:
        print("No results.")
        return
    for fragment, score in results:
        print(
            f"[{score:.3f}] {fragment.fragment_id} ({fragment.metadata.topic or 'no topic'}): "
            f"{fragment.content[:100]}"
        )


def cmd_compose(args: argparse.Namespace, cassetto: Cassetto) -> None:
    text = cassetto.compose(args.text, top_k=args.top_k)
    print(text)


def cmd_appeal(args: argparse.Namespace, cassetto: Cassetto) -> None:
    """Create an Appeal, evaluate it, and optionally submit it for review."""
    if cassetto.appeal_space.get(args.appeal_id):
        print(f"Error: appeal '{args.appeal_id}' already exists.", file=sys.stderr)
        sys.exit(1)

    sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()]
    missing = [s for s in sources if cassetto.archive.get(s) is None]
    if missing:
        print(
            f"Error: source fragment(s) not found in archive: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    ops = []
    if args.operation:
        try:
            op_type = OperationType(args.operation)
        except ValueError:
            print(
                f"Error: invalid operation '{args.operation}'. "
                f"Valid: {[o.value for o in OperationType]}",
                file=sys.stderr,
            )
            sys.exit(1)
        ops = [AppealOperation(op_type=op_type, description=f"CLI {op_type.value}")]

    cassetto.create_appeal(args.appeal_id, sources, ops)
    cassetto.appeal_space.update_proposal(
        args.appeal_id,
        proposed_content=args.content,
        explanation=args.explanation,
    )

    governance = cassetto.governance()
    report = governance.get_report(args.appeal_id)
    print(
        f"Appeal '{args.appeal_id}' created "
        f"(sources={sources}, coverage={report.metrics.coverage:.2f}, "
        f"risk={report.metrics.risk:.2f}, aggregate={report.metrics.aggregate_score:.2f})."
    )
    print(f"Explanation: {report.explanation}")

    if args.submit:
        governance.submit_for_review(args.appeal_id)
        print(f"Appeal '{args.appeal_id}' submitted for review.")
    _save_state(cassetto, args.state)


def cmd_approve(args: argparse.Namespace, cassetto: Cassetto) -> None:
    governance = cassetto.governance()
    fragment = governance.approve(args.appeal_id, approver=args.approver, notes=args.notes)
    print(f"Appeal '{args.appeal_id}' approved -> fragment '{fragment.fragment_id}' stored.")
    _save_state(cassetto, args.state)


def cmd_reject(args: argparse.Namespace, cassetto: Cassetto) -> None:
    governance = cassetto.governance()
    governance.reject(args.appeal_id, reason=args.reason)
    print(f"Appeal '{args.appeal_id}' rejected.")
    _save_state(cassetto, args.state)


def cmd_status(args: argparse.Namespace, cassetto: Cassetto) -> None:
    governance = cassetto.governance()
    stats = governance.get_statistics()
    print(f"Cassetto '{cassetto.config.name}':")
    print(f"  Fragments (archive):    {cassetto.count()}")
    print(f"  Rejected by guardian:   {cassetto.rejected_count()}")
    print(f"  Appeals (total):        {stats['total_appeals']}")
    print(f"    approved:             {stats['approved']}")
    print(f"    rejected:             {stats['rejected']}")
    print(f"    pending:              {stats['pending']}")
    print(f"    draft:                {stats['draft']}")
    print(f"  Quarantine:             {stats['quarantine_size']}")


def cmd_demo(args: argparse.Namespace, _cassetto: Cassetto) -> None:
    """Run a self-contained end-to-end demo (ingest -> query -> appeal -> approve)."""
    cassetto = _new_cassetto("demo")
    seed = [
        ("d1", "Il Sole converte l'idrogeno in elio attraverso la fusione nucleare.", "physics", 0.9),
        ("d2", "La fotosintesi usa la luce solare per produrre glucosio e ossigeno.", "biology", 0.85),
        ("d3", "Il teorema di Pitagora lega i cateti e l'ipotenusa di un triangolo rettangolo.", "math", 0.95),
    ]
    for fid, content, topic, quality in seed:
        fragment = Fragment(
            fragment_id=fid,
            content=content,
            metadata=FragmentMetadata(topic=topic, quality=quality, source="demo"),
        )
        cassetto.add(fragment)
    print(f"[demo] Ingested {cassetto.count()} fragments.")

    results = cassetto.search("luce del sole energia", top_k=2)
    print(f"[demo] Query 'luce del sole energia' -> {len(results)} result(s).")

    cassetto.appeal_space.create_appeal("demo-appeal", ["d2"])
    cassetto.appeal_space.update_proposal(
        "demo-appeal",
        proposed_content=(
            "La fotosintesi usa la luce solare per produrre glucosio e ossigeno, "
            "un processo fondamentale per la vita sulla Terra."
        ),
        explanation="Demo appeal: aggiunge contesto ecologico al frammento d2.",
    )
    governance = cassetto.governance()
    governance.submit_for_review("demo-appeal")
    fragment = governance.approve("demo-appeal", approver="demo")
    print(f"[demo] Appeal approved -> fragment '{fragment.fragment_id}' stored.")
    print(f"[demo] Final archive count: {cassetto.count()}.")
    _save_state(cassetto, args.state)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory-fragments",
        description="Memory Fragments V2 — modular cognitive model CLI.",
    )
    parser.add_argument(
        "--state",
        metavar="PATH",
        default=None,
        help="JSON file to load/save Cassetto state across invocations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the package version.")
    sub.add_parser("init", help="Initialise a (possibly empty) Cassetto state.")

    p_ingest = sub.add_parser("ingest", help="Add a fragment through the quality guardian.")
    p_ingest.add_argument("--content", required=True, help="Fragment content.")
    p_ingest.add_argument("--id", default=None, help="Fragment ID (auto-generated if omitted).")
    p_ingest.add_argument("--topic", default=None, help="Fragment topic.")
    p_ingest.add_argument("--source", default=None, help="Provenance source label.")
    p_ingest.add_argument("--author", default=None, help="Author of the fragment.")
    p_ingest.add_argument("--quality", type=float, default=0.9, help="Claimed quality in [0,1].")
    p_ingest.add_argument("--tags", nargs="*", default=None, help="Space-separated tags.")

    p_query = sub.add_parser("query", help="Hybrid (BM25 + embedding) search.")
    p_query.add_argument("--text", required=True, help="Search query.")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of results.")

    p_compose = sub.add_parser("compose", help="Compose a response from matching fragments.")
    p_compose.add_argument("--text", required=True, help="Composition query.")
    p_compose.add_argument("--top-k", type=int, default=5, help="Number of source fragments.")

    p_appeal = sub.add_parser("appeal", help="Create and evaluate an Appeal.")
    p_appeal.add_argument("--appeal-id", required=True, help="Unique appeal ID.")
    p_appeal.add_argument("--sources", default="", help="Comma-separated source fragment IDs.")
    p_appeal.add_argument("--content", required=True, help="Proposed content.")
    p_appeal.add_argument("--explanation", default="No explanation provided.", help="Why this change.")
    p_appeal.add_argument(
        "--operation",
        choices=[o.value for o in OperationType],
        default=None,
        help="Optional transformation operation.",
    )
    p_appeal.add_argument("--submit", action="store_true", help="Submit for review immediately.")

    p_approve = sub.add_parser("approve", help="Approve a pending appeal.")
    p_approve.add_argument("--appeal-id", required=True)
    p_approve.add_argument("--approver", default="cli-user")
    p_approve.add_argument("--notes", default="")

    p_reject = sub.add_parser("reject", help="Reject a pending appeal.")
    p_reject.add_argument("--appeal-id", required=True)
    p_reject.add_argument("--reason", default="")

    sub.add_parser("status", help="Show archive and governance statistics.")
    sub.add_parser("demo", help="Run a self-contained end-to-end demo.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``memory-fragments`` and ``python -m memory_fragments``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    commands = {
        "version": cmd_version,
        "init": cmd_init,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "compose": cmd_compose,
        "appeal": cmd_appeal,
        "approve": cmd_approve,
        "reject": cmd_reject,
        "status": cmd_status,
        "demo": cmd_demo,
    }

    # Commands that build their own Cassetto from scratch.
    if args.command in ("version", "demo"):
        commands[args.command](args, None)  # type: ignore[arg-type]
        return 0

    cassetto = _load_cassetto(args.state)
    try:
        commands[args.command](args, cassetto)
    except SystemExit as exc:
        # Commands may abort with sys.exit(1) on validation errors.
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

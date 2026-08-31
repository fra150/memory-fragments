#!/usr/bin/env python3
"""
Memory Fragments V2 — Demo Completa del Flusso
===============================================

Questa demo esercita il flusso completo V2 tramite l'orchestratore
:class:`memory_fragments.model.MemoryFragmentsModel`:

1. Ingest  — frammenti protetti dal FragmentGuardian
2. Query   — retrieval ibrido (BM25 + embedding) e composizione
3. Appeal  — proposta di modifica nel trial space con metriche automatiche
4. Governance — submit → human approval/rejection (human-in-the-loop)
5. Export  — statistiche aggregate e stato serializzato del modello

Prerequisiti:
- ``pip install -e .``
- Opzionale: ``pip install sentence-transformers`` per embedding reali.
  Senza, la demo usa embeddings deterministici di fallback (offline).

Esecuzione:
    python examples/run_demo_v2.py            # offline (default)
    python examples/run_demo_v2.py --online   # usa sentence-transformers se disponibile
"""

import argparse
import sys
from pathlib import Path

# Aggiungi la root del package al path se necessario
workspace = Path(__file__).resolve().parent.parent
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))

from memory_fragments.model import MemoryFragmentsModel
from memory_fragments.models import (
    AppealOperation,
    Fragment,
    FragmentMetadata,
    OperationType,
)


def _force_offline(model: MemoryFragmentsModel) -> None:
    """Forza embeddings di fallback (nessun download / chiamata di rete)."""
    model.retriever.embedding._fallback = True


def demo_ingest(model: MemoryFragmentsModel):
    """Fase 1 — Ingest di frammenti con il guardian della qualità."""
    print("\n" + "=" * 60)
    print("FASE 1: INGEST DI UN FRAGMENT")
    print("=" * 60)

    fragments = [
        Fragment(
            fragment_id="demo-fotosintesi-001",
            content=(
                "La fotosintesi clorofilliana è il processo attraverso cui "
                "le piante convertono la luce solare in energia chimica."
            ),
            metadata=FragmentMetadata(
                source="wiki/biologia/fotosintesi.md",
                topic="biologia",
                quality=0.92,
                tags=["fotosintesi", "piante", "energia"],
            ),
        ),
        Fragment(
            fragment_id="demo-fotosintesi-002",
            content=(
                "Durante la fotosintesi vengono prodotti glucosio e ossigeno "
                "come sottoprodotti a partire da anidride carbonica e acqua."
            ),
            metadata=FragmentMetadata(
                source="wiki/biologia/fotosintesi.md",
                topic="biologia",
                quality=0.90,
                tags=["fotosintesi", "glucosio", "ossigeno"],
            ),
        ),
    ]

    for fragment in fragments:
        result = model.ingest(fragment)
        if not result.ok:
            print(f"  RIFIUTATO {fragment.fragment_id}: {result.reason}")
            continue
        print(f"  Accettato: {fragment.fragment_id}")
        print(f"    - Topic: {fragment.metadata.topic}")
        print(f"    - Tags:  {', '.join(fragment.metadata.tags)}")
        print(f"    - Qualità: {fragment.metadata.quality:.2f}")

    return fragments


def demo_query(model: MemoryFragmentsModel):
    """Fase 2 — Retrieval ibrido (BM25 + embedding) e composizione."""
    print("\n" + "=" * 60)
    print("FASE 2: QUERY — RETRIEVAL IBRIDO (BM25 + EMBEDDING)")
    print("=" * 60)

    query = "come le piante producono energia dalla luce solare?"
    result = model.query(query, top_k=3)

    print(f"  Query: '{query}'")
    print(f"  Frammenti rilevanti: {len(result.fragments)}")
    for i, (frag, score) in enumerate(zip(result.fragments, result.scores), 1):
        print(f"    [{i}] score={score:.4f}  id={frag.fragment_id}")
        print(f"        preview: {frag.content[:70]}...")

    print("\n  Risposta composta:")
    for line in result.response.splitlines():
        print(f"    {line}")

    return result


def demo_appeal(model: MemoryFragmentsModel, fragment: Fragment):
    """Fase 3 — Appeal trial: proposta di modifica con metriche automatiche."""
    print("\n" + "=" * 60)
    print("FASE 3: APPEAL TRIAL (SANDBOX MODIFICHE)")
    print("=" * 60)

    modified_content = (
        "La fotosintesi clorofilliana è il processo attraverso cui le piante "
        "convertono la luce solare in energia chimica, producendo glucosio e "
        "ossigeno come sottoprodotti."
    )

    appeal = model.propose(
        appeal_id="A-demo-001",
        source_ids=[fragment.fragment_id],
        proposed_content=modified_content,
        explanation="Aggiunta dei sottoprodotti (glucosio e ossigeno) alla descrizione.",
        ops=[AppealOperation(op_type=OperationType.ANNOTATE)],
    )

    print(f"  Appeal creato: {appeal.appeal_id}")
    print(f"    - Sorgenti: {appeal.sources}")
    print(f"    - Status:   {appeal.status.value}")
    print(f"    - Metriche:")
    print(f"        delta_token    = {appeal.metrics.delta_token}")
    print(f"        coverage       = {appeal.metrics.coverage:.2%}")
    print(f"        risk           = {appeal.metrics.risk:.2%}")
    print(f"        aggregate_score= {appeal.metrics.aggregate_score:.4f}")
    print(f"    - Diff (aggiunte): {', '.join(appeal.diff.added[:5]) or '-'}")

    return appeal


def demo_governance(model: MemoryFragmentsModel, appeal):
    """Fase 4 — Governance human-in-the-loop: submit → approve."""
    print("\n" + "=" * 60)
    print("FASE 4: GOVERNANCE HUMAN-IN-THE-LOOP")
    print("=" * 60)

    report = model.submit(appeal.appeal_id)
    print(f"  Inviato per revisione: {report.appeal_id}")
    print(f"    - Status: {report.status.value}")

    risk_threshold = model.evaluator.config.risk_threshold
    auto_ok = appeal.metrics.risk < risk_threshold
    print(f"  Soglia di rischio: {risk_threshold:.2%}")
    print(f"  Rischio calcolato: {appeal.metrics.risk:.2%}")
    print(f"  Rischio sotto soglia (auto-approvabile): {auto_ok}")

    approved = model.approve(
        appeal.appeal_id,
        approver="reviewer",
        notes="Modifica coerente con le fonti.",
    )
    print(f"  Approvato! Nuovo frammento: {approved.fragment_id}")
    print(f"    - Parents: {approved.parents}")
    print(f"    - Qualità: {approved.metadata.quality:.2f}")

    return approved


def demo_statistics(model: MemoryFragmentsModel):
    """Fase 5 — Statistiche aggregate e stato esportabile."""
    print("\n" + "=" * 60)
    print("FASE 5: STATISTICHE E EXPORT DELLO STATO")
    print("=" * 60)

    stats = model.get_statistics()
    model_stats = stats.get("model", {})
    print(f"  Archive fragments : {model_stats.get('archive_count')}")
    print(f"  Appeal attivi     : {model_stats.get('active_appeals')}")
    print(f"  Nodi genealogy    : {model_stats.get('graph_nodes')}")

    state = model.export_state()
    print(f"  Stato esportato   : schema={state['schema']} version={state['version']}")

    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory Fragments V2 — demo del flusso completo."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="tenta di usare sentence-transformers per gli embedding reali",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MEMORY FRAGMENTS V2 — DEMO DEL FLUSSO COMPLETO")
    print("=" * 60)
    print("  1. Ingest -> 2. Query -> 3. Appeal -> 4. Governance -> 5. Export")

    try:
        model = MemoryFragmentsModel()
        if not args.online:
            _force_offline(model)
            print("  [offline] embeddings di fallback deterministici attivi")

        fragments = demo_ingest(model)
        demo_query(model)
        appeal = demo_appeal(model, fragments[0])
        demo_governance(model, appeal)
        demo_statistics(model)

        print("\n" + "=" * 60)
        print("DEMO COMPLETATA CON SUCCESSO")
        print("=" * 60)
        print("  Riferimenti:")
        print("    - Paper V1: doi:10.5281/zenodo.14534720")
        print("    - Paper V2: doi:10.5281/zenodo.17069503")
        print("    - GitHub  : https://github.com/fra150/memory-fragments")
        return 0

    except Exception as exc:  # noqa: BLE001 - la demo deve riportare l'errore
        print(f"\nERRORE DURANTE LA DEMO: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

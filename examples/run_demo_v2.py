#!/usr/bin/env python3
"""
Memory Fragments V2 — Demo Completa del Flusso

Questa demo esercita il flusso completo V2:
1. Ingest di documenti grezzi
2. Retrieval ibrido (BM25 + embedding)
3. Appeal trial per modifiche proposte
4. Governance human-in-the-loop

Prerequisiti:
- pip install -e .
- Documenti di esempio in /raw/ (opzionale)
"""

import sys
from pathlib import Path

# Aggiungi il workspace al path se necessario
workspace = Path(__file__).resolve().parent
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))

from memory_fragments.config import default_config
from memory_fragments.models.fragment import Fragment as MemoryFragment
from memory_fragments.models.appeal import Appeal
from memory_fragments.archive import AppealTrialSpace
from memory_fragments.models.graph import GenealogyGraph
from memory_fragments.engine.evaluator import Evaluator
from memory_fragments.retrieval.retriever import HybridRetriever
from memory_fragments.library.guardian import FragmentGuardian
from memory_fragments.governance.api import GovernanceAPI


def demo_ingest():
    """Dimostra l'ingest di un fragment."""
    print("\n" + "="*60)
    print("📥 FASE 1: INGEST DI UN FRAGMENT")
    print("="*60)
    
    from memory_fragments.models.fragment import FragmentMetadata
    
    # Crea un fragment di esempio
    fragment = MemoryFragment(
        fragment_id="demo-fotosintesi-001",
        content="La fotosintesi clorofilliana è il processo attraverso cui le piante convertono luce solare in energia chimica.",
        metadata=FragmentMetadata(
            source="wiki/biologia/fotosintesi.md",
            topic="biologia",
            tags=["fotosintesi", "piante", "energia", "biologia"]
        )
    )
    
    print(f"✓ Fragment creato:")
    print(f"  - ID: {fragment.fragment_id}")
    print(f"  - Topic: {fragment.metadata.topic}")
    print(f"  - Tags: {fragment.metadata.tags}")
    print(f"  - Content length: {len(fragment.content)} chars")
    
    return fragment


def demo_retrieval(fragment: MemoryFragment):
    """Dimostra il retrieval ibrido."""
    print("\n" + "="*60)
    print("🔍 FASE 2: RETRIEVAL IBRIDO (BM25 + Embedding)")
    print("="*60)
    
    # Crea un retriever ibrido
    retriever = HybridRetriever()
    
    # Aggiungi il fragment all'index
    retriever.add_fragment(fragment)
    
    # Esegui una query
    query = "come le piante producono energia dalla luce solare?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"✓ Query: '{query}'")
    print(f"✓ Risultati trovati: {len(results)}")
    
    for i, result in enumerate(results, 1):
        # results è una lista di tuple (Fragment, score)
        frag, score = result
        print(f"\n  [{i}] Score: {score:.4f}")
        print(f"      Fragment ID: {frag.fragment_id}")
        print(f"      Preview: {frag.content[:80]}...")
    
    return retriever, results


def demo_appeal(fragment: MemoryFragment):
    """Dimostra l'appeal trial per modifiche."""
    print("\n" + "="*60)
    print("⚖️  FASE 3: APPEAL TRIAL (SANDBOX MODIFICHE)")
    print("="*60)
    
    # Crea uno spazio di appeal
    trial_space = AppealTrialSpace()
    
    # Proponi una modifica al fragment
    modified_content = "La fotosintesi clorofilliana è il processo attraverso cui le piante convertono luce solare in energia chimica, producendo glucosio e ossigeno come sottoprodotti."
    
    diff_proposed = {
        "field": "content",
        "old_value": fragment.content,
        "new_value": modified_content,
        "reason": "Aggiunta informazioni sui sottoprodotti della fotosintesi"
    }
    
    # Avvia la sessione di appeal
    appeal = trial_space.start_edit_session(fragment, diff_proposed)
    
    print(f"✓ Appeal avviato:")
    print(f"  - Appeal ID: {appeal.id}")
    print(f"  - Status: {appeal.status}")
    print(f"  - Reason: {diff_proposed['reason']}")
    print(f"  - Modifica proposta: campo '{diff_proposed['field']}'")
    
    # Valuta la modifica con l'evaluator
    evaluator = Evaluator()
    metrics = evaluator.evaluate([fragment], [modified_content])
    
    print(f"\n✓ Metriche di valutazione:")
    print(f"  - Delta token: {metrics.delta_token}")
    print(f"  - Coverage: {metrics.coverage:.2%}")
    print(f"  - Risk score: {metrics.hallucination_risk:.2%}")
    
    return trial_space, appeal, metrics


def demo_governance(appeal: Appeal, metrics):
    """Dimostra la governance human-in-the-loop."""
    print("\n" + "="*60)
    print("👥 FASE 4: GOVERNANCE HUMAN-IN-THE-LOOP")
    print("="*60)
    
    # Crea l'API di governance
    governance = GovernanceAPI()
    
    # Determina se la modifica può essere auto-approvata
    risk_threshold = default_config.evaluator.risk_threshold
    auto_approve = metrics.hallucination_risk < risk_threshold
    
    print(f"✓ Soglia di rischio configurata: {risk_threshold:.2%}")
    print(f"✓ Rischio calcolato: {metrics.hallucination_risk:.2%}")
    print(f"✓ Auto-approvazione possibile: {auto_approve}")
    
    if auto_approve:
        print(f"\n✓ La modifica può essere auto-approvata (rischio < soglia)")
        # Simula l'approvazione automatica
        approved = True
        decision = "auto_approved"
    else:
        print(f"\n⚠️  La modifica richiede revisione umana (rischio >= soglia)")
        # Simula la richiesta di revisione umana
        print(f"  - Pending review in governance queue")
        approved = False
        decision = "pending_human_review"
    
    print(f"\n✓ Decisione: {decision}")
    
    return governance, approved, decision


def demo_genealogy(fragment: MemoryFragment, approved: bool):
    """Dimostra il tracciamento genealogico (DAG)."""
    print("\n" + "="*60)
    print("🌳 FASE 5: GENEALOGY GRAPH (DAG VERSIONING)")
    print("="*60)
    
    # Crea il grafo genealogico
    graph = GenealogyGraph()
    
    # Aggiungi il fragment originale
    graph.add_fragment(fragment)
    
    # Se approvato, crea una nuova versione
    if approved:
        modified_content = "La fotosintesi clorofilliana è il processo attraverso cui le piante convertono luce solare in energia chimica, producendo glucosio e ossigeno come sottoprodotti."
        
        new_fragment = MemoryFragment(
            content=modified_content,
            source=fragment.source,
            topic=fragment.topic,
            tags=fragment.tags,
            parent_id=fragment.id
        )
        
        graph.add_fragment(new_fragment)
        
        print(f"✓ Nuova versione creata:")
        print(f"  - Parent: {fragment.id}")
        print(f"  - Child: {new_fragment.id}")
    
    # Calcola metriche del grafo
    depth = graph.get_max_depth()
    node_count = len(graph.fragments)
    
    print(f"\n✓ Metriche del DAG:")
    print(f"  - Nodi totali: {node_count}")
    print(f"  - Profondità massima: {depth}")
    
    return graph


def main():
    """Esegue la demo completa del flusso V2."""
    print("\n" + "🧠"*30)
    print(" MEMORY FRAGMENTS V2 — DEMO COMPLETA")
    print("🧠"*30)
    print("\nQuesto script dimostra il flusso completo:")
    print("  1. Ingest → 2. Retrieval → 3. Appeal → 4. Governance → 5. Genealogy")
    
    try:
        # Fase 1: Ingest
        fragment = demo_ingest()
        
        # Fase 2: Retrieval
        retriever, results = demo_retrieval(fragment)
        
        # Fase 3: Appeal
        trial_space, appeal, metrics = demo_appeal(fragment)
        
        # Fase 4: Governance
        governance, approved, decision = demo_governance(appeal, metrics)
        
        # Fase 5: Genealogy
        graph = demo_genealogy(fragment, approved)
        
        # Summary finale
        print("\n" + "="*60)
        print("✅ DEMO COMPLETATA CON SUCCESSO")
        print("="*60)
        print(f"\nRiepilogo:")
        print(f"  ✓ Fragment ingestito: {fragment.id}")
        print(f"  ✓ Retrieval eseguito: {len(results)} risultati")
        print(f"  ✓ Appeal trial: {appeal.status}")
        print(f"  ✓ Governance decision: {decision}")
        print(f"  ✓ Genealogy graph: {len(graph.fragments)} nodi")
        
        print(f"\n📄 Per ulteriori informazioni:")
        print(f"  - Paper V1: doi:10.5281/zenodo.14534720")
        print(f"  - Paper V2: doi:10.5281/zenodo.17069503")
        print(f"  - GitHub: https://github.com/fra150/memory-fragments")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE LA DEMO: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

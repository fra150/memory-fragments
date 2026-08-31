# Memory Fragments V2

Modello cognitivo modulare per la risoluzione di problemi complessi basato su
"frammenti di memoria" (Bulla, 2024-2025). Il repository contiene il codice
**v0.4.0** dell'architettura V2 e conserva la legacy v0.1.0 in `archive/legacy/`.

## Architettura

| Componente | Descrizione |
|---|---|
| `StaticArchive` | Archivio immutabile di frammenti con checksum e controllo duplicati |
| `HybridRetriever` | Retrieval ibrido BM25 + embedding (con fallback offline deterministico) |
| `AppealTrialSpace` | Sandbox dove il modello propone modifiche ai frammenti |
| `Evaluator` | Metriche automatiche: `delta_token`, `coverage`, `risk`, `aggregate_score` |
| `DiffExplainEngine` | Diff strutturato + spiegazioni leggibili per la revisione |
| `GenealogyGraph` | DAG del versioning (antenati, discendenti, profondità, provenance) |
| `GovernanceAPI` | Workflow human-in-the-loop: submit → approve/reject → rollback |
| `AgentCircuit` | Circuito a 3 agenti deterministici (hard reject / grey zone / fast accept) |
| `FragmentGuardian` | Qualità gate all'ingresso dei `Cassetto` (threshold 0.80, override per source) |
| `Cassetto` | Scaffale di libreria per dominio con guardian all'ingresso |
| `MemoryFragmentsModel` | **Orchestratore unificato**: ingest → query → appeal → governance → export |
| `Rastrello`, `Modellatore`, `Dispatcher` | Pattern discovery, tabs/slots, instradamento |

## Struttura

```
memory_fragments/          ← package (flat-layout, la root IS il package)
├── __init__.py            ← export pubblico, __version__ = "0.4.0"
├── cli.py                 ← CLI (python -m memory_fragments)
├── __main__.py            ← entry point `python -m memory_fragments`
├── model.py               ← orchestratore MemoryFragmentsModel
├── config.py              ← default_config (Retriever/Evaluator/Governance/...)
├── models/                ← dataclass: Fragment, Appeal, AppealMetrics, GenealogyGraph, QualityProvenance
├── archive/               ← StaticArchive, AppealTrialSpace
├── retrieval/             ← BM25Indexer, EmbeddingIndexer, HybridRetriever
├── engine/                ← Evaluator, DiffExplainEngine, Composer, ConflictDetector, Intake, Rastrello, Modellatore
├── governance/            ← GovernanceAPI, GovernanceReport
├── library/               ← Cassetto, FragmentGuardian, AgentCircuit, Quarantine, Improver, LibrarySystem
├── calibration/           ← dataset di calibrazione + agenti deterministici
├── examples/              ← run_demo_v2.py (demo del flusso completo)
└── tests/                 ← 136 test pytest (verdi)
```

## Installazione

```bash
git clone https://github.com/fra150/memory-fragments.git
cd memory-fragments
pip install -e .
# Opzionali:
pip install -e ".[embedding]"   # sentence-transformers per embedding reali
pip install -e ".[llm]"         # openai + anthropic per gli improver LLM
pip install -e ".[dev]"         # pytest, black, flake8, mypy
```

## Quickstart

### Orchestratore (flusso completo)

```python
from memory_fragments import Fragment, FragmentMetadata, MemoryFragmentsModel

model = MemoryFragmentsModel()

# 1. Ingest
frag = Fragment(
    fragment_id="f1",
    content="La fotosintesi usa la luce solare per produrre glucosio.",
    metadata=FragmentMetadata(topic="biologia", quality=0.92, tags=["fotosintesi"]),
)
print(model.ingest(frag))  # IngestResult(accepted=True, ...)

# 2. Query (retrieval ibrido + composizione)
result = model.query("fotosintesi energia luce", top_k=3)
print(result.response)

# 3. Appeal trial con metriche automatiche
appeal = model.propose(
    appeal_id="A1",
    source_ids=["f1"],
    proposed_content="La fotosintesi usa la luce solare e produce ossigeno.",
    explanation="Aggiunta del prodotto ossigeno.",
)
print(appeal.metrics)  # delta_token, coverage, risk, aggregate_score

# 4. Governance human-in-the-loop
model.submit("A1")
approved = model.approve("A1", approver="reviewer", notes="ok")

# 5. Persistenza dello stato
model.save_state("model.json")
restored = MemoryFragmentsModel.load_state("model.json")
```

### CLI

```bash
python -m memory_fragments version
python -m memory_fragments init --state state.json
python -m memory_fragments ingest --state state.json --id f1 \
    --content "testo" --topic biologia --quality 0.9
python -m memory_fragments query --state state.json --text "fotosintesi"
python -m memory_fragments demo --state state.json
```

### Demo completa

```bash
python examples/run_demo_v2.py            # offline (embedding di fallback)
python examples/run_demo_v2.py --online   # sentence-transformers
```

## Test

```bash
pip install -e ".[dev]"
pytest            # 136 test verdi
```

Le metriche e gli agenti sono deterministici (seed `zlib.crc32`, nessun `hash()`
di stringhe); la suite è verificata identica sotto `PYTHONHASHSEED=1` vs `777`.

## Paper di riferimento

- **V1 (2024)**: *A Modular Cognitive Model for Solving Complex Problems Based on "Memory Fragments"* — doi:10.5281/zenodo.14534720
- **V2 (2025)**: *Memory Fragments V2* (Appeal Trial, DAG, metriche automatiche, governance) — doi:10.5281/zenodo.17069503

Citazione consigliata per il codice: vedere [`CITATION.cff`](CITATION.cff).

## Licenza

MIT — vedere `LICENSE` (se presente) e i paper Zenodo per il riferimento accademico.

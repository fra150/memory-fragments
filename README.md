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
| `AgentCircuit` | Circuito a n agenti (mock deterministici o LLM locali via Ollama): hard reject / grey zone / fast accept, voto di maggioranza + provenance |
| `FragmentGuardian` | Qualità gate all'ingresso dei `Cassetto` (threshold 0.80, override per source) |
| `Cassetto` | Scaffale di libreria per dominio con guardian all'ingresso |
| `MemoryFragmentsModel` | **Orchestratore unificato**: ingest → query → appeal → governance → export |
| `Rastrello`, `Modellatore`, `Dispatcher` | Pattern discovery (con fase opzionale n-agenti prima della certificazione), tabs/slots, instradamento |

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
└── tests/                 ← 133 test pytest (verdi)
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
pytest            # 133 test verdi
```

Le metriche e gli agenti mock sono deterministici (seed `zlib.crc32`, nessun `hash()`
di stringhe); la suite è verificata identica sotto `PYTHONHASHSEED=1` vs `777`.

## Agenti LLM locali (Ollama)

Il `AgentCircuit` funziona anche con LLM reali eseguiti in locale via **Ollama**
(nessun costo API). I frammenti vengono valutati da n agenti LLM con voto di
maggioranza e provenance completa (`majority_vote_n`).

```python
from memory_fragments.calibration.agents import AgentConfig, OllamaQualityAgent
from memory_fragments.library.circuit import AgentCircuit

# Scarica un modello piccolo:  ollama pull llama3.2:1b
def make_agent(name, temperature):
    cfg = AgentConfig(name=name, model_id="llama3.2:1b",
                      temperature=temperature, timeout_seconds=180)
    return OllamaQualityAgent(cfg, ollama_model="llama3.2:1b", fallback_to_mock=False)

circuit = AgentCircuit({
    "strict":   make_agent("agent-llm-strict", 0.1),
    "balanced": make_agent("agent-llm-balanced", 0.3),
    "generous": make_agent("agent-llm-generous", 0.5),
})

result = circuit.evaluate(fragment)
if result.accepted:
    vetted = circuit.create_accepted_fragment(fragment, result)
```

Il circuito LLM può essere iniettato anche nel **Rastrello**: ogni pattern estratto
passa dal voto n-agenti prima di diventare un frammento certificato:

```python
rastrello = Rastrello(guardian, archive, intake, circuit=circuit)
certified = rastrello.propose_candidates(rastrello.scan_code(code, context="mod"))
# certified: solo i pattern approvati dal voto, con id "<id>_vetted"
```

> Nota: gli agenti LLM a temperatura > 0 sono stocastici — lo stesso frammento può
> passare o essere filtrato tra esecuzioni diverse. Per gate deterministici usare i
> mock (`create_mock_agents()`) o `temperature=0`; l'LLM è pensato per la grey zone,
> dove la diversità di giudizio è un valore.

## Paper di riferimento

- **V1 (2024)**: *A Modular Cognitive Model for Solving Complex Problems Based on "Memory Fragments"* — doi:10.5281/zenodo.14534720
- **V2 (2025)**: *Memory Fragments V2* (Appeal Trial, DAG, metriche automatiche, governance) — doi:10.5281/zenodo.17069503

Citazione consigliata per il codice: vedere [`CITATION.cff`](CITATION.cff).

## Licenza

MIT — vedere `LICENSE` (se presente) e i paper Zenodo per il riferimento accademico.

# Memory Fragments — Project README

## 📌 Panoramica
**Memory Fragments** è un modello cognitivo modulare (v0.3.0 V2) per la risoluzione di problemi complessi basati su "frammenti di memoria". L'architettura integra:

- **StaticArchive** immutabile
- **AppealTrial** sandbox per modifiche approvate
- **GenealogyGraph** DAG del versioning
- **GovernanceAPI** human-in-the-loop
- **HybridRetriever** (BM25 + embedding)
- **Evaluator** con metriche automatiche (token/coverage/risk)
- **AgentCircuit** a 3 agenti + calibrazione
- **Rastrello** (pattern discovery)
- Library (`Cassetto`, `Quarantine`, `Circuit`, `Guardian`, `Improver`, `System`)

## 🗂️ Struttura del progetto
```
memory_fragments/
├── .gitignore              ← esclude .pyc, __pycache__, *.egg-info, /output/, /raw/
├── __init__.py             ← pacchetto Python principal
├── config.py               ← EvaluatorConfig + GovernanceConfig (v0.3.0)
├── raw/                    ← documenti grezzi + note temporanee
│   └── README.md           ← convenzione file compiled_
├── wiki/                   ← fonte della verità
│   ├── indice.md           ← indice delle sottocartelle
│   ├── progetti/
│   ├── clienti/
│   └── eventi/
├── output/                 ← report generati, riassunti, documenti
│   └── lavoro_da_svolgere.md
├── engine/                 ← modulo motore
│   ├── conflict.py         ← placeholder NLI (heuristic + TODO ONNX)
│   ├── evaluator.py        ← metriche + rilevamento contraddizioni (placeholder)
│   ├── modellatore.py      ← _extract_tabs_and_slots (embedding + heuristic)
│   └── ...
├── governance/             ← governance API human-in-the-loop
│   └── api.py
├── models/                 ← modelli dati (MemoryFragment, Appeal, Metrics, Graph)
│   ├── fragment.py
│   ├── appeal.py
│   ├── graph.py
│   └── quality.py
├── library/                ← cassette/utility
│   ├── cassetto.py
│   ├── circuit.py
│   ├── guardian.py
│   ├── improver.py
│   ├── quarantine.py
│   └── system.py
├── calibration/            ← dataset di calibrazione (15 esempi + TODO espansione)
│   ├── dataset.py
│   ├── agents.py
│   ├── validator.py
│   └── __init__.py
└── retrieval/              └── indice + retriever BM25+embedding
    ├── __init__.py
    ├── indexer.py
    └── retriever.py
```

## 🚀 Quickstart (esempio minimale)
```python
from memory_fragments import StaticArchive, HybridRetriever, Evaluator, GovernanceAPI

# 1. Inizializza l'archivio
archive = StaticArchive()

# 2. Aggiungi knowledge (es. documenti grezzi in /raw/)
archive.add_knowledge("path/to/raw/document.txt")

# 3. Query con retrieval ibrido
results = archive.query("concetto chiave", top_k=5)

# 4. Valuta con metriche automatiche
metrics = Evaluator.evaluate(results)

# 5. Avvia sessione di modifica in sandbox
appeal = appeal.start_edit_session(diff_proposed)
```

## 📄 Paper di riferimento
- **V1 (2024)**: *A Modular Cognitive Model for Solving Complex Problems Based on "Memory Fragments"* — doi:10.5281/zenodo.14534720
- **V2 (2025)**: *Memory Fragments V2* (Appeal Trial, DAG, metriche automatiche, governance) — doi:10.5281/zenodo.17069503

## 🛠️ Stato attuale (2026-08-31)
- `main` sincronizzato su GitHub: legacy v0.1.0 (`memory-fragments/`) + codice v0.3.0 V2 alla radice
- Work plan: `output/lavoro_da_svolgere.md` elenca gap funzionali, pulizie, test, allineamento
- `.gitignore` creato per escludere `/output/`, `/raw/`, `__pycache__/`, `*.egg-info/`
- Autenticazione GitHub attiva via `gh` CLI

## 📦 Installazione (dev)
```bash
git clone https://github.com/fra150/memory-fragments.git
cd memory-fragments
# Opzionale: crea un ambiente virtuale
python -m venv .venv && source .venv/Scripts/activate
pip install -e .
```
> Nota: le dipendenze opzionali (`sentence-transformers`, `openai`, `anthropic`) sono commentate in `config.py`; attivarle quando necessario.

## 📜 Licenza
Research code — vedere i paper Zenodo per citazioni e riferimento.
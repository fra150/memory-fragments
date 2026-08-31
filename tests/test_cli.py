"""End-to-end tests for the CLI (ingest -> query -> appeal -> approve)."""

import json

from memory_fragments.cli import main
from memory_fragments.retrieval.indexer import EmbeddingIndexer


def _offline_embedding(monkeypatch):
    """Force deterministic fallback embeddings (no network / model download)."""

    def _no_model(self):
        self._fallback = True

    monkeypatch.setattr(EmbeddingIndexer, "_ensure_model", _no_model)

    from memory_fragments.engine.conflict import ConflictDetector

    def _no_conflict_model(self):
        self._fallback = True

    monkeypatch.setattr(ConflictDetector, "_ensure_model", _no_conflict_model)


def test_version(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "memory-fragments" in out


def test_full_workflow(tmp_path, capsys, monkeypatch):
    _offline_embedding(monkeypatch)
    state = tmp_path / "state.json"

    assert main(["--state", str(state), "init"]) == 0

    assert (
        main([
            "--state", str(state), "ingest",
            "--id", "f1",
            "--content", "La fotosintesi usa la luce solare per produrre glucosio.",
            "--topic", "biology",
            "--quality", "0.9",
        ])
        == 0
    )
    assert (
        main([
            "--state", str(state), "ingest",
            "--id", "f2",
            "--content", "Il Sole converte idrogeno in elio con la fusione nucleare.",
            "--topic", "physics",
            "--quality", "0.85",
        ])
        == 0
    )

    # -- query --------------------------------------------------------------
    assert main(["--state", str(state), "query", "--text", "luce solare energia"]) == 0
    out = capsys.readouterr().out
    assert "f1" in out

    # -- appeal + submit ----------------------------------------------------
    assert (
        main([
            "--state", str(state), "appeal",
            "--appeal-id", "A1",
            "--sources", "f1,f2",
            "--content", "La fotosintesi usa la luce solare e il Sole genera energia.",
            "--explanation", "Unisce i due frammenti.",
            "--submit",
        ])
        == 0
    )
    out = capsys.readouterr().out
    assert "submitted for review" in out

    # -- approve ------------------------------------------------------------
    assert main(["--state", str(state), "approve", "--appeal-id", "A1"]) == 0
    out = capsys.readouterr().out
    assert "F-0001" in out

    # -- status -------------------------------------------------------------
    assert main(["--state", str(state), "status"]) == 0
    out = capsys.readouterr().out
    assert "Fragments (archive):    3" in out
    assert "approved:             1" in out

    # -- persisted state is valid JSON --------------------------------------
    data = json.loads(state.read_text(encoding="utf-8"))
    assert len(data["archive"]["fragments"]) == 3
    assert len(data["appeal_space"]["appeals"]) == 1


def test_appeal_rejects_unknown_source(tmp_path, capsys, monkeypatch):
    _offline_embedding(monkeypatch)
    state = tmp_path / "state.json"
    main(["--state", str(state), "init"])
    rc = main([
        "--state", str(state), "appeal",
        "--appeal-id", "A1",
        "--sources", "ghost",
        "--content", "contenuto",
    ])
    assert rc == 1
    out = capsys.readouterr().err
    assert "ghost" in out


def test_reject_workflow(tmp_path, capsys, monkeypatch):
    _offline_embedding(monkeypatch)
    state = tmp_path / "state.json"
    main(["--state", str(state), "init"])
    main([
        "--state", str(state), "ingest",
        "--id", "f1",
        "--content", "Contenuto sufficientemente lungo per superare la soglia di qualita del guardiano.",
        "--quality", "0.9",
    ])
    main([
        "--state", str(state), "appeal",
        "--appeal-id", "A1",
        "--sources", "f1",
        "--content", "Contenuto proposto dal miglioramento del frammento sorgente.",
        "--explanation", "proposta",
        "--submit",
    ])
    assert main(["--state", str(state), "reject", "--appeal-id", "A1", "--reason", "non pertinente"]) == 0
    out = capsys.readouterr().out
    assert "rejected" in out

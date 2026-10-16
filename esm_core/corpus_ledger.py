"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Small SQLite ledger for resumable, multi-dataset field extraction.

Candidate cache files alone do not answer which SMILES have been completed,
which run configuration produced them, or whether a later dataset request is
new work.  This ledger is append/update-on-completion only and keeps that
index on disk, without loading millions of molecule IDs into RAM.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Optional


class CorpusLedger:
    def __init__(self, path: str | Path, analysis_signature: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.analysis_signature = analysis_signature
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysed_molecules (
                analysis_signature TEXT NOT NULL,
                molecule_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_record_index INTEGER NOT NULL,
                smiles TEXT NOT NULL,
                canonical_smiles TEXT,
                cache_file TEXT NOT NULL,
                cube_file TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                completed_unix_s REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'completed',
                online_update_index INTEGER,
                PRIMARY KEY (analysis_signature, molecule_id)
            )
            """
        )
        for column, definition in (
            ("status", "TEXT NOT NULL DEFAULT 'completed'"),
            ("online_update_index", "INTEGER"),
        ):
            try:
                self.connection.execute(
                    f"ALTER TABLE analysed_molecules ADD COLUMN {column} {definition}"
                )
            except sqlite3.OperationalError:
                pass
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS analysed_molecules_signature_smiles "
            "ON analysed_molecules (analysis_signature, canonical_smiles)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def is_completed(self, molecule_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM analysed_molecules WHERE analysis_signature = ? AND molecule_id = ? "
            "AND status = 'completed'",
            (self.analysis_signature, molecule_id),
        ).fetchone()
        return row is not None

    def get_cube_file(self, molecule_id: str) -> Optional[str]:
        """Read-only lookup, added for retrospective analyses that join an
        archived occurrence (which deliberately does not carry a cube path --
        see ``_ArchivedCandidate`` in ``online_archive.py``) back to its
        source cube -- e.g. a nearest-atom sanity check on nuclear-attractor
        classes. Returns None if the molecule has no completed row for this
        ledger's ``analysis_signature``, or its cube was never recorded."""
        row = self.connection.execute(
            "SELECT cube_file FROM analysed_molecules WHERE analysis_signature = ? "
            "AND molecule_id = ? AND status = 'completed'",
            (self.analysis_signature, molecule_id),
        ).fetchone()
        return row[0] if row else None

    def mark_completed(
        self,
        *,
        molecule_id: str,
        dataset_name: str,
        dataset_record_index: int,
        smiles: str,
        canonical_smiles: Optional[str],
        cache_file: str,
        cube_file: str,
        metadata: Mapping[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO analysed_molecules (
                analysis_signature, molecule_id, dataset_name, dataset_record_index,
                smiles, canonical_smiles, cache_file, cube_file, metadata_json,
                completed_unix_s, status, online_update_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', NULL)
            """,
            (
                self.analysis_signature,
                molecule_id,
                dataset_name,
                int(dataset_record_index),
                smiles,
                canonical_smiles,
                cache_file,
                cube_file,
                json.dumps(metadata, ensure_ascii=False, default=str),
                time.time(),
            ),
        )
        self.connection.commit()

    def stage_update(self, update_index: int, records: list[Mapping[str, Any]]) -> None:
        """Durably reserve an append before its state checkpoint is written."""
        for entry in records:
            record = entry["record"]
            smiles = str(record.get("smiles", "") or "")
            self.connection.execute(
                """
                INSERT OR REPLACE INTO analysed_molecules (
                    analysis_signature, molecule_id, dataset_name, dataset_record_index,
                    smiles, canonical_smiles, cache_file, cube_file, metadata_json,
                    completed_unix_s, status, online_update_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'staged', ?)
                """,
                (
                    self.analysis_signature,
                    entry["molecule_id"],
                    entry["dataset_name"],
                    int(entry["dataset_record_index"]),
                    smiles,
                    entry.get("canonical_smiles"),
                    entry["cache_file"],
                    entry["cube_path"],
                    json.dumps(entry["metadata"], ensure_ascii=False, default=str),
                    time.time(),
                    int(update_index),
                ),
            )
        self.connection.commit()

    def staged_update_indices(self) -> list[int]:
        rows = self.connection.execute(
            "SELECT DISTINCT online_update_index FROM analysed_molecules "
            "WHERE analysis_signature = ? AND status = 'staged' "
            "ORDER BY online_update_index",
            (self.analysis_signature,),
        ).fetchall()
        return [int(row[0]) for row in rows if row[0] is not None]

    def complete_staged_update(self, update_index: int) -> None:
        self.connection.execute(
            "UPDATE analysed_molecules SET status = 'completed', completed_unix_s = ? "
            "WHERE analysis_signature = ? AND status = 'staged' AND online_update_index = ?",
            (time.time(), self.analysis_signature, int(update_index)),
        )
        self.connection.commit()

    def staged_molecule_ids(self, update_index: int) -> set[str]:
        rows = self.connection.execute(
            "SELECT molecule_id FROM analysed_molecules WHERE analysis_signature = ? "
            "AND status = 'staged' AND online_update_index = ?",
            (self.analysis_signature, int(update_index)),
        ).fetchall()
        return {str(row[0]) for row in rows}

    def count_completed(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM analysed_molecules WHERE analysis_signature = ? AND status = 'completed'",
            (self.analysis_signature,),
        ).fetchone()
        return int(row[0]) if row else 0

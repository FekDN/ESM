"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
One-time migration: drop online-consolidation settings from the stored
analysis signature, matching the esm_master_analytics.py fix that decouples
merge/consolidation policy from the extraction/ledger signature.

Run this ONCE, right after updating esm_master_analytics.py and
esm_core/online_archive.py, BEFORE changing any sidebar setting. It recomputes
the OLD-style signature using your CURRENT config (online_merge_enabled /
online_merge_min_similarity / online_merge_screen_max_prototypes as they are
right now), which is correct as long as you have not already hit the
signature-mismatch crash and hand-deleted a checkpoint to work around it.

Usage (from the project root, inside the esm_env conda environment):
    python migrate_signature.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import esm_master_analytics as m  # noqa: E402  (import after docstring is fine)


def _old_payload() -> dict:
    payload = m._analysis_signature_payload()  # already the NEW (post-fix) shape
    payload["online_consolidation"] = {
        "enabled": m.CONFIG.online_merge_enabled,
        "minimum_centroid_cosine_similarity": m.CONFIG.online_merge_min_similarity,
        "screen_max_prototypes": m.CONFIG.online_merge_screen_max_prototypes,
        "criterion": "transitive_centroid_cosine_graph",
    }
    return payload


def _signature_id(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    new_payload = m._analysis_signature_payload()
    old_payload = _old_payload()
    old_hash = _signature_id(old_payload)
    new_hash = _signature_id(new_payload)

    if old_hash == new_hash:
        print("Nothing to do: old and new signatures already match.")
        return

    ledger_path = Path(m.CORPUS_LEDGER_PATH)
    if ledger_path.exists():
        con = sqlite3.connect(ledger_path)
        try:
            cursor = con.execute(
                "UPDATE analysed_molecules SET analysis_signature = ? WHERE analysis_signature = ?",
                (new_hash, old_hash),
            )
            con.commit()
            print(f"Ledger ({ledger_path}): migrated {cursor.rowcount} row(s).")
        finally:
            con.close()
    else:
        print(f"Ledger not found at {ledger_path}; nothing to migrate there.")

    state_path = Path(m.ONLINE_STATE_PATH)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        signature = state.get("analysis_signature", {})
        if "online_consolidation" in signature:
            signature.pop("online_consolidation")
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"State ({state_path}): dropped online_consolidation from analysis_signature.")
        else:
            print(f"State ({state_path}): already has no online_consolidation field; left as-is.")
    else:
        print(f"State file not found at {state_path}; nothing to migrate there.")

    print("\nDone. The next online-append run should recognise all previously completed molecules again.")


if __name__ == "__main__":
    main()

"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
run_boundary_audit.py
======================
CLI for `esm_core.boundary_audit`: finds occurrences whose class
assignment has drifted from its own class's (running-mean) centroid, and,
with `--commit`, writes accepted reassignments to this archive's
`boundary_reassignments.jsonl` ledger. Dry-run (report only, nothing
written) unless `--commit` is passed -- same asymmetry as `recluster`'s
`--promote`: cheap to preview, explicit to make live.

Reads only `online_alphabet_state.json` and the descriptor archive under
`--online-dir`; no PySCF, no cube, no CorpusLedger -- the same "no DFT"
guarantee `recluster_from_archive` documents, for the same reason (this is
a replay/audit over what extraction already produced, not new discovery).

Usage:
    python run_boundary_audit.py --online-dir esm_analysis/runs/default/online/archive \\
        --state-path esm_analysis/runs/default/online/online_alphabet_state.json \\
        --audit-label 2026-09-20_weekly [--commit] [--drift-margin 0.0] [--min-improvement 0.0]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from esm_core.boundary_audit import find_boundary_candidates, write_boundary_reassignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--online-dir", required=True, help="Archive directory (occurrences.jsonl, descriptor_batches/) -- an ArtifactPaths.online_archive_dir")
    parser.add_argument("--state-path", required=True, help="online_alphabet_state.json for the SAME run (ArtifactPaths.online_state_path)")
    parser.add_argument("--audit-label", required=True, help="Recorded on every written reassignment; distinguishes this audit run in the ledger's history")
    parser.add_argument("--drift-margin", type=float, default=0.0)
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Hysteresis: a proposed class must be closer by at least this much")
    parser.add_argument("--commit", action="store_true", help="Write accepted candidates to boundary_reassignments.jsonl. Without this flag, report only.")
    args = parser.parse_args()

    state = json.loads(Path(args.state_path).read_text(encoding="utf-8"))
    candidates = find_boundary_candidates(
        state, args.online_dir, drift_margin=args.drift_margin, min_improvement=args.min_improvement,
    )

    print(f"{len(candidates)} boundary candidate(s) found.")
    if candidates:
        by_type = Counter(c.critical_point_type for c in candidates)
        for field_type, count in by_type.most_common():
            print(f"  {field_type}: {count}")
        print("\nSample (up to 10):")
        for c in candidates[:10]:
            print(f"  {c.molecule_id} [{c.critical_point_type}]: {c.current_class_id} (d={c.current_distance:.4f}, "
                  f"own threshold={c.current_threshold:.4f}) -> {c.proposed_class_id} (d={c.proposed_distance:.4f})")

    if args.commit:
        if not candidates:
            print("Nothing to commit.")
            return
        path = write_boundary_reassignments(args.online_dir, candidates, audit_label=args.audit_label)
        print(f"\nCommitted {len(candidates)} reassignment(s) to {path}")
    else:
        print("\nDry run -- nothing written. Re-run with --commit to record these in boundary_reassignments.jsonl.")


if __name__ == "__main__":
    main()

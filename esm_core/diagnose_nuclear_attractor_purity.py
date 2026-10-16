"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
diagnose_nuclear_attractor_purity.py
=====================================
A (3,-3) density maximum coincides with a nucleus by the cusp condition --
not a finding of this pipeline, a theorem. Nuclear charge is an integer, and
the periodic table is finite, so nuclear-attractor classes have a near
ground-truth floor no other critical-point type has: whatever a molecule's
element composition is, its maxima MUST separate by element at least that
coarsely. This script checks that floor directly against an already-archived
online run, joining back to each source cube's atom block only as a read-only
reference (exactly the role `_nearest_atom_reference` in `visualization.py`
already documents: "a continuous post-hoc spatial reference, never a label" --
this script uses it only to *audit* the label the pipeline already assigned,
never to feed it back into extraction or classification).

What a clean pass looks like: a small number of nuclear_attractor classes,
each dominated by one atomic number, with the count of classes per element
either close to 1 (raw element, no environment-sensitivity captured) or a
handful (real, plausible hybridization/environment sub-structure -- e.g. a
handful of chemically distinct carbon environments). A class mixing two
different atomic numbers is an outright bug, not an interpretation question
-- it means two different elements' maxima were placed in the same class,
which should never happen given the floor argument above.

What a bad pass looks like, and why it still matters even though every
individual class might look internally consistent: far more classes than a
generous reading of "element x plausible environment count" can explain,
concentrated in the one type that should be the *most* solidly discrete of
the four (nuclear_attractor, bond_critical_point, ring_critical_point,
cage_critical_point) rather than the least. That pattern points at threshold
calibration or numerical noise in the search/descriptor, not at real
chemistry -- exactly the kind of thing this script is for catching before
any claim about bond- or ring-type classes (which have no analogous
ground-truth floor to check against) gets taken seriously.

Usage:
    python diagnose_nuclear_attractor_purity.py \\
        --online-dir /path/to/online_archive \\
        --ledger /path/to/corpus_ledger.sqlite \\
        --analysis-signature '<same string the run used>'

Only reads: occurrences.jsonl, descriptor_batches/*.npy (via
`iter_archived_occurrences`, so no DFT/PySCF/cube-of-the-full-grid access --
only each cube's cheap text header, via `read_cube_header`), and the ledger's
`cube_file` column (via the new `CorpusLedger.get_cube_file`). Writes
nothing; this is a read-only audit.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

from esm_core.corpus_ledger import CorpusLedger
from esm_core.online_archive import iter_archived_occurrences
from esm_core.online_alphabet import resolve_online_class_id
from esm_core.visualization import read_cube_header, _nearest_atom_reference


_ELEMENT_SYMBOL = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
    17: "Cl", 35: "Br", 53: "I",
}


def _symbol(atomic_number: int) -> str:
    return _ELEMENT_SYMBOL.get(atomic_number, f"Z{atomic_number}")


def diagnose(
    online_dir: str,
    ledger_path: str,
    analysis_signature: str,
    *,
    field_type: str = "nuclear_attractor",
    nearest_atom_distance_warn: float = 0.6,
) -> None:
    ledger = CorpusLedger(ledger_path, analysis_signature)
    state_path = Path(online_dir) / "online_alphabet_state.json"
    # Read-only audit: load the state's raw JSON directly rather than
    # `load_online_alphabet_state`, which insists the caller's
    # analysis_signature payload matches the state's exactly -- the right
    # check for something about to WRITE more candidates into that state,
    # not for a script that only wants its `class_aliases` to resolve
    # historical assignments to their current, possibly-merged class.
    alphabet_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None

    cube_header_cache: Dict[str, Optional[dict]] = {}
    class_element_counts: Dict[str, Counter] = defaultdict(Counter)
    class_far_reference_count: Dict[str, int] = Counter()
    n_seen = 0
    n_no_cube = 0

    for record, _vector in iter_archived_occurrences(online_dir):
        if record["critical_point_type"] != field_type:
            continue
        n_seen += 1
        molecule_id = record["molecule_id"]
        class_id = record["class_id"]
        if alphabet_state is not None:
            class_id = resolve_online_class_id(alphabet_state, record["class_id_at_append"])

        if molecule_id not in cube_header_cache:
            cube_file = ledger.get_cube_file(molecule_id)
            cube_header_cache[molecule_id] = read_cube_header(cube_file) if cube_file else None
        header = cube_header_cache[molecule_id]
        if header is None:
            n_no_cube += 1
            continue

        reference = _nearest_atom_reference(record["critical_point"]["position"], header["atoms"])
        if reference is None:
            continue
        class_element_counts[class_id][reference["atomic_number"]] += 1
        if reference["distance"] > nearest_atom_distance_warn:
            class_far_reference_count[class_id] += 1

    ledger.close()

    print(f"{field_type}: {n_seen} occurrences, {len(class_element_counts)} classes "
          f"({n_no_cube} skipped -- no ledger/cube row for their molecule)")
    if not class_element_counts:
        print("Nothing to report -- check --online-dir/--ledger/--analysis-signature.")
        return

    classes_per_element: Dict[str, list] = defaultdict(list)
    mixed_classes = []
    for class_id, counts in class_element_counts.items():
        dominant_z, dominant_n = counts.most_common(1)[0]
        total = sum(counts.values())
        classes_per_element[_symbol(dominant_z)].append(class_id)
        if len(counts) > 1:
            mixed_classes.append((class_id, dict(counts), total))

    print("\nClasses per element (dominant nearest-atom species):")
    for symbol, class_ids in sorted(classes_per_element.items(), key=lambda kv: -len(kv[1])):
        print(f"  {symbol:>3}: {len(class_ids):5d} classes")

    print(f"\nMixed-element classes (more than one distinct nearest-atom species; "
          f"should be zero if the floor argument holds and references aren't noisy): "
          f"{len(mixed_classes)}")
    for class_id, counts, total in sorted(mixed_classes, key=lambda item: -item[2])[:20]:
        breakdown = ", ".join(f"{_symbol(z)}={n}" for z, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        far = class_far_reference_count.get(class_id, 0)
        flag = (f"  ({far}/{total} nearest-atom distances exceeded {nearest_atom_distance_warn}, "
                f"reference itself may be unreliable there)") if far else ""
        print(f"    {class_id}: {breakdown}{flag}")

    total_classes = len(class_element_counts)
    total_elements = len(classes_per_element)
    print(f"\n{total_classes} nuclear_attractor classes over {total_elements} elements seen "
          f"({', '.join(sorted(classes_per_element))}).")
    print("A handful of classes per element (raw identity, or a few real hybridisation/")
    print("environment groups) is the expected floor. Anything far beyond that -- concentrated")
    print("in the type that should be the MOST discrete of the four, not the least -- points at")
    print("threshold calibration or descriptor noise, not at chemistry.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-dir", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--analysis-signature", required=True)
    parser.add_argument("--field-type", default="nuclear_attractor")
    args = parser.parse_args()
    diagnose(args.online_dir, args.ledger, args.analysis_signature, field_type=args.field_type)


if __name__ == "__main__":
    main()

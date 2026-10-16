"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
interpret_alphabet.py
======================
Reads `esm_master_analytics.py`'s two output files and renders what the
pipeline actually *found* -- the discovered "letter" structure -- rather
than re-narrating pipeline health metrics (CV selection, warnings, timing).

This is deliberately NOT an atom-recovery report. A `LetterClass` is an
equivalence class over topological critical points of a scalar field
(§15 steps 1-3); whether a given class corresponds to what chemistry
already calls an atom, a bond, or something with no existing name at all
is exactly the open question §3.1/§3.2's hypothesis test is still deciding
-- this tool never assumes an answer by translating a class into an
element/functional-group label the pipeline itself never produced. Where a
class DOES correlate with a named physical property, that correlation
(`semantic_status.named_correlates`) is reported as-is: property name and
correlation strength, nothing invented on top.

Two input files, two different jobs:
    - `esm_analytics_report.json` (large; has `alphabet_slim.classes`, the
      per-*letter* records, plus `phase_analyses` with per-letter
      commutation/distributional/paradigm detail) -- required for anything
      below the corpus-aggregate level.
    - `esm_analytics_summary.json` (small; `report.json`'s top-level keys
      minus `alphabet_slim`/`phase_analyses`/`falsification_profiles_S19`
      -- see `esm_master_analytics.py`'s `main()`, where
      `full_report = {**summary, "alphabet_slim": ..., ...}`) -- corpus-
      level-only context (CV selection, control-vs-structure, theory-gate
      tallies) when the multi-megabyte report isn't at hand or isn't
      needed.

Usage:
    python interpret_alphabet.py --report esm_analytics_report.json
    python interpret_alphabet.py --summary esm_analytics_summary.json   # corpus-level only, no letters
    python interpret_alphabet.py --report R.json --json-out alphabet_digest.json
    streamlit run view_esm_alphabet.py -- --manifest esm_visualization_data/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ==========================================================================
# loading
# ==========================================================================


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_visualization_evidence(report: dict, manifest: dict) -> None:
    """Fill report-era omissions from a scene bundle without changing IDs.

    Reports written before visualisation export omitted occurrence/reuse fields
    to stay compact.  A scene bundle supplies those values from the same
    candidate assignment and points to concrete 3-D realisations; it never
    forms a second alphabet or changes an existing semantic conclusion.
    """
    alphabet = report.get("alphabet_slim") or {}
    records = alphabet.get("classes", []) if isinstance(alphabet, dict) else []
    visual_records = {
        item.get("class_id"): item
        for item in manifest.get("letter_classes", [])
        if item.get("class_id")
    }
    for record in records:
        visual = visual_records.get(record.get("class_id"))
        if visual is None:
            continue
        for key in ("n_occurrences", "n_molecules", "semantic_status"):
            if record.get(key) is None and visual.get(key) is not None:
                record[key] = visual[key]
        if not record.get("example_molecule_ids"):
            record["example_molecule_ids"] = [
                example["molecule_id"]
                for example in visual.get("example_realisations", [])
                if example.get("molecule_id")
            ][:8]


# ==========================================================================
# cross-referencing: everything below reads only `report.json`'s own keys,
# by the field names `esm_master_analytics.py` actually writes (`slim_class`,
# `slim_fund_results`, and the `phase_analyses` block in `main()`) -- not a
# schema invented for this script.
# ==========================================================================


def _as_float(x: Any) -> Optional[float]:
    """Coerce a JSON-decoded numeric field to float, or None.

    `esm_master_analytics.sanitize_for_json` turns NaN/Inf into their
    `str()` form (`"nan"`, `"inf"`, `"-inf"`) rather than leaving them as
    JSON numbers (plain `json.dump` cannot represent them at all) -- so a
    score field that was a real NaN upstream (e.g. a `ConfidenceBlock`
    bootstrapped from `n=1`) arrives here as a *string*, not a float, and
    would otherwise blow up `f"{x:.2f}"` formatting or `>=` comparisons
    several calls later instead of at the point that actually needs
    guarding.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)  # handles "nan"/"inf"/"-inf" too
        except ValueError:
            return None
    return None


def index_axis_annotations(report: dict) -> Dict[str, List[dict]]:
    """class_id -> list of per-axis commutation-test verdicts (§11).

    `commutation_S11.annotations` is a DICT keyed by class_id, not a list
    -- see `esm_master_analytics.py`: `axis_annotations = {}`, then
    `axis_annotations[cls.class_id] = build_axis_annotations(...)`, packed
    as `sanitize_for_json(axis_annotations)`, which preserves dict
    structure (`{str(k): sanitize_for_json(v) ...}`). Each value also
    happens to carry its own redundant `class_id` field (from the
    `AxisAnnotations` dataclass itself), which is what made the bug here
    easy to miss reading the dataclass alone rather than the call site.
    """
    block = report.get("phase_analyses", {}).get("commutation_S11", {})
    raw = block.get("annotations", {})
    out: Dict[str, List[dict]] = {}
    if isinstance(raw, dict):
        for cid, ann in raw.items():
            out[cid] = (ann or {}).get("axes", [])
    elif isinstance(raw, list):
        # defensive fallback if a future version packs this as a list instead
        for ann in raw:
            cid = ann.get("class_id")
            if cid is not None:
                out[cid] = ann.get("axes", [])
    return out


def index_distributional_decisions(report: dict) -> Dict[str, dict]:
    """class_id -> {decision, n_linked, n_total, type} (§3.1/§3.2)."""
    block = report.get("phase_analyses", {}).get("distributional_test_S3_1_summary", {})
    return {d["class_id"]: d for d in block.get("per_class_decisions", []) if "class_id" in d}


def index_paradigm_membership(report: dict) -> Dict[str, List[str]]:
    """class_id -> [paradigm_id, ...] it participates in (§4.4-4.6)."""
    block = report.get("phase_analyses", {}).get("paradigms_S4_4", {})
    out: Dict[str, List[str]] = defaultdict(list)
    for p in block.get("paradigms", []):
        for cid in p.get("class_ids", []):
            out[cid].append(p.get("paradigm_id"))
    return dict(out)


def index_synonym_pairs(report: dict) -> Dict[str, List[dict]]:
    """class_id -> [SynonymRelation dict, ...] it appears in either side of."""
    block = report.get("phase_analyses", {}).get("paradigms_S4_4", {})
    out: Dict[str, List[dict]] = defaultdict(list)
    for rel in block.get("synonyms", []):
        pair = rel.get("class_id_pair") or []
        for cid in pair:
            out[cid].append(rel)
    return dict(out)


# ==========================================================================
# per-letter record assembly
# ==========================================================================


class Letter:
    """One discovered class, with every other phase's verdict about it
    folded in. This -- not the raw JSON dict -- is "a letter" for the rest
    of this script."""

    __slots__ = (
        "class_id", "critical_point_type", "cluster_stability_score",
        "cluster_stability_n", "n_occurrences", "n_molecules",
        "example_molecule_ids", "named_correlates", "is_orphan",
        "orphan_confidence_score", "axes", "distributional", "paradigm_ids",
        "synonym_relations", "raw",
    )

    def __init__(self, raw: dict):
        self.raw = raw
        self.class_id = raw.get("class_id")
        self.critical_point_type = raw.get("critical_point_type")
        cs = raw.get("cluster_stability") or {}
        self.cluster_stability_score = _as_float(cs.get("score"))
        self.cluster_stability_n = cs.get("n")
        self.n_occurrences = raw.get("n_occurrences")
        self.n_molecules = raw.get("n_molecules")
        self.example_molecule_ids = raw.get("example_molecule_ids", [])
        sem = raw.get("semantic_status") or {}
        self.named_correlates = sem.get("named_correlates", [])
        self.is_orphan = sem.get("orphan", True)
        oc = sem.get("orphan_confidence") or {}
        self.orphan_confidence_score = _as_float(oc.get("score"))
        self.axes: List[dict] = []
        self.distributional: Optional[dict] = None
        self.paradigm_ids: List[str] = []
        self.synonym_relations: List[dict] = []

    # -- derived properties, not stored fields -----------------------------

    @property
    def is_named(self) -> bool:
        return bool(self.named_correlates)

    @property
    def is_multi_molecule(self) -> bool:
        return (self.n_molecules or 0) > 1

    @property
    def axis_tally(self) -> Counter:
        return Counter(a.get("commutation_test_result", "untested") for a in self.axes)

    @property
    def n_significant_axes(self) -> int:
        return self.axis_tally.get("significant", 0)

    def one_line(self) -> str:
        tag = "named" if self.is_named else ("orphan" if self.is_orphan else "untested")
        stab = f"{self.cluster_stability_score:.2f}" if self.cluster_stability_score is not None else "?"
        reuse = f"{self.n_molecules}mol/{self.n_occurrences}occ" if self.n_molecules is not None else "?"
        return f"[{self.class_id}] {self.critical_point_type:<20} stability={stab} reuse={reuse} ({tag})"


def build_letters(report: dict) -> List[Letter]:
    alphabet = report.get("alphabet_slim")
    if alphabet is None:
        raise ValueError(
            "report has no 'alphabet_slim' block -- this looks like a "
            "summary.json, not a report.json. Per-letter interpretation "
            "needs --report, not --summary."
        )
    letters = [Letter(c) for c in alphabet.get("classes", [])]

    axes_by_class = index_axis_annotations(report)
    dist_by_class = index_distributional_decisions(report)
    paradigms_by_class = index_paradigm_membership(report)
    synonyms_by_class = index_synonym_pairs(report)

    for letter in letters:
        letter.axes = axes_by_class.get(letter.class_id, [])
        letter.distributional = dist_by_class.get(letter.class_id)
        letter.paradigm_ids = paradigms_by_class.get(letter.class_id, [])
        letter.synonym_relations = synonyms_by_class.get(letter.class_id, [])

    return letters


# ==========================================================================
# views: each answers one concrete "what did we find" question, not a
# generic dump of all 1000+ classes
# ==========================================================================


def named_letters(letters: Sequence[Letter]) -> List[Letter]:
    """The actually-decoded portion of the alphabet: classes that correlate
    with an already-named physical property (§16.1/§16.3)."""
    out = [l for l in letters if l.is_named]
    out.sort(
        key=lambda l: max((_as_float(c.get("correlation_strength")) or 0.0 for c in l.named_correlates), default=0.0),
        reverse=True,
    )
    return out


def confident_orphans(letters: Sequence[Letter], min_confidence: float = 0.7, min_molecules: int = 2) -> List[Letter]:
    """Structurally solid, cross-molecule-reused classes with no named
    correlate AND a statistically powerful (not just "not found yet")
    absence of correlation (§16.3's equivalence-tested orphan_confidence)
    -- the candidates worth someone's actual attention, not the whole
    orphan pile (most orphans are simply untested, orphan_confidence=None).

    The `min_molecules` reuse gate only applies when `n_molecules` is
    actually known for a class. `(n_molecules or 0) >= min_molecules`
    would treat "don't know" the same as "confirmed single-molecule" --
    harmless when `n_molecules` is missing for a few classes, but if
    `slim_class()` was called without `candidates`/`membership_index` (see
    its docstring) and it's missing for *every* class, that silently
    empties this entire section regardless of how strong the orphan
    evidence is. "Unknown" should not silently fail a reuse check it
    can't evaluate -- see `render_markdown`'s note on this section for how
    to tell the two cases apart in the rendered output.
    """
    out = [
        l for l in letters
        if l.is_orphan
        and l.orphan_confidence_score is not None
        and l.orphan_confidence_score >= min_confidence
        and (l.n_molecules is None or l.n_molecules >= min_molecules)
    ]
    out.sort(key=lambda l: (l.orphan_confidence_score, l.n_molecules if l.n_molecules is not None else -1), reverse=True)
    return out


def most_reused_letters(letters: Sequence[Letter], top_n: int = 20) -> List[Letter]:
    """Classes that recur across the most distinct molecules -- the part of
    the alphabet least likely to be one molecule's clustering idiosyncrasy."""
    out = [l for l in letters if l.n_molecules is not None]
    out.sort(key=lambda l: (l.n_molecules, l.n_occurrences or 0), reverse=True)
    return out[:top_n]


def paradigm_families(letters: Sequence[Letter], report: dict) -> List[dict]:
    """§4.4-4.6: groups of letters linked by one reusable transformation --
    the closest thing to actual discovered grammar (not just a word list)."""
    by_id = {l.class_id: l for l in letters}
    block = report.get("phase_analyses", {}).get("paradigms_S4_4", {})
    families = []
    for p in block.get("paradigms", []):
        members = [by_id[cid] for cid in p.get("class_ids", []) if cid in by_id]
        families.append({
            "paradigm_id": p.get("paradigm_id"),
            "varying_axis": p.get("varying_axis"),
            "rule_reusability_score": _as_float((p.get("rule_reusability") or {}).get("score")),
            "members": members,
        })
    families.sort(key=lambda f: f["rule_reusability_score"] or 0.0, reverse=True)
    return families


def distributional_determinate(letters: Sequence[Letter]) -> Dict[str, List[Letter]]:
    """§3.1/§3.2: the FEW letters the distributional test actually reached a
    verdict on (hypothesis A/B), split out from the "indeterminate" majority
    -- these are the informative cases, not a count to bury in a tally."""
    out: Dict[str, List[Letter]] = defaultdict(list)
    for l in letters:
        if l.distributional and l.distributional.get("decision") != "indeterminate":
            out[l.distributional["decision"]].append(l)
    return dict(out)


def type_breakdown(letters: Sequence[Letter]) -> Dict[str, dict]:
    by_type: Dict[str, List[Letter]] = defaultdict(list)
    for l in letters:
        by_type[l.critical_point_type].append(l)
    out = {}
    for t, group in by_type.items():
        stabilities = [
            l.cluster_stability_score for l in group
            if l.cluster_stability_score is not None and l.cluster_stability_score == l.cluster_stability_score
        ]  # `x == x` is False for NaN -- filters it without importing math for one check
        has_reuse_data = any(l.n_molecules is not None for l in group)
        out[t] = {
            "n_classes": len(group),
            "n_multi_molecule": (sum(1 for l in group if l.is_multi_molecule) if has_reuse_data else None),
            "n_named": sum(1 for l in group if l.is_named),
            "n_orphan_confident": sum(
                1 for l in group if l.is_orphan and (l.orphan_confidence_score or 0) >= 0.7
            ),
            "mean_stability": sum(stabilities) / len(stabilities) if stabilities else None,
        }
    return out


# ==========================================================================
# rendering
# ==========================================================================


def _fmt_letter_block(l: Letter, indent: str = "  ") -> List[str]:
    lines = [f"{indent}{l.one_line()}"]
    if l.named_correlates:
        for c in l.named_correlates:
            r = _as_float(c.get("correlation_strength"))
            r_str = f"{r:.3f}" if r is not None else "?"
            lines.append(f"{indent}    correlate: {c.get('property')} (r={r_str})")
    if l.distributional and l.distributional.get("decision") != "indeterminate":
        d = l.distributional
        lines.append(f"{indent}    distributional: {d['decision']} ({d['n_linked']}/{d['n_total']} linked)")
    if l.n_significant_axes:
        lines.append(f"{indent}    axes: {dict(l.axis_tally)}")
    if l.paradigm_ids:
        lines.append(f"{indent}    paradigm: {', '.join(l.paradigm_ids)}")
    if l.example_molecule_ids:
        lines.append(f"{indent}    e.g. {', '.join(l.example_molecule_ids)}")
    return lines


def render_markdown(
    report: dict,
    letters: List[Letter],
    top_n: int,
    visualization_manifest: Optional[Path] = None,
    letter_archive_manifest: Optional[Path] = None,
) -> str:
    lines: List[str] = []
    meta = report.get("metadata", {})
    lines.append(f"# Discovered alphabet -- {meta.get('n_classes', len(letters))} classes "
                 f"from {meta.get('n_candidates', '?')} candidates, {meta.get('n_molecules', '?')} molecules")
    lines.append("")
    lines.append("Letters here are equivalence classes over field critical points, not atoms; "
                  "`atom_recovery`-style checks are a separate machinery sanity check and are not "
                  "part of this alphabet.")
    lines.append("")

    field_first = meta.get("field_first_methodology") or {}
    strict_field_first = (
        field_first.get("atomic_coordinates_used_to_discover_candidates") is False
        and str(field_first.get("candidate_discovery_method", "")).startswith("blind_scalar_field_")
    )
    if strict_field_first:
        lines.append(
            "Candidate discovery was field-first: atom coordinates were excluded from critical-point "
            "existence, type, descriptor and class assignment. Atom coincidence is only a post-hoc "
            "spatial comparison in the 3-D scenes."
        )
    else:
        lines.append(
            "> **Method warning:** this report has no confirmed field-first discovery provenance. "
            "Do not use it to decide whether a letter coincides with an atom; rerun the current "
            "`esm_master_analytics.py` extraction."
        )
    lines.append("")

    dynamics = meta.get("dynamics_and_context_status") or {}
    if dynamics:
        lines.append("## Dynamic/context status")
        lines.append(
            f"Observation regime: `{dynamics.get('observation_regime', 'unspecified')}`; "
            f"transition grammar inferred: `{dynamics.get('grammar_inferred')}`; "
            f"cross-context identity inferred: `{dynamics.get('cross_context_identity_inferred')}`."
        )
        if dynamics.get("identity_rule"):
            lines.append(str(dynamics["identity_rule"]))
        lines.append("")

    if visualization_manifest is not None:
        lines.append("## 3-D field-letter scenes")
        lines.append(
            "Each class links to concrete critical-point realisations, including ring/cage "
            "points in apparent voids. Open the Cube-Suite-style viewer with:"
        )
        lines.append("")
        lines.append(
            f"`streamlit run view_esm_alphabet.py -- --manifest {visualization_manifest}`"
        )
        lines.append("")
        lines.append(
            "The viewer overlays atoms as a post-hoc reference, all selected critical points, "
            "the descriptor-support shell and a selected density-field realisation. Its paired "
            "view compares two occurrences of one class without treating atom overlap as a label. "
            "Nuclear attractors are not reconstructed zero-flux Bader-basin surfaces."
        )
        lines.append("")

    if letter_archive_manifest is not None:
        lines.append("## Reusable analysis archive")
        lines.append(
            f"`{letter_archive_manifest}` contains all class centroids, every critical-point "
            "occurrence and its float64 invariant vector (memory-mappable), plus molecular "
            "metadata and references to the complete source cubes. It is the input for later "
            "analysis scripts; it is not a visual-only export."
        )
        lines.append("")

    lines.append("## Type breakdown")
    for t, stats in sorted(type_breakdown(letters).items(), key=lambda kv: -kv[1]["n_classes"]):
        ms = f"{stats['mean_stability']:.2f}" if stats["mean_stability"] is not None else "?"
        mm = "n/a" if stats["n_multi_molecule"] is None else str(stats["n_multi_molecule"])
        lines.append(
            f"- **{t}**: {stats['n_classes']} classes, {mm} cross-molecule, "
            f"{stats['n_named']} named, {stats['n_orphan_confident']} confident orphans, "
            f"mean stability {ms}"
        )
    lines.append("")

    named = named_letters(letters)
    lines.append(f"## Named letters ({len(named)}) -- the decoded portion")
    if not named:
        lines.append("_None cleared the significance + effect-size bar this run._")
    for l in named[:top_n]:
        lines.extend(_fmt_letter_block(l))
    lines.append("")

    orphans = confident_orphans(letters)
    lines.append(f"## Confident orphans ({len(orphans)}) -- candidates for undescribed function")
    lines.append("_Structurally stable, cross-molecule-reused, and equivalence-tested (not just "
                  "'not found yet') absence of correlation with anything named._")
    if all(l.n_molecules is None for l in letters):
        lines.append(
            "_No `n_molecules` data this run (see the Most-reused-letters note above) -- the "
            "reuse gate below could not be evaluated and was skipped, not enforced. Treat this "
            "list as confidence-only; it may include single-molecule findings._"
        )
    if not orphans:
        lines.append("_None cleared `min_confidence`/`min_molecules` this run._")
    for l in orphans[:top_n]:
        lines.extend(_fmt_letter_block(l))
    lines.append("")

    reused = most_reused_letters(letters, top_n)
    lines.append(f"## Most-reused letters (top {top_n})")
    if not reused:
        if all(l.n_molecules is None for l in letters):
            lines.append(
                "_No `n_molecules`/`n_occurrences` data in this report -- "
                "`slim_class()` only adds them when called with both "
                "`candidates` and `membership_index` non-None (see its "
                "docstring in esm_master_analytics.py). Check that call "
                "site: at STEP 12 (Write reports) in this run, at least one "
                "of the two was `None`._"
            )
        else:
            lines.append("_None reused across more than one molecule this run._")
    for l in reused:
        lines.extend(_fmt_letter_block(l))
    lines.append("")

    families = paradigm_families(letters, report)
    lines.append(f"## Paradigm families ({len(families)}) -- discovered declension patterns")
    if not families:
        lines.append("_None found this run._")
    for fam in families[:top_n]:
        rr = f"{fam['rule_reusability_score']:.2f}" if fam["rule_reusability_score"] is not None else "?"
        lines.append(f"  [{fam['paradigm_id']}] axis={fam['varying_axis']} reusability={rr}")
        for m in fam["members"]:
            lines.append(f"    - {m.one_line()}")
    lines.append("")

    det = distributional_determinate(letters)
    lines.append("## Distributional test verdicts (§3.1/§3.2) -- the determinate minority")
    for decision, group in det.items():
        lines.append(f"- **{decision}**: {len(group)} classes -- "
                      f"{', '.join(l.class_id for l in group[:10])}"
                      f"{' ...' if len(group) > 10 else ''}")
    if not det:
        lines.append("_All tested classes were indeterminate this run._")
    lines.append("")

    return "\n".join(lines)


def render_digest(
    report: dict,
    letters: List[Letter],
    visualization_manifest: Optional[Path] = None,
    letter_archive_manifest: Optional[Path] = None,
) -> dict:
    """Compact, structured (not prose) form of the same views, for feeding
    into something else rather than reading directly."""
    digest = {
        "n_letters": len(letters),
        "type_breakdown": type_breakdown(letters),
        "named_letters": [l.raw for l in named_letters(letters)],
        "confident_orphans": [l.raw for l in confident_orphans(letters)],
        "most_reused": [l.raw for l in most_reused_letters(letters, top_n=50)],
        "paradigm_families": [
            {
                "paradigm_id": f["paradigm_id"],
                "varying_axis": f["varying_axis"],
                "rule_reusability_score": f["rule_reusability_score"],
                "member_class_ids": [m.class_id for m in f["members"]],
            }
            for f in paradigm_families(letters, report)
        ],
        "distributional_determinate": {
            decision: [l.class_id for l in group]
            for decision, group in distributional_determinate(letters).items()
        },
        "dynamics_and_context_status": report.get("metadata", {}).get("dynamics_and_context_status"),
    }
    if visualization_manifest is not None:
        digest["visualization_manifest"] = str(visualization_manifest)
    if letter_archive_manifest is not None:
        digest["letter_archive_manifest"] = str(letter_archive_manifest)
    return digest


def render_summary_only(summary: dict) -> str:
    """Corpus-level context when only summary.json is available -- no
    per-letter data exists in this file at all (see module docstring)."""
    meta = summary.get("metadata", {})
    gates = summary.get("theory_gate_checks", {})
    lines = [
        f"# Corpus-level context ({meta.get('n_molecules', '?')} molecules, "
        f"{meta.get('n_candidates', '?')} candidates -> {meta.get('n_classes', '?')} classes)",
        "",
        "No per-letter data in summary.json -- pass --report for the alphabet itself.",
        "",
        f"- structure beats scrambled control: {gates.get('S15_2_structure_beats_scrambled_control')}",
        f"- fundamental candidates: {gates.get('S12_fundamentality', {}).get('n_fundamental')}"
        f" / {gates.get('S12_fundamentality', {}).get('n_candidates')}"
        f" (raw-candidate reuse test, not a per-class one -- see script docstring)",
        f"- semantics tested/named/orphan: {gates.get('S15_3_semantics')}",
        f"- isostere overall pass: {gates.get('S14_4_isostere_overall_pass')} "
        f"(quadrants: {gates.get('S14_4_isostere_quadrants')})",
    ]
    return "\n".join(lines)


# ==========================================================================
# CLI
# ==========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", help="path to esm_analytics_report.json")
    ap.add_argument("--summary", help="path to esm_analytics_summary.json (corpus-level only)")
    ap.add_argument("--top-n", type=int, default=15, help="examples per section in the markdown view")
    ap.add_argument("--min-orphan-confidence", type=float, default=0.7)
    ap.add_argument("--min-orphan-molecules", type=int, default=2)
    ap.add_argument("--json-out", help="write the structured digest here instead of/besides markdown")
    ap.add_argument("--out", help="write markdown here instead of stdout")
    ap.add_argument(
        "--visual-manifest", type=Path,
        help="manifest.json made by esm_master_analytics.py or export_esm_visualization.py",
    )
    ap.add_argument(
        "--letter-archive", type=Path,
        help="manifest.json of esm_letter_archive for downstream-analysis provenance",
    )
    args = ap.parse_args()

    if not args.report and not args.summary:
        ap.error("pass --report and/or --summary")

    if not args.report:
        summary = load_json(args.summary)
        text = render_summary_only(summary)
        (Path(args.out).write_text(text, encoding="utf-8") if args.out else print(text))
        return

    report_path = Path(args.report).resolve()
    report = load_json(args.report)
    visual_manifest_path = args.visual_manifest
    if visual_manifest_path is None:
        configured = (report.get("metadata", {}) or {}).get("visualization_manifest")
        candidate = (
            Path(configured)
            if configured
            else report_path.parent / "esm_visualization_data" / "manifest.json"
        )
        if candidate.exists():
            visual_manifest_path = candidate
    if visual_manifest_path is not None:
        visual_manifest_path = visual_manifest_path.resolve()
        if not visual_manifest_path.exists():
            ap.error(f"visualisation manifest not found: {visual_manifest_path}")
        merge_visualization_evidence(report, load_json(str(visual_manifest_path)))
    letter_archive_path = args.letter_archive
    if letter_archive_path is None:
        configured = (report.get("metadata", {}) or {}).get("letter_archive_manifest")
        candidate = (
            Path(configured)
            if configured
            else report_path.parent / "esm_letter_archive" / "manifest.json"
        )
        if candidate.exists():
            letter_archive_path = candidate
    if letter_archive_path is not None:
        letter_archive_path = letter_archive_path.resolve()
        if not letter_archive_path.exists():
            ap.error(f"letter archive manifest not found: {letter_archive_path}")
    letters = build_letters(report)

    if args.json_out:
        digest = render_digest(report, letters, visual_manifest_path, letter_archive_path)
        Path(args.json_out).write_text(json.dumps(digest, indent=2, ensure_ascii=False), encoding="utf-8")

    text = render_markdown(
        report, letters, args.top_n, visual_manifest_path, letter_archive_path
    )
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Interactive Cube-Suite-style 3-D view of ESM letter realisations.

Run after ``export_esm_visualization.py``:

    streamlit run view_esm_alphabet.py -- --manifest esm_visualization_data/manifest.json

The viewer draws the original molecule from the Gaussian cube atom block,
critical-point "letters" as coloured markers, and an optional density
isosurface. Selecting a marker can extract/cache a local cube around that
specific realisation, including ring and cage points located away from atoms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from esm_core.visualization import (
    TYPE_COLOURS,
    basin_boundary_relative_path,
    extract_field_component_boundary,
    extract_gradient_basin_boundaries,
    extract_local_field_cube,
    field_component_boundary_relative_path,
    field_slices_through_point,
    plane_through_points,
    validate_density_attractors,
)

from esm_core.config import AnalysisConfig
from esm_core.online_alphabet import online_prototype_evidence


ELEMENTS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
]

# PySCF `cubegen` writes Bohr coordinates with positive voxel counts, and the
# 3Dmol CUBE parser therefore rescales volumetric data by this factor on its
# own.  Geometry that is handed to 3Dmol through any *other* channel -- an XYZ
# model, addSphere, addLabel -- is interpreted as Angstrom and is not rescaled,
# so the viewer must apply the conversion itself or the isosurface will appear
# 0.529x the size of the molecule it belongs to.
#
# Only quantities that are *drawn in the 3-D scene* are converted.  Values that
# are fed back into the cube grid (extract_local_field_cube centres,
# field_slices_through_point, validate_density_attractors) must stay in native
# cube units, as must numeric coordinates reported in tables and captions.
BOHR_TO_ANGSTROM = 0.529177210903

def _magma_transfer_function(
    quantiles: Mapping[str, float],
    opacity_scale: float = 1.0,
) -> list[Dict[str, Any]]:
    """Build a 3Dmol volumetric transfer function from the same colour map,
    and the same per-molecule value scale, as the 2-D density sections.

    ``quantiles`` is the ``sections["quantiles"]`` dict already produced by
    ``field_slices_through_point`` for this realisation, so the two views
    share one colour-to-density mapping instead of two independently chosen
    ranges.

    Ray-marched compositing accumulates opacity as 1-(1-a)^n over the voxels
    a ray crosses. A local field cube at the default radius/resolution is
    roughly 30 voxels deep, so even a per-voxel opacity of 0.5 is fully
    opaque after 5-6 voxels -- the ray never reaches the far side, and the
    render looks like a solid, gradient-shaded box rather than a transparent
    density cloud. The base opacities here stay low enough (max ~0.16) that
    a path has to cross a genuinely dense, extended region before it
    saturates. ``opacity_scale`` is a live multiplier so this can still be
    tuned per molecule without editing these constants.
    """
    try:
        from matplotlib import colormaps
        magma = colormaps["magma"]
    except ImportError:
        import matplotlib.cm as cm
        magma = cm.get_cmap("magma")
    from matplotlib.colors import to_hex

    fractions = [0.05, 0.25, 0.50, 0.75, 0.90, 0.98]
    opacities = [0.000, 0.005, 0.02, 0.05, 0.09, 0.16]
    stops = []
    for fraction, opacity in zip(fractions, opacities):
        key = "0.5" if fraction == 0.50 and "0.5" in quantiles else str(fraction)
        if key not in quantiles:
            continue
        r, g, b, _ = magma(fraction)
        scaled_opacity = min(1.0, opacity * float(opacity_scale))
        stops.append({"color": to_hex((r, g, b)), "opacity": scaled_opacity, "value": float(quantiles[key])})
    return stops

def _project_onto_plane(
    position: Sequence[float],
    centroid: Sequence[float],
    u_hat: Sequence[float],
    v_hat: Sequence[float],
    normal_hat: Sequence[float],
) -> tuple[float, float, float]:
    """(u, v, signed distance to plane) for one point.

    All of ``position``, ``centroid``, and the basis vectors must already be
    in the same unit -- this function does no unit conversion. Used to place
    atoms and QTAIM critical points on the same 2-D frame that
    ``plane_through_points`` samples density on.
    """
    delta = np.asarray(position, dtype=float) - np.asarray(centroid, dtype=float)
    u_hat = np.asarray(u_hat, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    normal_hat = np.asarray(normal_hat, dtype=float)
    return float(np.dot(delta, u_hat)), float(np.dot(delta, v_hat)), float(np.dot(delta, normal_hat))

def _scene_xyz(position: Sequence[float]) -> tuple[float, float, float]:
    """Convert one native-cube-unit position into 3Dmol scene Angstrom."""
    x, y, z = (float(value) * BOHR_TO_ANGSTROM for value in position)
    return x, y, z


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cube-dir", type=Path, default=None)
    parser.add_argument("--patch-radius", type=float, default=3.0)
    args, _ = parser.parse_known_args()
    if args.manifest is None:
        args.manifest = AnalysisConfig.from_environment().paths.visualization_dir / "manifest.json"
    return args


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def _truncate_smiles(smiles: str, limit: int = 40) -> str:
    """Keep the selectbox row on one line for unusually long SMILES."""
    return smiles if len(smiles) <= limit else smiles[: limit - 1] + "…"


def _molecule_label(molecule_id: str, entries_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    """'qm9_000360 (CC(=O)Nc1ccc(O)cc1)', or the bare id if no SMILES is on record."""
    smiles = str((entries_by_id.get(molecule_id) or {}).get("smiles", "") or "")
    return f"{molecule_id} ({_truncate_smiles(smiles)})" if smiles else molecule_id

def _element_symbol(atomic_number: int) -> str:
    return ELEMENTS[atomic_number] if 0 < atomic_number < len(ELEMENTS) else "X"


def _xyz_from_atoms(atoms: Sequence[Mapping[str, Any]]) -> str:
    """Build an XYZ model for 3Dmol from the cube atom block.

    The cube atom block is in native cube units (Bohr for PySCF density
    cubes).  XYZ has no unit field and 3Dmol always reads it as Angstrom, so
    the conversion is applied here.  Besides fixing the scale mismatch against
    the isosurface, this also restores automatic bond perception: at Bohr
    magnitudes every interatomic distance is ~1.89x too large and 3Dmol finds
    no bonds at all, which is why the stick style renders nothing.
    """
    lines = [
        str(len(atoms)),
        "atoms read from the Gaussian cube header (converted to Angstrom for 3Dmol)",
    ]
    for atom in atoms:
        x, y, z = _scene_xyz(atom["position"])
        lines.append(
            f"{_element_symbol(int(atom['atomic_number']))} {x:.8f} {y:.8f} {z:.8f}"
        )
    return "\n".join(lines) + "\n"


def _occurrence_caption(occurrence: Mapping[str, Any]) -> str:
    distance = occurrence.get("descriptor_distance_to_class_centroid")
    distance_text = f", d={distance:.3g}" if distance is not None else ""
    return (
        f"{occurrence['critical_point_type']} | {occurrence.get('current_class_id', occurrence['class_id'])} "
        f"| ρ={float(occurrence['field_value']):.3g}{distance_text} "
        f"| {occurrence['occurrence_id']}"
    )


def _initial_isovalue(field_value: float) -> float:
    # Density values span many orders of magnitude near nuclei.  This is only
    # a starting value; the viewer always exposes a direct user control.
    return max(1e-5, min(1.0, abs(float(field_value)) * 0.1))


def _letter_evidence(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise old and new manifest records into an honest evidence view.

    The viewer deliberately shows *what has been checked*, rather than
    fabricating a confidence percentage for the ontological claim that a
    provisional descriptor group is a "letter".
    """
    supplied = record.get("evidence")
    if isinstance(supplied, Mapping):
        output = dict(supplied)
        output.setdefault("n_occurrences", record.get("n_occurrences"))
        output.setdefault("n_distinct_molecules_estimate", record.get("n_molecules"))
        return output
    stability = record.get("cluster_stability") or {}
    if isinstance(stability, Mapping) and stability.get("method"):
        return {
            "evidence_status": "batch_bootstrap_or_resampling_evaluated",
            "n_occurrences": record.get("n_occurrences"),
            "n_distinct_molecules_estimate": record.get("n_molecules"),
            "within_class_rms_distance": None,
            "assignment_threshold": record.get("persistence_threshold"),
            "bootstrap_method": stability.get("method"),
            "bootstrap_score": stability.get("score"),
            "bootstrap_n": stability.get("n"),
            "bootstrap_stale": stability.get("stale"),
            "validation_status": "batch_result; inspect method, score and stale flag",
            "not_a_correctness_probability": True,
        }
    return {
        "evidence_status": "legacy_or_scene_only_provisional",
        "n_occurrences": record.get("n_occurrences"),
        "n_distinct_molecules_estimate": record.get("n_molecules"),
        "within_class_rms_distance": None,
        "assignment_threshold": record.get("persistence_threshold"),
        "validation_status": "no online evidence block or batch stability result in this manifest",
        "not_a_correctness_probability": True,
    }


def _evidence_table(class_records: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for class_id, record in sorted(class_records.items()):
        evidence = _letter_evidence(record)
        rows.append({
            "class_id": class_id,
            "type": record.get("critical_point_type"),
            "evidence_status": evidence.get("evidence_status"),
            "occurrences": evidence.get("n_occurrences"),
            "distinct_molecules": evidence.get("n_distinct_molecules_estimate"),
            "relative_rms_to_assignment_threshold": evidence.get("relative_rms_to_threshold"),
            "merged_variants": evidence.get("merged_variant_count"),
            "bootstrap_score": evidence.get("bootstrap_score"),
            "bootstrap_stale": evidence.get("bootstrap_stale"),
            "named_correlates": len((record.get("semantic_status") or {}).get("named_correlates", [])),
        })
    return rows

def _molblock_with_bond_orders(
    atoms: Sequence[Mapping[str, Any]],
    smiles: str = "",
) -> tuple[Optional[str], str]:
    """Build an MDL molblock carrying real bond orders for the cube atoms.

    The cube atom block stores only elements and positions, so an XYZ model
    gives 3Dmol nothing but order-1 bonds and double/triple bonds can never be
    drawn.  Two sources of connectivity are tried, best first:

    1. ``smiles`` from the scene metadata.  For SMILES-derived records the
       PySCF molecule is built from ``Chem.AddHs(Chem.MolFromSmiles(...))`` in
       RDKit atom order, and ``cubegen`` writes atoms in PySCF order, so cube
       index i corresponds to RDKit index i.  The element sequence is verified
       before the topology is trusted; a mismatch falls through instead of
       drawing bonds that belong to a different molecule.
    2. Geometric bond perception via ``rdDetermineBonds``, which needs no
       metadata but infers orders from interatomic distances and can fail on
       radicals, unusual charges or poor geometry.

    Returns ``(molblock_or_None, route)``; ``route`` is reported in the UI so
    the displayed bond orders are never mistaken for a measured result.
    """
    if not atoms:
        return None, "no_atoms"
    if len(atoms) > 999:
        # V2000 counts line cannot express more than 999 atoms.
        return None, "too_many_atoms_for_molblock"
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D
    except ModuleNotFoundError:
        return None, "rdkit_unavailable"

    symbols = [_element_symbol(int(atom["atomic_number"])) for atom in atoms]
    positions = [_scene_xyz(atom["position"]) for atom in atoms]

    def _with_cube_geometry(mol: Any) -> str:
        """Replace any RDKit geometry with the actual cube coordinates."""
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index, (x, y, z) in enumerate(positions):
            conformer.SetAtomPosition(index, Point3D(x, y, z))
        mol.RemoveAllConformers()
        mol.AddConformer(conformer, assignId=True)
        return Chem.MolToMolBlock(mol, kekulize=True)

    if smiles:
        try:
            parsed = Chem.MolFromSmiles(smiles)
            if parsed is not None:
                topology = Chem.AddHs(parsed)
                if [a.GetSymbol() for a in topology.GetAtoms()] == symbols:
                    return _with_cube_geometry(Chem.RWMol(topology)), "smiles_topology"
        except Exception:  # noqa: BLE001 - never break the viewer over drawing
            pass

    try:
        raw = Chem.MolFromXYZBlock(_xyz_from_atoms(atoms))
        if raw is not None:
            perceived = Chem.RWMol(raw)
            rdDetermineBonds.DetermineBonds(perceived, charge=0)
            return _with_cube_geometry(perceived), "geometry_perception"
    except Exception:  # noqa: BLE001
        return None, "perception_failed"
    return None, "no_topology"


BOND_SOURCE_NOTES = {
    "smiles_topology": (
        "Bond orders come from the record SMILES, index-matched to the cube atom block. "
        "They are input provenance, not a result measured from the density field."
    ),
    "geometry_perception": (
        "Bond orders were perceived from interatomic distances by RDKit because no matching "
        "SMILES was available. Treat them as a drawing aid only."
    ),
    "perception_failed": "Bond-order perception failed; atoms are drawn with single bonds only.",
    "rdkit_unavailable": "RDKit is not installed; atoms are drawn with single bonds only.",
    "no_topology": "No connectivity source was available; atoms are drawn with single bonds only.",
    "too_many_atoms_for_molblock": "Molecule is too large for a V2000 molblock; single bonds only.",
    "no_atoms": "This scene has no cube atom block.",
}


def _add_scene_to_viewer(
    viewer: Any,
    scene: Mapping[str, Any],
    visible: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    *,
    show_labels: bool,
    marker_radius: float,
    show_descriptor_support: bool,
    stick_radius: float = 0.05,
    sphere_scale: float = 0.20,
) -> str:
    header = scene.get("cube_header") or {}
    atoms = header.get("atoms", [])
    bond_source = "no_atoms"
    if atoms:
        smiles = str((scene.get("metadata") or {}).get("smiles", "") or "")
        molblock, bond_source = _molblock_with_bond_orders(atoms, smiles)
        if molblock is not None:
            # An SDF model carries explicit bond orders, so 3Dmol can draw the
            # second and third lines of double and triple bonds.
            viewer.addModel(molblock, "sdf")
        else:
            viewer.addModel(_xyz_from_atoms(atoms), "xyz")
        viewer.setStyle({}, {
            "stick": {"radius": float(stick_radius), "singleBonds": False},
            "sphere": {"scale": float(sphere_scale)},
        })
    # `marker_radius` is a cosmetic slider calibrated against the Bohr-scale
    # scene, so it is converted with the geometry to keep markers the same size
    # relative to the molecule as before.  `outer_radius` is a physical
    # descriptor radius in cube units and must be converted for correctness.
    scene_marker_radius = float(marker_radius) * BOHR_TO_ANGSTROM
    for occurrence in visible:
        x, y, z = _scene_xyz(occurrence["position"])
        type_colour = occurrence.get(
            "type_colour", TYPE_COLOURS.get(occurrence["critical_point_type"], "#ffffff")
        )
        viewer.addSphere({
            "center": {"x": x, "y": y, "z": z},
            "radius": scene_marker_radius,
            "color": type_colour,
            "opacity": 0.90,
        })
        if show_labels:
            viewer.addLabel(occurrence.get("current_class_id", occurrence["class_id"]), {
                "position": {"x": x, "y": y, "z": z},
                "fontColor": "white",
                "backgroundColor": "#222222",
                "backgroundOpacity": 0.72,
                "fontSize": 10,
                "showBackground": True,
            })
    if selected is not None:
        x, y, z = _scene_xyz(selected["position"])
        viewer.addSphere({
            "center": {"x": x, "y": y, "z": z},
            "radius": scene_marker_radius * 1.8,
            "color": selected.get("class_colour", "#ffffff"),
            "wireframe": True,
            "opacity": 1.0,
        })
        support = selected.get("descriptor_support") or {}
        outer_radius = support.get("outer_radius")
        if show_descriptor_support and outer_radius is not None and float(outer_radius) > 0.0:
            viewer.addSphere({
                "center": {"x": x, "y": y, "z": z},
                "radius": float(outer_radius) * BOHR_TO_ANGSTROM,
                "color": selected.get("class_colour", "#ffffff"),
                "wireframe": True,
                "opacity": 0.30,
            })
    return bond_source


def _find_occurrence(scene: Mapping[str, Any], occurrence_id: str) -> Optional[Mapping[str, Any]]:
    return next(
        (item for item in scene.get("letters", []) if item.get("occurrence_id") == occurrence_id),
        None,
    )


def main() -> None:
    try:
        import py3Dmol
        import streamlit as st
        import streamlit.components.v1 as components
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Install the viewer dependencies first: pip install streamlit py3Dmol"
        ) from exc

    args = _parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise SystemExit(f"visualisation manifest not found: {manifest_path}")
    manifest_root = manifest_path.parent
    manifest = _load_json(manifest_path)
    cube_dir = (
        args.cube_dir.resolve() if args.cube_dir is not None
        else (manifest_root / manifest.get("cube_dir_hint", "../cube_files_db")).resolve()
    )
    class_records = {item["class_id"]: item for item in manifest.get("letter_classes", [])}
    class_aliases = manifest.get("class_aliases", {}) or {}

    def resolve_class_id(class_id: Any) -> str:
        current = str(class_id)
        seen = set()
        while current in class_aliases and current not in seen:
            seen.add(current)
            current = str(class_aliases[current])
        return current

    # An older visual manifest is intentionally immutable.  Enrich its
    # display-only evidence from the current online sufficient statistics
    # when they sit beside it, without rewriting scenes or assignments.
    online_state_path = manifest_root.parent / "online" / "online_alphabet_state.json"
    online_state = _load_json(online_state_path) if online_state_path.exists() else {}
    prototypes_by_id = {
        str(prototype.get("class_id")): prototype
        for prototypes in online_state.get("prototypes_by_type", {}).values()
        if isinstance(prototypes, list)
        for prototype in prototypes
        if isinstance(prototype, Mapping) and prototype.get("class_id")
    }
    threshold_by_type = online_state.get("threshold_by_type", {}) or {}
    for class_id, record in class_records.items():
        canonical_id = resolve_class_id(class_id)
        prototype = prototypes_by_id.get(canonical_id)
        if prototype is None or isinstance(record.get("evidence"), Mapping):
            continue
        field_type = str(record.get("critical_point_type", ""))
        record["evidence"] = online_prototype_evidence(
            prototype, float(threshold_by_type.get(field_type, 0.0)),
        )
        record.setdefault("n_occurrences", int(prototype.get("count", 0)))
        record.setdefault("n_molecules", record["evidence"].get("n_distinct_molecules_estimate"))
    molecules = manifest.get("molecules", [])
    if not molecules:
        raise SystemExit("manifest contains no molecular scenes")

    st.set_page_config(page_title="ESM alphabet viewer", layout="wide")
    st.title("ESM alphabet: field letters in molecular space")
    provenance = manifest.get("candidate_discovery_provenance", {}) or {}
    field_first_confirmed = (
        str(provenance.get("candidate_discovery_method", "")).startswith("blind_scalar_field_")
        and provenance.get("atomic_coordinates_used_to_discover_candidates") is False
    )
    if field_first_confirmed:
        st.caption(
            "Markers are discovered from the density field only. Atoms come from the original cube header "
            "as a separate reference overlay; they do not define a marker or its class. "
            "Ring/cage markers remain visible in apparent voids. Nuclear-attractor markers are not "
            "exported zero-flux Bader-basin surfaces."
        )
    else:
        st.warning(
            "This bundle has legacy/unknown candidate-discovery provenance. It is useful for visual "
            "inspection, but must not be interpreted as a field-first test. Rerun esm_master_analytics.py "
            "with the current field-first cache version for the strict method."
        )

    evidence_rows = _evidence_table(class_records)
    evidence_counts: Dict[str, int] = {}
    for row in evidence_rows:
        status = str(row.get("evidence_status", "unknown"))
        evidence_counts[status] = evidence_counts.get(status, 0) + 1
    evidence_columns = st.columns(4)
    evidence_columns[0].metric("Current displayed classes", len(evidence_rows))
    evidence_columns[1].metric(
        "Classes with batch resampling",
        sum(row["evidence_status"] == "batch_bootstrap_or_resampling_evaluated" for row in evidence_rows),
    )
    evidence_columns[2].metric(
        "Online provisional classes",
        sum(str(row["evidence_status"]).startswith(("exploratory", "provisional", "reused_streaming")) for row in evidence_rows),
    )
    evidence_columns[3].metric(
        "Classes with named correlations",
        sum(int(row["named_correlates"] or 0) > 0 for row in evidence_rows),
    )
    with st.expander("Letter evidence and uncertainty", expanded=False):
        st.caption(
            "These fields quantify support, compactness and completed validation steps. They are not a probability "
            "that a pattern is a correct letter, nor evidence of grammar, dynamics or atomic identity."
        )
        st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
        st.json({"evidence_status_count": evidence_counts})

    molecule_ids = [item["molecule_id"] for item in molecules]
    molecule_entries_by_id = {item["molecule_id"]: item for item in molecules}
    known_class_ids = sorted(
        class_id
        for class_id, record in class_records.items()
        if (record.get("semantic_status", {}) or {}).get("named_correlates")
    )
    jump_to_known = st.sidebar.selectbox(
        "Jump to named letter", ["—"] + known_class_ids,
        help="Selects a molecule containing one realisation of a known-correlate class.",
    )
    preferred_molecule = molecule_ids[0]
    if jump_to_known != "—":
        examples = class_records[jump_to_known].get("example_realisations", [])
        if examples:
            preferred_molecule = examples[0].get("molecule_id", preferred_molecule)
    requested_molecule = st.sidebar.selectbox(
        "Molecule", molecule_ids,
        index=molecule_ids.index(preferred_molecule) if preferred_molecule in molecule_ids else 0,
        format_func=lambda mol_id: _molecule_label(mol_id, molecule_entries_by_id),
    )
    molecule_entry = next(item for item in molecules if item["molecule_id"] == requested_molecule)
    scene_path = manifest_root / molecule_entry["scene_file"]
    scene = _load_json(scene_path)
    molecule_smiles = str((scene.get("metadata") or {}).get("smiles", "") or "")
    st.header(requested_molecule + (f"  ·  `{molecule_smiles}`" if molecule_smiles else ""))
    all_occurrences = scene.get("letters", [])
    # Historical scenes keep their raw append-time ID as evidence.  Resolve
    # it only for current filters, labels and comparisons; no scene JSON is
    # rewritten when a later online prototype merge occurs.
    for occurrence in all_occurrences:
        occurrence["current_class_id"] = resolve_class_id(occurrence.get("class_id"))
    source_cube = cube_dir / str(scene.get("cube_file")) if scene.get("cube_file") else None

    present_types = sorted({item["critical_point_type"] for item in all_occurrences})
    selected_types = st.sidebar.multiselect(
        "Critical-point types", present_types, default=present_types,
        help="Classes are never mixed across these QTAIM types during assignment.",
    )
    known_only = st.sidebar.checkbox(
        "Only known / named letters", value=False,
        help="Uses named correlates from the ESM report; unlabelled classes remain exploratory.",
    )
    present_class_ids = sorted({item["current_class_id"] for item in all_occurrences})
    selected_class_ids = st.sidebar.multiselect(
        "Restrict to class IDs", present_class_ids,
        default=[jump_to_known] if jump_to_known in present_class_ids else [],
    )
    max_markers = st.sidebar.slider("Maximum displayed markers", 1, max(1, len(all_occurrences)), min(80, max(1, len(all_occurrences))))
    marker_radius = st.sidebar.slider("Critical-point marker radius", 0.03, 0.45, 0.13, 0.01)
    stick_radius = st.sidebar.slider(
        "Bond stick radius (Å)", 0.01, 0.20, 0.05, 0.01,
        help=(
            "Scene geometry is drawn in Angstrom. Values above roughly 0.10 make the sticks "
            "thick enough to hide the second and third lines of multiple bonds."
        ),
    )
    sphere_scale = st.sidebar.slider(
        "Atom sphere scale", 0.05, 0.60, 0.20, 0.01,
        help="Fraction of the van der Waals radius. Large spheres swallow short multiple bonds.",
    )
    show_labels = st.sidebar.checkbox("Show class labels", value=False)
    show_descriptor_support = st.sidebar.checkbox(
        "Show descriptor-support shell",
        value=False,
        help=(
            "Shows the outer sampled radius of the selected local field descriptor. "
            "It is neither an atom nor a QTAIM basin boundary."
        ),
    )

    def visible(item: Mapping[str, Any]) -> bool:
        if item["critical_point_type"] not in selected_types:
            return False
        current_class_id = item.get("current_class_id", item["class_id"])
        if selected_class_ids and current_class_id not in selected_class_ids:
            return False
        if known_only:
            record = class_records.get(current_class_id, {})
            if not (record.get("semantic_status", {}) or {}).get("named_correlates"):
                return False
        return True

    visible_occurrences = [item for item in all_occurrences if visible(item)]
    visible_occurrences.sort(
        key=lambda item: (
            item["critical_point_type"],
            item.get("current_class_id", item["class_id"]),
            item.get("descriptor_distance_to_class_centroid") is None,
            item.get("descriptor_distance_to_class_centroid") or np.inf,
        )
    )
    displayed_occurrences = visible_occurrences[:max_markers]

    selection_labels = [_occurrence_caption(item) for item in visible_occurrences]
    selected_occurrence: Optional[Mapping[str, Any]] = None
    if selection_labels:
        selected_caption = st.selectbox("Selected field-letter realisation", selection_labels)
        selected_occurrence = visible_occurrences[selection_labels.index(selected_caption)]
    else:
        st.warning("No letter realisations match the active filters.")

    if selected_occurrence is not None:
        selected_class_id = str(selected_occurrence.get("current_class_id", selected_occurrence["class_id"]))
        selected_record = class_records.get(selected_class_id, {})
        selected_evidence = _letter_evidence(selected_record)
        st.subheader("Evidence for the selected provisional class")
        selected_columns = st.columns(4)
        selected_columns[0].metric("Evidence status", str(selected_evidence.get("evidence_status", "unknown")))
        selected_columns[1].metric("Occurrences", selected_evidence.get("n_occurrences") or "n/a")
        selected_columns[2].metric(
            "Distinct molecules",
            selected_evidence.get("n_distinct_molecules_estimate") or "not recorded",
        )
        rms_ratio = selected_evidence.get("relative_rms_to_threshold")
        selected_columns[3].metric(
            "RMS / assignment threshold",
            f"{float(rms_ratio):.3g}" if rms_ratio is not None else "not available",
        )
        st.caption(
            "For the selected occurrence: field root stage="
            f"{selected_occurrence.get('field_discovery_stage', 'unknown')}; "
            f"|∇ρ|={selected_occurrence.get('gradient_norm_at_field_root', 'not recorded')}; "
            "nearest-atom fields below are reference overlays only."
        )
        with st.expander("Selected-class validation detail", expanded=False):
            st.json(selected_evidence)

    show_field = st.checkbox("Show density isosurfaces", value=selected_occurrence is not None)
    field_path: Optional[Path] = None
    if selected_occurrence is not None:
        local_field_path = manifest_root / selected_occurrence["local_field_cube"]
        if not local_field_path.exists() and source_cube is not None and source_cube.exists():
            if st.button("Extract local field cube for selected letter"):
                with st.spinner("Extracting a local density cube around the selected critical point…"):
                    extract_local_field_cube(
                        source_cube,
                        selected_occurrence["position"],
                        local_field_path,
                        radius=args.patch_radius,
                    )
                st.rerun()
        use_full_cube = st.checkbox(
            "Use complete molecular cube instead of local patch",
            value=False,
            help="May use more RAM and transfer more data to the browser. The local field patch is the default.",
        )
        if use_full_cube and source_cube is not None and source_cube.exists():
            field_path = source_cube
            st.caption("Complete molecular cube selected; it may use more RAM than the local patch.")
        else:
            field_path = local_field_path if local_field_path.exists() else None
        if field_path == local_field_path and local_field_path.exists():
            st.caption(f"Local realisation field: `{local_field_path.relative_to(manifest_root)}`")
        elif field_path is None and source_cube is not None and source_cube.exists():
            st.caption(
                "Extract the local cube to display this realisation. The complete molecular "
                "density is not transferred unless the fallback switch is explicitly enabled."
            )

    nuclear_attractor_positions = [
        occurrence["position"]
        for occurrence in all_occurrences
        if occurrence["critical_point_type"] == "nuclear_attractor"
    ]
    basin_boundary_path: Optional[Path] = None
    if len(nuclear_attractor_positions) >= 2:
        audit_attractors = st.checkbox(
            "Audit field-detected nuclear attractors on this cube grid",
            value=False,
            help=(
                "Checks each field root against nearby density samples before it can be used as a watershed marker. "
                "No atom position participates in the audit."
            ),
        )
        if audit_attractors and source_cube is not None and source_cube.exists():
            with st.spinner("Checking field roots against the density grid…"):
                attractor_audit = validate_density_attractors(source_cube, nuclear_attractor_positions)
            st.caption(
                "Grid-scale attractor audit (diagnostic only; a finite cube does not certify a converged zero-flux topology):"
            )
            st.dataframe(attractor_audit["attractors"], use_container_width=True, hide_index=True)
        show_basin_boundaries = st.checkbox(
            "Show classical gradient-basin boundaries (approximate)", value=False,
            help=(
                "On-demand marker-controlled topographic watershed of the source density from field-detected "
                "attractors, snapped to nearby density peaks; not a certified zero-flux surface."
            ),
        )
        candidate_boundary_path = manifest_root / basin_boundary_relative_path(scene["molecule_id"])
        if show_basin_boundaries:
            if not candidate_boundary_path.exists() and source_cube is not None and source_cube.exists():
                if st.button("Extract gradient-basin boundary overlay"):
                    with st.spinner("Computing a discrete gradient-basin overlay from the density grid…"):
                        extract_gradient_basin_boundaries(
                            source_cube,
                            nuclear_attractor_positions,
                            candidate_boundary_path,
                        )
                    st.rerun()
            basin_boundary_path = candidate_boundary_path if candidate_boundary_path.exists() else None
            if basin_boundary_path is not None:
                st.caption(
                    "Classical basin overlay: a grid-watershed approximation from detected nuclear attractors; "
                    "it is not claimed as a converged zero-flux QTAIM surface."
                )

    field_letter_boundary_path: Optional[Path] = None
    field_letter_boundary_metadata: Optional[Dict[str, Any]] = None
    if selected_occurrence is not None:
        show_field_letter_boundary = st.checkbox(
            "Show a crisp boundary for the selected field feature",
            value=False,
            help=(
                "Extracts the connected component of the density field around the selected critical point, "
                "then draws its exact grid-level surface. This uses no atom positions. The chosen threshold "
                "is an explicit operational boundary, not a claim that every letter has one unique physical border."
            ),
        )
        if show_field_letter_boundary:
            eigenvalues = np.asarray(selected_occurrence.get("hessian_eigenvalues", []), dtype=float)
            if eigenvalues.size == 3 and np.all(eigenvalues < 0):
                default_side_index = 0
            elif eigenvalues.size == 3 and np.all(eigenvalues > 0):
                default_side_index = 1
            else:
                default_side_index = 0
            boundary_side_label = st.selectbox(
                "Field component around the root",
                ["Higher-density side (ρ ≥ level)", "Lower-density side (ρ ≤ level)"],
                index=default_side_index,
                help=(
                    "For a maximum, the higher-density side is usually the compact local feature; "
                    "for a minimum, the lower-density side is usually the compact feature. At saddles both "
                    "choices are exploratory and may extend beyond the local crop."
                ),
            )
            boundary_side = "above" if boundary_side_label.startswith("Higher") else "below"
            boundary_fraction_pct = st.slider(
                "Boundary level: fraction of field contrast from background to root (%)",
                min_value=2, max_value=90, value=2, step=1,
                help=(
                    "For the higher-density side, level = cube minimum + fraction × "
                    "(sampled root value − cube minimum). For the lower-density side, the analogous level "
                    "is measured down from the cube maximum. Larger fractions move the cut toward the root "
                    "and usually make the component smaller. Compare values to assess boundary stability."
                ),
            )
            boundary_source_scope = "full" if field_path is not None and source_cube is not None and field_path == source_cube else "local"
            boundary_relative_path = field_component_boundary_relative_path(
                str(scene["molecule_id"]),
                int(selected_occurrence["global_candidate_index"]),
                boundary_side,
                boundary_fraction_pct / 100.0,
                source_scope=boundary_source_scope,
            )
            field_letter_boundary_path = manifest_root / boundary_relative_path
            metadata_path = field_letter_boundary_path.with_suffix(".json")
            boundary_exists = field_letter_boundary_path.exists()
            if field_letter_boundary_path.exists() and metadata_path.exists():
                try:
                    field_letter_boundary_metadata = _load_json(metadata_path)
                except (OSError, json.JSONDecodeError):
                    field_letter_boundary_metadata = None
            if field_path is None or not field_path.exists():
                st.info("Extract the selected realisation's local field cube first, or enable the complete-cube fallback.")
                field_letter_boundary_path = None
            else:
                if boundary_exists and field_letter_boundary_metadata:
                    st.caption(
                        "White surface = the selected field component's threshold boundary; "
                        f"Boundary level ρ={field_letter_boundary_metadata['level']:.5g}; "
                        f"component volume={field_letter_boundary_metadata['volume']:.5g} "
                        f"{field_letter_boundary_metadata['coordinate_units']}³."
                    )
                    if field_letter_boundary_metadata.get("boundary_is_clipped"):
                        st.warning(
                            "This component reaches the cube edge, so the displayed boundary is clipped. "
                            "Increase the local patch radius or use the complete molecular cube."
                        )
                elif boundary_exists:
                    st.caption("Boundary cube exists, but its metadata is missing or unreadable.")
                button_label = "Recompute crisp field boundary" if boundary_exists else "Compute crisp field boundary"
                if st.button(button_label, type="primary"):
                    with st.spinner("Finding the connected field component around the selected root…"):
                        try:
                            field_letter_boundary_metadata = extract_field_component_boundary(
                                field_path,
                                selected_occurrence["position"],
                                field_letter_boundary_path,
                                side=boundary_side,
                                level_fraction=boundary_fraction_pct / 100.0,
                            )
                        except (ValueError, RuntimeError, OSError) as exc:
                            st.error(f"Could not extract this field boundary: {exc}")
                        else:
                            with metadata_path.open("w", encoding="utf-8") as handle:
                                json.dump(field_letter_boundary_metadata, handle, ensure_ascii=False, indent=2)
                            st.rerun()

    default_iso = _initial_isovalue(selected_occurrence["field_value"]) if selected_occurrence else 0.03
    isovalue = st.sidebar.number_input("Density isovalue", min_value=1e-8, value=float(default_iso), format="%.6g")
    field_colour = st.sidebar.color_picker("Isosurface colour", "#66d9ef")
    n_density_levels = st.sidebar.slider(
        "Nested density levels", 1, 5, 3,
        help=(
            "Several nested isosurfaces expose internal density layers. They are samples of the scalar field, "
            "not walls of a rigid or hollow object."
        ),
    )
    density_level_ratio = st.sidebar.slider("Level ratio", 1.15, 4.0, 1.8, 0.05)
    show_field_sections = st.checkbox(
        "Show internal density sections through the selected letter",
        value=selected_occurrence is not None,
        help="Three grid-aligned scalar-field slices reveal internal maxima, minima and saddle-like structure.",
    )
    show_volumetric_render = st.checkbox(
        "Show 3D volumetric density render",
        value=False,
        help=(
            "Ray-marches the same local density grid as the sections below, coloured with the same "
            "magma scale, so the internal structure that the 2-D slices show is also visible directly "
            "in 3-D instead of only as flat cuts. Requires WebGL2; heavier to render than the isosurfaces."
        ),
    )
    volumetric_opacity_scale = st.sidebar.slider(
        "Volumetric render opacity", 0.1, 1.0, 0.5, 0.1,
        help=(
            "Multiplies every opacity stop. A local cube is crossed by roughly 30 voxels, so opacity "
            "accumulates fast; raise this only until the interior structure starts to disappear behind "
            "the entry face, then back off."
        ),
    ) if show_volumetric_render else 1.0

    sections = None
    can_read_sections = selected_occurrence is not None and field_path is not None and field_path.exists()
    if (show_field_sections or show_volumetric_render) and can_read_sections:
        with st.spinner("Reading scalar-field sections…"):
            sections = field_slices_through_point(field_path, selected_occurrence["position"])

    viewer = py3Dmol.view(width=1050, height=730)
    viewer.setBackgroundColor("#111318")
    bond_source = _add_scene_to_viewer(
        viewer, scene, displayed_occurrences, selected_occurrence,
        show_labels=show_labels, marker_radius=marker_radius,
        show_descriptor_support=show_descriptor_support,
        stick_radius=stick_radius, sphere_scale=sphere_scale,
    )
    if show_field and field_path is not None and field_path.exists():
        field_text = field_path.read_text(encoding="utf-8")
        # Outer-to-inner field levels, intentionally transparent enough that
        # their nesting is legible rather than rendered as a single shell.
        level_colours = [field_colour, "#48c9b0", "#f5b041", "#ec7063", "#af7ac5"]
        for level_index in range(n_density_levels):
            level = float(isovalue) * (float(density_level_ratio) ** level_index)
            viewer.addVolumetricData(
                field_text, "cube",
                {
                    "isoval": level,
                    "color": level_colours[level_index],
                    "opacity": max(0.18, 0.54 - 0.07 * level_index),
                },
            )
    if show_volumetric_render and field_path is not None and field_path.exists():
        if sections is None:
            st.warning("Extract the local field cube first to enable the volumetric render.")
        else:
            field_text = field_path.read_text(encoding="utf-8")
            transferfn = _magma_transfer_function(sections["quantiles"], volumetric_opacity_scale)
            if transferfn:
                # `addVolumetricData` (not `addVolumetricRender`) is what py3Dmol
                # can drive with a raw cube string; passing a `transferfn` spec
                # instead of `isoval` switches it from an isosurface to a
                # ray-marched volume render, so this is the same call already
                # used for the isosurfaces above, with a different spec shape.
                viewer.addVolumetricData(field_text, "cube", {"transferfn": transferfn})
            else:
                st.warning("Not enough density variation in this local cube to build a volumetric render.")

    if basin_boundary_path is not None:
        viewer.addVolumetricData(
            basin_boundary_path.read_text(encoding="utf-8"), "cube",
            {"isoval": 0.5, "color": "#ff5c8a", "opacity": 0.28},
        )
    if field_letter_boundary_path is not None and field_letter_boundary_path.exists():
        viewer.addVolumetricData(
            field_letter_boundary_path.read_text(encoding="utf-8"), "cube",
            {"isoval": 0.5, "color": "#ffffff", "opacity": 0.82},
        )
    viewer.zoomTo()
    viewer.render()
    components.html(viewer._make_html(), height=750, scrolling=False)
    st.caption(BOND_SOURCE_NOTES.get(bond_source, f"Bond-order source: {bond_source}."))
    if show_volumetric_render and sections is None and can_read_sections is False:
        st.info("Select a letter realisation and extract its local field cube to enable the volumetric render.")

    if show_field_sections and selected_occurrence is not None:
        if field_path is None or not field_path.exists():
            st.info("Extract the local field cube first to view internal density sections.")
        elif sections is not None:
            st.subheader("Internal density structure of this realisation")
            st.caption(
                "These are sampled density values through the selected field point. They complement the nested "
                "isosurfaces and must not be interpreted as a solid boundary of a letter."
            )
            try:
                import matplotlib.pyplot as plt
                figure, axes = plt.subplots(1, 3, figsize=(13, 4))
                all_values = np.concatenate([np.ravel(value) for value in sections["slices"].values()])
                lower, upper = np.quantile(all_values, [0.02, 0.98])
                for axis, (name, values) in zip(axes, sections["slices"].items()):
                    image = axis.imshow(values.T, origin="lower", cmap="magma", vmin=lower, vmax=upper)
                    axis.set_title(name.replace("_", " "))
                    axis.set_xlabel("grid index")
                figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="density")
                st.pyplot(figure, clear_figure=True, use_container_width=True)
            except ModuleNotFoundError:
                st.warning("matplotlib is unavailable; density sections cannot be plotted in this environment.")
            st.json({
                "selected_grid_index": sections["grid_index"],
                "density_minimum": sections["minimum"],
                "density_maximum": sections["maximum"],
                "density_quantiles": sections["quantiles"],
            })

    st.subheader("Arbitrary-plane cross-section through three atoms")
    plane_atoms = (scene.get("cube_header") or {}).get("atoms", [])
    if source_cube is None or not source_cube.exists():
        st.info("This scene has no source cube on disk; a cross-section needs one to sample.")
    elif len(plane_atoms) < 3:
        st.info("This scene's cube header has fewer than three atoms; a plane needs three.")
    else:
        st.caption(
            "The embedded 3-D view above is a static render with no click-back channel to this script "
            "(py3Dmol's `components.html` embed is one-way), so the three atoms are picked from the "
            "dropdowns below rather than by clicking them in the viewer. A genuine click-to-pick would "
            "need a custom bidirectional Streamlit component; this reaches the same result -- three "
            "chosen atoms -> one plane -- without that extra machinery."
        )

        def _plane_atom_label(index: int) -> str:
            atom = plane_atoms[index]
            x, y, z = _scene_xyz(atom["position"])
            symbol = _element_symbol(int(atom["atomic_number"]))
            return f"#{index} {symbol}  ({x:.2f}, {y:.2f}, {z:.2f} Å)"

        atom_columns = st.columns(3)
        default_c = min(2, len(plane_atoms) - 1)
        atom_index_a = atom_columns[0].selectbox(
            "Atom A (sets the u-axis origin)", range(len(plane_atoms)),
            index=0, format_func=_plane_atom_label, key="plane_atom_a",
        )
        atom_index_b = atom_columns[1].selectbox(
            "Atom B (sets the u-axis direction)", range(len(plane_atoms)),
            index=min(1, len(plane_atoms) - 1), format_func=_plane_atom_label, key="plane_atom_b",
        )
        atom_index_c = atom_columns[2].selectbox(
            "Atom C (completes the plane)", range(len(plane_atoms)),
            index=default_c, format_func=_plane_atom_label, key="plane_atom_c",
        )

        if len({atom_index_a, atom_index_b, atom_index_c}) < 3:
            st.warning("Pick three different atoms to define a plane.")
        else:
            plane_extent_ang = st.slider(
                "Cross-section half-extent (Å)", 1.0, 15.0, 6.0, 0.5,
                help="How far the sampled plane extends from the centroid of the three atoms, in each in-plane direction.",
            )
            plane_resolution = st.slider(
                "Cross-section sampling grid", 40, 200, 100, 10,
                help="Higher is smoother but slower: cost grows with the square of this value (100 -> ~0.15 s, 200 -> ~0.6 s).",
            )
            plane_tolerance_ang = st.slider(
                "Atom / QTAIM in-plane tolerance (Å)", 0.1, 3.0, 0.6, 0.1,
                help="Atoms and critical points farther than this from the plane are left off the cross-section.",
            )
            if st.button("Compute cross-section", key="plane_compute_button"):
                point_a = plane_atoms[atom_index_a]["position"]
                point_b = plane_atoms[atom_index_b]["position"]
                point_c = plane_atoms[atom_index_c]["position"]
                try:
                    with st.spinner("Sampling the density field on the chosen plane…"):
                        plane = plane_through_points(
                            source_cube, point_a, point_b, point_c,
                            half_extent=float(plane_extent_ang) / BOHR_TO_ANGSTROM,
                            n_samples=int(plane_resolution),
                        )
                except ValueError as exc:
                    st.warning(f"Could not build a plane from these atoms: {exc}")
                else:
                    centroid = plane["centroid"]
                    u_hat, v_hat, normal_hat = plane["u_axis"], plane["v_axis"], plane["normal"]
                    tolerance_native = float(plane_tolerance_ang) / BOHR_TO_ANGSTROM

                    on_plane_atoms = []
                    for index, atom in enumerate(plane_atoms):
                        u, v, distance = _project_onto_plane(atom["position"], centroid, u_hat, v_hat, normal_hat)
                        if abs(distance) <= tolerance_native:
                            on_plane_atoms.append({
                                "u": u * BOHR_TO_ANGSTROM,
                                "v": v * BOHR_TO_ANGSTROM,
                                "symbol": _element_symbol(int(atom["atomic_number"])),
                                "is_defining": index in (atom_index_a, atom_index_b, atom_index_c),
                            })

                    on_plane_letters = []
                    for occurrence in all_occurrences:
                        u, v, distance = _project_onto_plane(
                            occurrence["position"], centroid, u_hat, v_hat, normal_hat
                        )
                        if abs(distance) <= tolerance_native:
                            on_plane_letters.append({
                                "u": u * BOHR_TO_ANGSTROM,
                                "v": v * BOHR_TO_ANGSTROM,
                                "type": occurrence["critical_point_type"],
                                "class_id": occurrence.get("current_class_id", occurrence["class_id"]),
                            })

                    try:
                        import matplotlib.pyplot as plt
                        figure, axis = plt.subplots(figsize=(7.5, 7))
                        axis_coords_ang = np.asarray(plane["u_coords"]) * BOHR_TO_ANGSTROM
                        density = np.asarray(plane["density"])
                        mesh = axis.pcolormesh(
                            axis_coords_ang, axis_coords_ang, density.T,
                            cmap="magma", shading="auto",
                        )
                        figure.colorbar(mesh, ax=axis, shrink=0.85, label="density (native units)")

                        # An isosurface intersected with a plane is a contour
                        # line at that density value; reuse the same nested
                        # levels the 3-D isosurfaces above use so the two
                        # views agree with each other.
                        levels = sorted({
                            float(isovalue) * (float(density_level_ratio) ** level_index)
                            for level_index in range(n_density_levels)
                        })
                        levels = [level for level in levels if density.min() < level < density.max()]
                        if levels:
                            axis.contour(
                                axis_coords_ang, axis_coords_ang, density.T,
                                levels=levels, colors="#66d9ef", linewidths=1.2,
                            )

                        for atom_point in on_plane_atoms:
                            axis.scatter(
                                atom_point["u"], atom_point["v"],
                                s=260 if atom_point["is_defining"] else 110,
                                facecolor="none",
                                edgecolor="white" if atom_point["is_defining"] else "#aaaaaa",
                                linewidths=1.6, zorder=5,
                            )
                            axis.annotate(
                                atom_point["symbol"], (atom_point["u"], atom_point["v"]),
                                color="white", fontsize=9, ha="center", va="center", zorder=6,
                            )
                        for letter_point in on_plane_letters:
                            colour = TYPE_COLOURS.get(letter_point["type"], "#ffffff")
                            axis.scatter(
                                letter_point["u"], letter_point["v"],
                                s=90, color=colour, marker="D",
                                edgecolor="black", linewidths=0.6, zorder=7,
                            )

                        axis.set_xlabel("in-plane u (Å, atom A → atom B)")
                        axis.set_ylabel("in-plane v (Å)")
                        axis.set_aspect("equal")
                        axis.set_title("Density on the plane through the three chosen atoms")
                        st.pyplot(figure, clear_figure=True, use_container_width=True)
                    except ModuleNotFoundError:
                        st.warning("matplotlib is unavailable; the cross-section cannot be plotted in this environment.")

                    st.caption(
                        "Diamonds are QTAIM critical-point realisations from this scene, coloured by critical-point "
                        f"type; circles are atoms (bold circle = one of the three defining atoms). Both are shown "
                        f"only within ±{plane_tolerance_ang:.2g} Å of the plane, projected onto it -- an off-plane "
                        "point is not moved onto the cut to make it visible, so a sparse cross-section usually means "
                        "the interesting structure is not coplanar with these three atoms, not that it is absent. "
                        "Cyan contour lines mark where the same isovalue levels used for the 3-D nested isosurfaces "
                        "intersect this plane; they are read off the interpolated grid, not measured independently. "
                        "Density values outside the source cube's own extent are clamped to its boundary."
                    )

    st.subheader("What is on this scene")
    st.write(
        f"{len(all_occurrences)} extracted critical-point realisations in molecule; "
        f"{len(visible_occurrences)} match filters; {len(displayed_occurrences)} are rendered."
    )
    st.dataframe([
        {
            "class_id": item.get("current_class_id", item["class_id"]),
            "class_id_at_scene_append": item["class_id"],
            "type": item["critical_point_type"],
            "class_evidence_status": _letter_evidence(
                class_records.get(item.get("current_class_id", item["class_id"]), {})
            ).get("evidence_status"),
            "field_value": item["field_value"],
            "field_root_gradient_norm": item.get("gradient_norm_at_field_root"),
            "field_discovery_stage": item.get("field_discovery_stage"),
            "distance_to_centroid": item.get("descriptor_distance_to_class_centroid"),
            "descriptor_outer_radius": (item.get("descriptor_support") or {}).get("outer_radius"),
            "nearest_atom_distance": (item.get("nearest_atom_reference") or {}).get("distance"),
            "nearest_atom_Z": (item.get("nearest_atom_reference") or {}).get("atomic_number"),
            "position": ", ".join(f"{float(value):.3f}" for value in item["position"]),
        }
        for item in visible_occurrences
    ], use_container_width=True, hide_index=True)

    st.subheader("Compare one discovered class in two molecular fields")
    comparison_class_ids = [
        class_id
        for class_id, record in sorted(class_records.items())
        if len(record.get("example_realisations", [])) >= 2
    ]
    if comparison_class_ids:
        default_comparison_class = (
            selected_occurrence.get("current_class_id", selected_occurrence.get("class_id"))
            if selected_occurrence is not None and selected_occurrence.get("current_class_id", selected_occurrence.get("class_id")) in comparison_class_ids
            else comparison_class_ids[0]
        )
        comparison_class_id = st.selectbox(
            "Field-letter class for comparison",
            comparison_class_ids,
            index=comparison_class_ids.index(default_comparison_class),
            help=(
                "The two panels show separate field realisations assigned to one invariant class. "
                "The atom layer is retained only to make coincidence or non-coincidence inspectable."
            ),
        )
        comparison_examples = class_records[comparison_class_id]["example_realisations"]

        def comparison_label(index: int) -> str:
            example = comparison_examples[index]
            distance = example.get("descriptor_distance_to_class_centroid")
            distance_text = f", d={distance:.3g}" if distance is not None else ""
            return f"{example['molecule_id']} | {example['occurrence_id']}{distance_text}"

        first_index = st.selectbox(
            "First realisation", range(len(comparison_examples)), format_func=comparison_label,
            key="comparison_first",
        )
        alternative_indices = [index for index in range(len(comparison_examples)) if index != first_index]
        second_index = st.selectbox(
            "Second realisation", alternative_indices, format_func=comparison_label,
            key="comparison_second",
        )
        show_comparison_fields = st.checkbox(
            "Show local density fields in both comparison panels",
            value=False,
            help=(
                "Local cubes are extracted only for the two selected occurrences. "
                "This avoids transferring whole molecular fields into the browser."
            ),
            key="comparison_show_fields",
        )
        molecule_entries_by_id = {item["molecule_id"]: item for item in molecules}
        comparison_columns = st.columns(2)
        for column, example_index in zip(comparison_columns, (first_index, second_index)):
            example = comparison_examples[example_index]
            entry = molecule_entries_by_id.get(example["molecule_id"])
            with column:
                if entry is None:
                    st.warning(f"Scene for {example['molecule_id']} is not present in this bundle.")
                    continue
                comparison_scene = _load_json(manifest_root / entry["scene_file"])
                comparison_occurrence = _find_occurrence(comparison_scene, example["occurrence_id"])
                if comparison_occurrence is None:
                    st.warning(f"Occurrence {example['occurrence_id']} is absent from its scene.")
                    continue
                comparison_field_path = manifest_root / comparison_occurrence["local_field_cube"]
                comparison_source_cube = (
                    cube_dir / str(comparison_scene.get("cube_file"))
                    if comparison_scene.get("cube_file") else None
                )
                if (
                    show_comparison_fields
                    and not comparison_field_path.exists()
                    and comparison_source_cube is not None
                    and comparison_source_cube.exists()
                ):
                    if st.button(
                        "Extract this local field",
                        key=f"comparison_patch_{comparison_occurrence['occurrence_id']}",
                    ):
                        extract_local_field_cube(
                            comparison_source_cube,
                            comparison_occurrence["position"],
                            comparison_field_path,
                            radius=args.patch_radius,
                        )
                        st.rerun()
                comparison_viewer = py3Dmol.view(width=500, height=440)
                comparison_viewer.setBackgroundColor("#111318")
                _add_scene_to_viewer(
                    comparison_viewer,
                    comparison_scene,
                    [comparison_occurrence],
                    comparison_occurrence,
                    show_labels=False,
                    marker_radius=marker_radius,
                    show_descriptor_support=show_descriptor_support,
                    stick_radius=stick_radius,
                    sphere_scale=sphere_scale,
                )
                if show_comparison_fields and comparison_field_path.exists():
                    comparison_viewer.addVolumetricData(
                        comparison_field_path.read_text(encoding="utf-8"), "cube",
                        {"isoval": float(isovalue), "color": field_colour, "opacity": 0.55},
                    )
                comparison_viewer.zoomTo()
                comparison_viewer.render()
                components.html(comparison_viewer._make_html(), height=455, scrolling=False)
                reference = comparison_occurrence.get("nearest_atom_reference") or {}
                st.caption(
                    f"{comparison_occurrence['critical_point_type']}; nearest atom reference: "
                    f"Z={reference.get('atomic_number', 'n/a')}, "
                    f"distance={reference.get('distance', 'n/a')} (cube units)."
                )
                if show_comparison_fields and not comparison_field_path.exists():
                    st.caption("Select “Extract this local field” to add this field realisation.")
    else:
        st.info("This scene bundle has no class with two displayed molecular realisations yet.")

    st.subheader("Legend")
    st.markdown("  ".join(
        f"<span style='color:{colour}'>●</span> `{field_type}`"
        for field_type, colour in manifest.get("type_colours", TYPE_COLOURS).items()
    ), unsafe_allow_html=True)
    st.caption(
        "Numeric coordinates, atom-reference distances and descriptor radii in the tables and captions "
        "are in native cube units (PySCF density cubes normally use Bohr). The 3-D scene is drawn in "
        "Angstrom so that the atomic model and the volumetric isosurfaces share one scale. "
        "The shell is a descriptor sampling region, not a letter volume."
    )


if __name__ == "__main__":
    main()

"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
umcs_datasets.py — Загрузчики датасетов для обучения и дообучения UMCS.

Поддерживаемые источники
─────────────────────────
  qm9         QM9 (молекулярные квантовые свойства, ~134k молекул)
  oqmd        OQMD v1.8 — Open Quantum Materials Database  (REST API / bulk dump)
  mp          Materials Project (mp-api / pymatgen, требует API-ключ)
  jarvis      JARVIS-DFT / JARVIS-CFID (figshare или ColabFit mirror)
  aflow       AFLOW (REST API, без ключа)
  cod         Crystallography Open Database (REST API, без ключа)
  nomad       NOMAD Archive (REST API, без ключа)
  colabfit    ColabFit DS_u8strp7hm0cy_0 — JARVIS_CFID_OQMD зеркало
  hdf5        Любой HDF5-файл в формате UMCS (ключи smiles / targets)
  csv         Произвольный CSV
  json        Произвольный JSON / JSON-Lines

Унифицированный выход
─────────────────────
Каждая функция load_*_records() возвращает:
  List[Dict[str, Any]]  — каждый dict содержит обязательные поля:
    smiles   (str)   — SMILES молекулы / Poscar / формула-заглушка
    targets  (Dict[str, float])  — целевые свойства в единицах SI/eV

Записи без SMILES автоматически преобразуются через pymatgen или ase
(если установлены); если ни один конвертер недоступен, запись пропускается
и в журнал выводится предупреждение.

Использование в umcs_interface.py
──────────────────────────────────
  from umcs_datasets import DATASET_REGISTRY, load_dataset_records

  records = load_dataset_records("oqmd", n_samples=500, api_key=None)
  records = load_dataset_records("mp", n_samples=200, api_key="YOUR_KEY")

Использование из CLI (через umcs_interface.py)
─────────────────────────────────────────────
  python umcs_interface.py train-dataset --dataset oqmd --max-samples 500
  python umcs_interface.py train-dataset --dataset mp   --max-samples 200 \\
        --api-key YOUR_MP_KEY
  python umcs_interface.py train-dataset --dataset jarvis --max-samples 1000
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import pathlib
import shutil
import tempfile
import time
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _try_import(package: str, install_hint: str = ""):
    """Ленивый импорт с понятным сообщением об ошибке."""
    try:
        import importlib
        return importlib.import_module(package)
    except ImportError:
        hint = f"  pip install {install_hint or package}" if install_hint else ""
        raise ImportError(
            f"Package '{package}' not installed.\n{hint}\n"
            f"Install it or skip this dataset source."
        )


def _get_session():
    """requests.Session с retry."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _download_file(url: str, dest: pathlib.Path, chunk_size: int = 1 << 20) -> pathlib.Path:
    """Скачивает файл с прогрессом."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[download] {url} → {dest}")
    with _get_session().get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(f"\r[download] {pct:5.1f}%  {downloaded >> 20}MB / {total >> 20}MB",
                          end="", flush=True)
    print()
    return dest

def _pdb_to_xyz(pdb_text: str) -> Optional[str]:
    """Парсит текст PDB-файла (RCSB) и извлекает XYZ координаты атомов."""
    lines = pdb_text.splitlines()
    atoms = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Попытка прочитать символ из колонок 77-78
            sym = line[76:78].strip()
            if not sym:
                # Fallback: вырезаем из имени атома
                sym = line[12:16].strip()
                import re
                sym = re.sub(r'\d+', '', sym)
                sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append((sym, x, y, z))
            except ValueError:
                pass
    if not atoms:
        return None
    out = [str(len(atoms)), "RCSB PDB Structure"]
    for sym, x, y, z in atoms:
        out.append(f"{sym} {x} {y} {z}")
    return "\n".join(out)

def _structure_to_smiles(structure, backend: str = "auto") -> Optional[str]:
    """Конвертирует pymatgen/ase Structure → SMILES через доступный бэкенд."""
    # Для кристаллических структур SMILES не определён;
    # используем формулу как идентификатор (UMCS обработает через build_from_formula).
    # Настоящий SMILES генерируется только для молекул из XYZ/SDF.
    try:
        if hasattr(structure, "formula"):
            return None  # сигнал: использовать formula fallback
    except Exception:
        pass
    return None


def _formula_record(formula: str, targets: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Создаёт запись через формулу (без SMILES) — только если formula непустая."""
    formula = (formula or "").strip()
    if not formula:
        return None
    # UMCS принимает ключ 'formula' как альтернативу 'smiles'
    # в normalize_training_records (будет добавлено ниже через патч).
    return {"formula": formula, "smiles": formula, "targets": targets}


def _truncate(records: List, n: Optional[int]) -> List:
    if n and n > 0:
        return records[:n]
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 1. QM9
# ─────────────────────────────────────────────────────────────────────────────

# Карта меток QM9 → имена голов UMCS
QM9_TARGET_MAP: Dict[str, str] = {
    "mu":    "dipole_moment",       # D
    "alpha": "polarizability",      # a₀³
    "homo":  "homo_energy",         # eV
    "lumo":  "lumo_energy",         # eV
    "gap":   "electronic_gap",      # eV
    "zpve":  "zpve",                # eV
    "u0":    "internal_energy_0k",  # eV
    "u298":  "internal_energy_298k",
    "h298":  "enthalpy_298k",
    "g298":  "gibbs_energy_298k",
    "cv":    "heat_capacity_298k",  # cal/mol/K
    "r2":    "electronic_spatial_extent",
}

QM9_CSV_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
)
QM9_RAW_URL = (
    "https://ndownloader.figstatic.com/files/3195389"  # original .tar.bz2
)


def load_qm9_records(
    n_samples: Optional[int] = None,
    csv_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    target_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает QM9 из CSV (deepchem mirror) или локального кэша.

    Parameters
    ----------
    n_samples   : максимальное число записей (None = все)
    csv_path    : локальный путь / URL к CSV-файлу (переопределяет дефолт)
    cache_dir   : папка для кэширования
    target_names: список целевых имён из QM9_TARGET_MAP (None = все)
    """
    import pandas as pd
    import requests

    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    local_csv = cache / "qm9.csv"

    url = csv_path or QM9_CSV_URL
    if csv_path and pathlib.Path(csv_path).exists():
        local_csv = pathlib.Path(csv_path)
    elif not local_csv.exists():
        _download_file(url, local_csv)

    df = pd.read_csv(local_csv, nrows=n_samples)

    target_keys = target_names or list(QM9_TARGET_MAP.keys())
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        smiles = str(row.get("smiles") or row.get("SMILES") or "").strip()
        if not smiles:
            continue
        targets: Dict[str, float] = {}
        for col, head_name in QM9_TARGET_MAP.items():
            if col not in target_keys and head_name not in target_keys:
                continue
            val = row.get(col)
            if val is not None:
                try:
                    targets[head_name] = float(val)
                except (TypeError, ValueError):
                    pass
        if targets:
            records.append({"smiles": smiles, "targets": targets})

    logger.info(f"[qm9] loaded {len(records)} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. OQMD v1.8
#    REST API:  https://oqmd.org/oqmdapi/
#    Bulk dump: https://oqmd.org/download/  (SQL dump + CSV)
# ─────────────────────────────────────────────────────────────────────────────

OQMD_API_BASE = "https://oqmd.org/oqmdapi/formationenergy"

# Карта полей OQMD → UMCS heads
OQMD_TARGET_MAP: Dict[str, str] = {
    "delta_e":          "formation_energy",        # eV/atom
    "stability":        "thermodynamic_stability",  # eV/atom above hull
    "band_gap":         "electronic_gap",           # eV
    "magnetic_moment":  "magnetic_moment",          # μB/f.u.
    "volume":           "unit_cell_volume",         # Å³
}


def load_oqmd_records(
    n_samples: Optional[int] = 500,
    cache_dir: Optional[str] = None,
    filters: Optional[Dict[str, str]] = None,
    api_base: str = OQMD_API_BASE,
) -> List[Dict[str, Any]]:
    """
    Загружает записи из OQMD через REST API.

    Parameters
    ----------
    n_samples : сколько записей скачать (None = все, осторожно — >1M)
    filters   : дополнительные фильтры REST API, например
                {"element_set": "Fe,O", "stability": "lte:0.1"}
    api_base  : базовый URL (переопределяется для зеркал)

    Пример фильтра через CLI:
      --oqmd-filter element_set=Fe,O --oqmd-filter stability=lte:0.1

    Примечание: OQMD отдаёт кристаллические структуры.
    SMILES генерируется как химическая формула (используется build_from_formula).
    """
    import requests

    limit = min(n_samples or 100, 100)  # OQMD max page = 100
    total_needed = n_samples or 10_000_000
    records: List[Dict[str, Any]] = []
    offset = 0

    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache_file = cache / f"oqmd_{total_needed}.jsonl"
    cache.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        logger.info(f"[oqmd] loading from cache {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
                if len(records) >= total_needed:
                    break
        return records

    session = _get_session()
    print(f"[oqmd] fetching up to {total_needed} records from {api_base}")
    with open(cache_file, "w", encoding="utf-8") as fh_cache:
        while len(records) < total_needed:
            params: Dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "fields": "name,composition,delta_e,stability,band_gap,volume",
                "format": "json",
            }
            if filters:
                params.update(filters)

            try:
                resp = session.get(api_base, params=params, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                logger.warning(f"[oqmd] request failed at offset {offset}: {exc}")
                break

            data = payload.get("data") or payload.get("results") or []
            if not data:
                break

            for item in data:
                formula = (item.get("name") or item.get("composition") or "").strip()
                if not formula:
                    continue
                targets: Dict[str, float] = {}
                for field, head in OQMD_TARGET_MAP.items():
                    val = item.get(field)
                    if val is not None:
                        try:
                            targets[head] = float(val)
                        except (TypeError, ValueError):
                            pass
                if targets:
                    rec = {"smiles": formula, "formula": formula, "targets": targets,
                           "source": "oqmd"}
                    records.append(rec)
                    fh_cache.write(json.dumps(rec) + "\n")

            offset += limit
            if len(data) < limit:
                break  # конец данных
            print(f"\r[oqmd] fetched {len(records)} records", end="", flush=True)
            time.sleep(0.1)

    print()
    logger.info(f"[oqmd] loaded {len(records)} records total")
    return _truncate(records, n_samples)


def load_oqmd_dump_records(
    dump_path: str,
    n_samples: Optional[int] = None,
    sep: str = ",",
) -> List[Dict[str, Any]]:
    """
    Загружает OQMD из локального CSV-дампа (oqmd.org/download/).

    Ожидаемые колонки (любые из):
      composition, delta_e, stability, band_gap, volume, magnetic_moment
    """
    import pandas as pd

    path = pathlib.Path(dump_path)
    if not path.exists():
        raise FileNotFoundError(f"OQMD dump not found: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, sep=sep, nrows=n_samples)

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        formula = str(row.get("composition") or row.get("name") or "").strip()
        if not formula:
            continue
        targets: Dict[str, float] = {}
        for col, head in OQMD_TARGET_MAP.items():
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    targets[head] = float(val)
                except (TypeError, ValueError):
                    pass
        if targets:
            records.append({"smiles": formula, "formula": formula,
                             "targets": targets, "source": "oqmd_dump"})

    logger.info(f"[oqmd_dump] loaded {len(records)} records from {path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3. Materials Project
#    Требует: pip install mp-api
#    API-ключ: https://materialsproject.org/api
# ─────────────────────────────────────────────────────────────────────────────

MP_TARGET_MAP: Dict[str, str] = {
    "formation_energy_per_atom": "formation_energy",
    "band_gap":                  "electronic_gap",
    "energy_above_hull":         "thermodynamic_stability",
    "volume":                    "unit_cell_volume",
    "density":                   "density",
    "total_magnetization":       "magnetic_moment",
}


def load_mp_records(
    n_samples: Optional[int] = 500,
    api_key: Optional[str] = None,
    elements: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает структуры из Materials Project через mp-api.

    Parameters
    ----------
    api_key  : ключ MP (или переменная окружения MP_API_KEY)
    elements : фильтр по элементам, например ["Fe", "O"]
    fields   : список полей MP (по умолчанию — все из MP_TARGET_MAP)

    Требует:
      pip install mp-api

    Пример:
      records = load_mp_records(n_samples=200, api_key="YOUR_KEY",
                                elements=["Li", "Fe", "O"])
    """
    mp_api = _try_import("mp_api", "mp-api")
    from mp_api.client import MPRester

    key = api_key or os.environ.get("MP_API_KEY") or ""
    if not key:
        raise ValueError(
            "Materials Project API key required.\n"
            "  Получить: https://materialsproject.org/api\n"
            "  Передать: --api-key KEY  или  export MP_API_KEY=KEY"
        )

    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    el_tag = "_".join(sorted(elements or [])) or "all"
    cache_file = cache / f"mp_{el_tag}_{n_samples or 'all'}.jsonl"

    if cache_file.exists():
        logger.info(f"[mp] loading from cache {cache_file}")
        records: List[Dict[str, Any]] = []
        with open(cache_file, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
                if n_samples and len(records) >= n_samples:
                    break
        return records

    requested_fields = fields or list(MP_TARGET_MAP.keys()) + ["formula_pretty", "material_id"]

    records = []
    print(f"[mp] querying Materials Project (elements={elements}, n={n_samples})")
    with MPRester(key) as mpr:
        kwargs: Dict[str, Any] = {
            "fields": requested_fields,
            "num_chunks": None,
        }
        if elements:
            kwargs["elements"] = elements
        if n_samples:
            kwargs["chunk_size"] = min(n_samples, 1000)

        entries = mpr.summary.search(**kwargs)  # type: ignore[arg-type]

    with open(cache_file, "w", encoding="utf-8") as fh_cache:
        for entry in (entries or []):
            formula = getattr(entry, "formula_pretty", None) or ""
            targets: Dict[str, float] = {}
            for attr, head in MP_TARGET_MAP.items():
                val = getattr(entry, attr, None)
                if val is not None:
                    try:
                        targets[head] = float(val)
                    except (TypeError, ValueError):
                        pass
            if targets:
                rec = {"smiles": formula, "formula": formula,
                       "material_id": getattr(entry, "material_id", ""),
                       "targets": targets, "source": "mp"}
                records.append(rec)
                fh_cache.write(json.dumps(rec) + "\n")
            if n_samples and len(records) >= n_samples:
                break

    logger.info(f"[mp] loaded {len(records)} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4. JARVIS-DFT / JARVIS-CFID
#    Требует: pip install jarvis-tools  (для figshare) или HTTP-fetch
#    ColabFit mirror: https://materials.colabfit.org/id/DS_u8strp7hm0cy_0
# ─────────────────────────────────────────────────────────────────────────────

JARVIS_DATASETS: Dict[str, str] = {
    "dft_3d":        "JARVIS-DFT 3D (76k структур, оптические, магнитные, упругие)",
    "dft_2d":        "JARVIS-DFT 2D (1k структур)",
    "cfid_3d":       "JARVIS-CFID 3D (дескрипторы + DFT свойства)",
    "qe_tb":         "JARVIS-TB (tight-binding)",
    "jff":           "JARVIS-FF (force-field)",
}

JARVIS_TARGET_MAP: Dict[str, str] = {
    "formation_energy_peratom": "formation_energy",
    "optb88vdw_bandgap":        "electronic_gap",
    "mbj_bandgap":              "electronic_gap_mbj",
    "bulk_modulus_kv":          "bulk_modulus",
    "shear_modulus_gv":         "shear_modulus",
    "epsx":                     "dielectric_eps_x",
    "mepsx":                    "mepsx",
    "magmom_oszicar":           "magnetic_moment",
    "slme":                     "solar_cell_efficiency",
    "spillage":                 "spillage",
    "n-Seebeck":                "seebeck_n",
    "p-Seebeck":                "seebeck_p",
    "n-powerfact":              "power_factor_n",
    "p-powerfact":              "power_factor_p",
    "exfoliation_energy":       "exfoliation_energy",
    "ehull":                    "thermodynamic_stability",
}

# Прямой URL к JSON-файлам JARVIS на figshare (нет регистрации)
JARVIS_FIGSHARE_URLS: Dict[str, str] = {
    "dft_3d":  "https://figshare.com/ndownloader/files/40705723",  # dft_3d_2023
    "dft_2d":  "https://figshare.com/ndownloader/files/40705720",
    "cfid_3d": "https://figshare.com/ndownloader/files/40705726",
}

# ColabFit mirror для JARVIS_CFID_OQMD
COLABFIT_DS_URL = "https://materials.colabfit.org/id/DS_u8strp7hm0cy_0"
COLABFIT_API_BASE = "https://materials.colabfit.org/api"


def load_jarvis_records(
    n_samples: Optional[int] = 1000,
    dataset_name: str = "dft_3d",
    cache_dir: Optional[str] = None,
    use_jarvis_tools: bool = True,
) -> List[Dict[str, Any]]:
    """
    Загружает JARVIS через jarvis-tools (figshare) или прямой HTTP-фетч.

    Parameters
    ----------
    dataset_name    : "dft_3d" | "dft_2d" | "cfid_3d" | "qe_tb" | "jff"
    use_jarvis_tools: True — использовать pip install jarvis-tools
                      False — прямой HTTP-фетч (не требует jarvis-tools)

    Требует (опционально):
      pip install jarvis-tools
    """
    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"jarvis_{dataset_name}_{n_samples or 'all'}.jsonl"

    if cache_file.exists():
        logger.info(f"[jarvis] loading from cache {cache_file}")
        records: List[Dict[str, Any]] = []
        with open(cache_file, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
                if n_samples and len(records) >= n_samples:
                    break
        return records

    raw_data: Optional[List[Dict]] = None

    # Попытка 1: jarvis-tools
    if use_jarvis_tools:
        try:
            from jarvis.db.figshare import data as jarvis_data  # type: ignore
            print(f"[jarvis] loading {dataset_name} via jarvis-tools …")
            raw_data = jarvis_data(dataset_name)
        except ImportError:
            logger.info("[jarvis] jarvis-tools not installed, falling back to HTTP fetch")
        except Exception as exc:
            logger.warning(f"[jarvis] jarvis-tools failed: {exc}; falling back to HTTP")

    # Попытка 2: прямой HTTP-фетч JSON с figshare
    if raw_data is None:
        url = JARVIS_FIGSHARE_URLS.get(dataset_name)
        if url is None:
            raise ValueError(
                f"Unknown JARVIS dataset '{dataset_name}'. "
                f"Available: {list(JARVIS_FIGSHARE_URLS)}\n"
                f"Install jarvis-tools for more: pip install jarvis-tools"
            )
        tmp = cache / f"jarvis_{dataset_name}.json.gz"
        if not tmp.exists():
            _download_file(url, tmp)
        print(f"[jarvis] parsing {tmp} …")
        opener = gzip.open if tmp.suffix == ".gz" else open
        try:
            with opener(tmp, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            with open(tmp, "rb") as fh:
                raw = fh.read()
            try:
                raw_data_text = gzip.decompress(raw).decode("utf-8")
            except Exception:
                raw_data_text = raw.decode("utf-8", errors="replace")
            payload = json.loads(raw_data_text)
        raw_data = payload if isinstance(payload, list) else payload.get("data", [])

    records: List[Dict[str, Any]] = []
    with open(cache_file, "w", encoding="utf-8") as fh_cache:
        for item in (raw_data or []):
            if not isinstance(item, dict):
                continue
            formula = (item.get("formula") or item.get("composition") or
                       item.get("compound") or "").strip()
            if not formula:
                # пробуем восстановить из atoms dict
                atoms = item.get("atoms")
                if isinstance(atoms, dict):
                    formula = atoms.get("composition", "")
            if not formula:
                continue
            targets: Dict[str, float] = {}
            for key, head in JARVIS_TARGET_MAP.items():
                val = item.get(key)
                if val is not None and val != "na" and val != "N/A":
                    try:
                        fval = float(val)
                        if not (np.isnan(fval) or np.isinf(fval)):
                            targets[head] = fval
                    except (TypeError, ValueError):
                        pass
            if targets:
                rec = {"smiles": formula, "formula": formula,
                       "jid": item.get("jid", ""),
                       "targets": targets, "source": f"jarvis_{dataset_name}"}
                records.append(rec)
                fh_cache.write(json.dumps(rec) + "\n")
            if n_samples and len(records) >= n_samples:
                break

    logger.info(f"[jarvis] loaded {len(records)} records from {dataset_name}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 5. AFLOW (aflow.org)
#    REST API без ключей: http://aflow.org/API/aflowlib.out
# ─────────────────────────────────────────────────────────────────────────────

AFLOW_API_BASE = "http://aflow.org/API/aflowlib.out"

AFLOW_TARGET_MAP: Dict[str, str] = {
    "Egap":             "electronic_gap",
    "enthalpy_atom":    "formation_energy",
    "volume_atom":      "volume_per_atom",
    "spin_atom":        "magnetic_moment_per_atom",
    "Debye_temperature":"debye_temperature",
    "bulk_modulus_voigt":"bulk_modulus",
}


def load_aflow_records(
    n_samples: Optional[int] = 500,
    elements: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает записи AFLOW через REST API.

    Parameters
    ----------
    elements : фильтр по элементам (например ["Ti", "O"])
    """
    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    el_tag = "_".join(sorted(elements or [])) or "all"
    cache_file = cache / f"aflow_{el_tag}_{n_samples or 'all'}.jsonl"

    if cache_file.exists():
        records: List[Dict[str, Any]] = []
        with open(cache_file, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
                if n_samples and len(records) >= n_samples:
                    break
        return records

    import requests
    session = _get_session()
    records = []
    paging_size = min(n_samples or 200, 200)

    fields = ["compound", "Egap", "enthalpy_atom", "volume_atom",
              "spin_atom", "Debye_temperature", "bulk_modulus_voigt"]
    preamble = f"?format=json&paging={paging_size}&fields={','.join(fields)}"
    if elements:
        preamble += f"&species({','.join(elements)})"

    url = AFLOW_API_BASE + preamble
    print(f"[aflow] fetching {url}")

    with open(cache_file, "w", encoding="utf-8") as fh_cache:
        page = 1
        while True:
            try:
                resp = session.get(url + f"&page={page}", timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning(f"[aflow] page {page} failed: {exc}")
                break

            if not data:
                break

            for item in (data if isinstance(data, list) else [data]):
                formula = (item.get("compound") or "").strip()
                if not formula:
                    continue
                targets: Dict[str, float] = {}
                for key, head in AFLOW_TARGET_MAP.items():
                    val = item.get(key)
                    if val is not None and val != "":
                        try:
                            targets[head] = float(val)
                        except (TypeError, ValueError):
                            pass
                if targets:
                    rec = {"smiles": formula, "formula": formula,
                           "targets": targets, "source": "aflow"}
                    records.append(rec)
                    fh_cache.write(json.dumps(rec) + "\n")
                if n_samples and len(records) >= n_samples:
                    break

            if n_samples and len(records) >= n_samples:
                break
            if len(data) < paging_size:
                break
            page += 1
            time.sleep(0.2)
            print(f"\r[aflow] fetched {len(records)}", end="", flush=True)

    print()
    logger.info(f"[aflow] loaded {len(records)} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 6. COD — Crystallography Open Database
#    REST API: https://www.crystallography.net/cod/result?
# ─────────────────────────────────────────────────────────────────────────────

COD_API_BASE = "https://www.crystallography.net/cod/result"


def load_cod_records(
    n_samples: Optional[int] = 200,
    elements: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает из COD через REST API (только структурные данные, без свойств DFT).
    Возвращает записи с formula + volume + Z (число формульных единиц).

    Parameters
    ----------
    elements : строка вроде "Fe-O" или "Li-Co-O"
    """
    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    el_tag = (elements or "all").replace("-", "_")
    cache_file = cache / f"cod_{el_tag}_{n_samples}.jsonl"

    if cache_file.exists():
        records: List[Dict[str, Any]] = []
        with open(cache_file, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
                if n_samples and len(records) >= n_samples:
                    break
        return records

    import requests
    params: Dict[str, Any] = {
        "format": "json",
        "limit":  min(n_samples or 200, 200),
    }
    if elements:
        params["el"] = elements

    session = _get_session()
    records = []
    with open(cache_file, "w", encoding="utf-8") as fh_cache:
        try:
            resp = session.get(COD_API_BASE, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"[cod] request failed: {exc}")
            return []

        for item in (data if isinstance(data, list) else []):
            formula = (item.get("formula") or "").strip()
            if not formula:
                continue
            targets: Dict[str, float] = {}
            vol = item.get("vol")
            if vol:
                try:
                    targets["unit_cell_volume"] = float(vol)
                except (TypeError, ValueError):
                    pass
            z = item.get("Z") or item.get("z")
            if z:
                try:
                    targets["z_formula_units"] = float(z)
                except (TypeError, ValueError):
                    pass
            if targets:
                rec = {"smiles": formula, "formula": formula,
                       "cod_id": item.get("file", ""),
                       "targets": targets, "source": "cod"}
                records.append(rec)
                fh_cache.write(json.dumps(rec) + "\n")
            if n_samples and len(records) >= n_samples:
                break

    logger.info(f"[cod] loaded {len(records)} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 7. NOMAD Archive
#    REST API v1: https://nomad-lab.eu/prod/v1/api/v1/
# ─────────────────────────────────────────────────────────────────────────────

NOMAD_API_BASE = "https://nomad-lab.eu/prod/v1/api/v1"

NOMAD_TARGET_MAP: Dict[str, str] = {
    "results.properties.electronic.band_structure_electronic.band_gap.value":
        "electronic_gap",
    "results.properties.thermodynamic.trajectory.available_properties":
        "nomad_trajectory",
}


def load_nomad_records(
    n_samples: Optional[int] = 200,
    elements: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает вычислительные записи из NOMAD Archive.

    Parameters
    ----------
    elements : фильтр по элементам, например ["Li", "Fe", "O"]
    """
    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    el_tag = "_".join(sorted(elements or [])) or "all"
    cache_file = cache / f"nomad_{el_tag}_{n_samples}.jsonl"

    if cache_file.exists():
        records: List[Dict[str, Any]] = []
        with open(cache_file, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
                if n_samples and len(records) >= n_samples:
                    break
        return records

    import requests
    session = _get_session()
    records = []
    page_size = min(n_samples or 100, 100)

    query: Dict[str, Any] = {
        "results.method.simulation.program_name": "VASP",
    }
    if elements:
        query["results.material.elements"] = {"all": elements}

    body = {
        "query": query,
        "required": {
            "results": {
                "material": {"chemical_formula_reduced": "*", "elements": "*"},
                "properties": {"electronic": "*"},
            }
        },
        "pagination": {"page_size": page_size},
    }

    url = f"{NOMAD_API_BASE}/entries/query"
    with open(cache_file, "w", encoding="utf-8") as fh_cache:
        after = None
        while True:
            if after:
                body["pagination"]["page_after_value"] = after
            try:
                resp = session.post(url, json=body, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                logger.warning(f"[nomad] request failed: {exc}")
                break

            entries = payload.get("data") or []
            pagination = payload.get("pagination") or {}

            for entry in entries:
                mat = (entry.get("results") or {}).get("material") or {}
                formula = mat.get("chemical_formula_reduced", "").strip()
                if not formula:
                    continue
                targets: Dict[str, float] = {}
                # Попытка извлечь band gap
                props = (entry.get("results") or {}).get("properties") or {}
                elec = props.get("electronic") or {}
                bs_list = elec.get("band_structure_electronic") or []
                for bs in (bs_list if isinstance(bs_list, list) else [bs_list]):
                    for bg in (bs.get("band_gap") or []):
                        val = bg.get("value")
                        if val is not None:
                            try:
                                targets["electronic_gap"] = float(val)
                                break
                            except (TypeError, ValueError):
                                pass
                    if "electronic_gap" in targets:
                        break
                if targets:
                    rec = {"smiles": formula, "formula": formula,
                           "entry_id": entry.get("entry_id", ""),
                           "targets": targets, "source": "nomad"}
                    records.append(rec)
                    fh_cache.write(json.dumps(rec) + "\n")
                if n_samples and len(records) >= n_samples:
                    break

            after = pagination.get("next_page_after_value")
            if not after or not entries:
                break
            if n_samples and len(records) >= n_samples:
                break
            print(f"\r[nomad] fetched {len(records)}", end="", flush=True)
            time.sleep(0.15)

    print()
    logger.info(f"[nomad] loaded {len(records)} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 8. ColabFit — DS_u8strp7hm0cy_0 (JARVIS_CFID_OQMD зеркало)
#    https://materials.colabfit.org/id/DS_u8strp7hm0cy_0
# ─────────────────────────────────────────────────────────────────────────────

COLABFIT_DATASET_ID = "DS_u8strp7hm0cy_0"

COLABFIT_TARGET_MAP: Dict[str, str] = {
    "formation_energy":   "formation_energy",
    "band_gap":           "electronic_gap",
    "ehull":              "thermodynamic_stability",
    "bulk_modulus":       "bulk_modulus",
    "shear_modulus":      "shear_modulus",
    "total_energy":       "total_energy",
}


def load_colabfit_records(
    n_samples: Optional[int] = 500,
    dataset_id: str = COLABFIT_DATASET_ID,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает датасет из ColabFit по ID.

    По умолчанию: DS_u8strp7hm0cy_0 = JARVIS_CFID_OQMD
    https://materials.colabfit.org/id/DS_u8strp7hm0cy_0

    ColabFit предоставляет REST API без ключей.
    """
    cache = pathlib.Path(cache_dir or os.path.join(tempfile.gettempdir(), "umcs_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"colabfit_{dataset_id}_{n_samples}.jsonl"

    if cache_file.exists():
        records: List[Dict[str, Any]] = []
        with open(cache_file, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
                if n_samples and len(records) >= n_samples:
                    break
        return records

    import requests
    session = _get_session()

    # Шаг 1: получить список configuration_set_ids для датасета
    ds_url = f"{COLABFIT_API_BASE}/datasets/{dataset_id}"
    try:
        resp = session.get(ds_url, timeout=60)
        resp.raise_for_status()
        ds_info = resp.json()
    except Exception as exc:
        logger.warning(f"[colabfit] dataset info failed: {exc}")
        # Fallback: JARVIS dft_3d
        logger.info("[colabfit] falling back to JARVIS dft_3d")
        return load_jarvis_records(n_samples=n_samples, dataset_name="dft_3d",
                                   cache_dir=cache_dir)

    # Шаг 2: получить configurations
    configs_url = f"{COLABFIT_API_BASE}/configurations"
    records = []
    page = 1
    page_size = min(n_samples or 100, 100)

    with open(cache_file, "w", encoding="utf-8") as fh_cache:
        while True:
            try:
                resp = session.get(
                    configs_url,
                    params={"dataset_id": dataset_id,
                            "page": page, "per_page": page_size},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning(f"[colabfit] configs page {page} failed: {exc}")
                break

            items = data if isinstance(data, list) else data.get("data", [])
            if not items:
                break

            for item in items:
                formula = (item.get("chemical_formula_reduced")
                           or item.get("formula") or "").strip()
                if not formula:
                    continue
                targets: Dict[str, float] = {}
                props = item.get("properties") or {}
                for key, head in COLABFIT_TARGET_MAP.items():
                    val = props.get(key)
                    if val is not None:
                        try:
                            targets[head] = float(val)
                        except (TypeError, ValueError):
                            pass
                if targets:
                    rec = {"smiles": formula, "formula": formula,
                           "targets": targets,
                           "config_id": item.get("id", ""),
                           "source": f"colabfit_{dataset_id}"}
                    records.append(rec)
                    fh_cache.write(json.dumps(rec) + "\n")
                if n_samples and len(records) >= n_samples:
                    break

            if n_samples and len(records) >= n_samples:
                break
            if len(items) < page_size:
                break
            page += 1
            time.sleep(0.1)
            print(f"\r[colabfit] fetched {len(records)}", end="", flush=True)

    print()
    if not records:
        # Конечный fallback: JARVIS dft_3d (те же данные, другой маршрут)
        logger.info("[colabfit] empty result, falling back to jarvis dft_3d")
        return load_jarvis_records(n_samples=n_samples, dataset_name="dft_3d",
                                   cache_dir=cache_dir)

    logger.info(f"[colabfit] loaded {len(records)} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 9. HDF5 — универсальный формат UMCS
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5_records(
    path: str,
    smiles_key: str = "smiles",
    targets_group: str = "targets",
    n_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает записи из HDF5-файла в формате UMCS.

    Ожидаемая структура:
      /smiles          — dataset строк (UTF-8)
      /targets/<name>  — dataset float64 (той же длины)

    Parameters
    ----------
    smiles_key     : имя датасета со SMILES (default "smiles")
    targets_group  : имя группы с таргетами (default "targets")
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        raise ImportError("pip install h5py")

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HDF5 file not found: {p}")

    records: List[Dict[str, Any]] = []
    with h5py.File(p, "r") as hf:
        smiles_ds = hf[smiles_key]
        n = len(smiles_ds)
        limit = min(n_samples or n, n)

        target_names = list(hf[targets_group].keys()) if targets_group in hf else []
        target_arrays = {name: hf[targets_group][name][:limit]
                         for name in target_names}

        for i in range(limit):
            raw = smiles_ds[i]
            smiles = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            if not smiles.strip():
                continue
            targets: Dict[str, float] = {}
            for name, arr in target_arrays.items():
                val = float(arr[i])
                if not (np.isnan(val) or np.isinf(val)):
                    targets[name] = val
            if targets:
                records.append({"smiles": smiles, "targets": targets,
                                 "source": f"hdf5:{p.name}"})

    logger.info(f"[hdf5] loaded {len(records)} records from {path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 10. Универсальные CSV и JSON загрузчики (расширенные vs умолчальные в interface)
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_records(
    path: str,
    smiles_col: str = "SMILES",
    target_map: Optional[Dict[str, str]] = None,
    delimiter: str = ",",
    n_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает произвольный CSV.

    Parameters
    ----------
    target_map : {"head_name": "column_name", ...}
                 Если None — все числовые колонки кроме smiles_col считаются таргетами.
    """
    import pandas as pd

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p, sep=delimiter, nrows=n_samples)
    records: List[Dict[str, Any]] = []

    auto_targets = target_map is None
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != smiles_col]

    for _, row in df.iterrows():
        smiles = str(row.get(smiles_col) or "").strip()
        if not smiles:
            continue
        targets: Dict[str, float] = {}
        if auto_targets:
            for col in numeric_cols:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    targets[col] = float(val)
        else:
            for head, col in (target_map or {}).items():
                val = row.get(col)
                if val is not None:
                    try:
                        targets[head] = float(val)
                    except (TypeError, ValueError):
                        pass
        if targets:
            records.append({"smiles": smiles, "targets": targets, "source": f"csv:{p.name}"})

    return records


def load_json_records(
    path: str,
    smiles_key: str = "smiles",
    targets_key: str = "targets",
    n_samples: Optional[int] = None,
    jsonlines: bool = False,
) -> List[Dict[str, Any]]:
    """
    Загружает JSON или JSON-Lines файл.

    Parameters
    ----------
    jsonlines : True если файл в формате JSON-Lines (одна запись на строку)
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")

    records: List[Dict[str, Any]] = []
    if jsonlines:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                smiles = str(item.get(smiles_key) or item.get("SMILES") or "").strip()
                targets = item.get(targets_key) or {}
                if smiles and isinstance(targets, dict) and targets:
                    records.append({"smiles": smiles,
                                     "targets": {k: float(v) for k, v in targets.items()
                                                 if v is not None},
                                     "source": f"jsonl:{p.name}"})
                if n_samples and len(records) >= n_samples:
                    break
    else:
        with open(p, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            payload = payload.get("records") or payload.get("data") or list(payload.values())
        for item in (payload or []):
            if not isinstance(item, dict):
                continue
            smiles = str(item.get(smiles_key) or item.get("SMILES") or "").strip()
            targets = item.get(targets_key) or {}
            if smiles and isinstance(targets, dict) and targets:
                records.append({"smiles": smiles,
                                 "targets": {k: float(v) for k, v in targets.items()
                                             if v is not None},
                                 "source": f"json:{p.name}"})
            if n_samples and len(records) >= n_samples:
                break

    logger.info(f"[json] loaded {len(records)} records from {path}")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# 11. RCSB PDB (Protein Data Bank / CSD)
#    Скачивание структурных PDB файлов и привязка к таргетам из CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_rcsb_records(
    n_samples: Optional[int] = 500,
    csv_path: Optional[str] = None,
    smiles_col: str = "PDB_ID",
    target_map: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает структуры из RCSB PDB. 
    Поскольку сам PDB не содержит QM-свойств, требуется CSV файл,
    где в колонке указаны PDB ID, а в других - эталонные значения для голов.
    """
    import pandas as pd
    
    if not csv_path or not pathlib.Path(csv_path).exists():
        raise FileNotFoundError("Для RCSB требуется CSV-файл с PDB ID и целевыми параметрами (--csv)")
        
    df = pd.read_csv(csv_path, nrows=n_samples)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != smiles_col]
    
    session = _get_session()
    records = []
    
    for _, row in df.iterrows():
        pdb_id = str(row.get(smiles_col) or "").strip()
        if not pdb_id or len(pdb_id) < 4:
            continue
            
        targets = {}
        if target_map is None:
            for col in numeric_cols:
                val = row.get(col)
                if pd.notna(val):
                    targets[col] = float(val)
        else:
            for head, col in target_map.items():
                val = row.get(col)
                if pd.notna(val):
                    targets[head] = float(val)
                    
        if not targets:
            continue # Обучаем только если есть эталонные параметры
            
        # Скачивание PDB
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            resp = session.get(pdb_url, timeout=15)
            if resp.status_code != 200:
                continue
            pdb_text = resp.text
        except Exception:
            continue
            
        xyz_text = _pdb_to_xyz(pdb_text)
        if xyz_text:
            records.append({
                "smiles": pdb_id,
                "xyz": xyz_text,
                "targets": targets,
                "source": f"rcsb:{pdb_id}"
            })
            
        if n_samples and len(records) >= n_samples:
            break
            
    logger.info(f"[rcsb] loaded {len(records)} records with exact 3D coordinates")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# 12. Facebook OMC25 (Open Materials Catalog)
#    Загрузка через HuggingFace Datasets: facebook/OMC25
# ─────────────────────────────────────────────────────────────────────────────
def load_omc25_records(
    n_samples: Optional[int] = 1000,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Загружает OMC25 в стриминговом режиме. Добавлено логирование структуры данных,
    чтобы понять, какие ключи реально возвращает датасет, а также улучшенный парсинг.
    """
    datasets = _try_import("datasets", "datasets")
    import ast
    
    ALL_SYMBOLS = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", 
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", 
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", 
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", 
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", 
        "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    
    logger.info(f"[omc25] loading OMC25 via HuggingFace streaming...")
    ds = datasets.load_dataset("facebook/OMC25", split="train", streaming=True)
    
    records = []
    first_item_logged = False
    
    for item in ds:
        if not first_item_logged:
            logger.info(f"OMC25 FIRST ITEM KEYS: {list(item.keys())}")
            # Печатаем первые 100 символов каждого значения для диагностики
            preview = {k: str(v)[:100] for k, v in item.items()}
            logger.info(f"OMC25 FIRST ITEM PREVIEW: {preview}")
            first_item_logged = True
            
        targets = {}
        # Пробуем разные варианты названий для энергии
        for e_key in ["energy", "total_energy", "formation_energy", "energy_per_atom", "e_above_hull"]:
            if e_key in item and item[e_key] is not None and str(item[e_key]).strip():
                try:
                    targets["total_energy"] = float(item[e_key])
                    break
                except ValueError:
                    pass
                    
        # Пробуем разные варианты для band gap
        for bg_key in ["band_gap", "bandgap", "electronic_gap"]:
            if bg_key in item and item[bg_key] is not None and str(item[bg_key]).strip():
                try:
                    targets["electronic_gap"] = float(item[bg_key])
                    break
                except ValueError:
                    pass
            
        if not targets:
            # Если нет таргетов, логируем это один раз
            if len(records) == 0:
                logger.warning(f"Skipping record because no targets found. Available keys: {list(item.keys())}")
            continue
            
        # В датасетах HuggingFace часто данные из CSV лежат в виде строк, парсим их
        atomic_numbers = item.get("atomic_numbers")
        positions = item.get("positions")
        
        # Если есть структура CIF (иногда бывает в таких датасетах)
        cif_data = item.get("cif") or item.get("structure")
        
        if (atomic_numbers is None or positions is None) and cif_data is None:
            continue
            
        if atomic_numbers is not None and positions is not None:
            if isinstance(atomic_numbers, str):
                try:
                    atomic_numbers = ast.literal_eval(atomic_numbers)
                except:
                    continue
            if isinstance(positions, str):
                try:
                    positions = ast.literal_eval(positions)
                except:
                    continue
                    
            symbols = []
            for z in atomic_numbers:
                if isinstance(z, int) and 1 <= z <= len(ALL_SYMBOLS):
                    symbols.append(ALL_SYMBOLS[z-1])
                else:
                    symbols.append("X")
                    
            xyz_lines = [str(len(symbols)), "OMC25 Structure"]
            for sym, pos in zip(symbols, positions):
                xyz_lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
                
            xyz_text = "\n".join(xyz_lines)
            formula = "".join(list(set(symbols)))
            
        elif cif_data is not None:
            # Если есть структура в текстовом виде (CIF)
            try:
                from ase.io import read
                import io
                atoms = read(io.StringIO(cif_data), format='cif')
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                
                xyz_lines = [str(len(symbols)), "OMC25 Structure"]
                for sym, pos in zip(symbols, positions):
                    xyz_lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
                xyz_text = "\n".join(xyz_lines)
                formula = atoms.get_chemical_formula()
            except ImportError:
                if len(records) == 0:
                    logger.warning("ASE library is required to parse CIF structures. Run: pip install ase")
                continue
            except Exception as e:
                continue
                
        records.append({
            "smiles": formula,
            "xyz": xyz_text,
            "targets": targets,
            "source": "omc25"
        })
        
        if n_samples and len(records) >= n_samples:
            break
            
    logger.info(f"[omc25] loaded {len(records)} records with exact 3D coordinates")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Реестр датасетов
# ─────────────────────────────────────────────────────────────────────────────

#: Маппинг имени датасета → загрузчик.
#: Каждый загрузчик принимает (n_samples, **kwargs) и возвращает List[Dict].
# ─────────────────────────────────────────────────────────────────────────────
# EDBench (github.com/HongxinXiang/EDBench)
#   Molecules drawn from PCQM4Mv2; each ships an already-computed electron
#   density cube (Psi4, B3LYP/6-31G**, 6-31+G** where sulfur is present;
#   0.4 Bohr grid spacing) alongside its CSV of targets. This is a
#   DIFFERENT level of theory than this pipeline's own PySCF path (default
#   PBE/def2-SVP, but see `AnalysisConfig.level_of_theory` -- now a real,
#   configurable field, not a literal), so records from here must never
#   share an analysis_signature with QM9-through-this-pipeline candidates
#   (doc §15/§20.5's comparability discipline) -- see
#   EDBENCH_LEVEL_OF_THEORY below, and run with `level_of_theory=
#   EDBENCH_LEVEL_OF_THEORY` and a dedicated `field_cache_namespace` when
#   using this loader's `precomputed_cube_path`.
#
#   Six benchmarks, not five (paper Table 2, arXiv:2505.09262 -- confirmed
#   there, not just the repo README): ed_energy_5w (ED5-EC, 47,986 mol),
#   ed_homo_lumo_5w (ED5-OE, 43,510), ed_multipole_moments_5w (ED5-MM,
#   49,917), ed_open_shell_5w (ED5-OCS, 50,000), ed_retrieval_5w (ED5-MER,
#   50,000), ed_prediction_5w (ED5-EDP, 50,000) -- "_5w" names a target
#   ceiling of 50,000, not every benchmark's actual count. For a first ESM
#   smoke test, ed_energy_5w is the safest starting point: its target
#   order is transcribed directly from the paper's own table, the other
#   two prediction benchmarks below are now ALSO from the paper (not a
#   guess), and ED5-OCS/MER/EDP's labels are not scalar chemistry targets
#   in the same sense (open/closed-shell class, retrieval pairing,
#   structure-to-density supervision) so they matter less for what this
#   pipeline actually uses `targets` for (§16.5 anchors).
#
#   File layout is still only README-sourced (github.com/HongxinXiang/
#   EDBench), NOT verified against an actual downloaded archive -- the
#   Harvard Dataverse deposit itself is outside this environment's network
#   access, both for automated download and for confirming file layout
#   directly. Spot-check the first extracted subset's folder structure and
#   CSV column names before a full run:
#     {edbench_root}/{benchmark}/raw/{benchmark}.csv
#     {edbench_root}/{benchmark}/raw/psi4_grid0.4_cubes/{mol_index}/Mol1_Dt.cube
#     {edbench_root}/{benchmark}/raw/psi4_grid0.4_cubes/{mol_index}/Mol1_ESP.cube
#   CSV columns (per README): index, smiles, canonical_smiles,
#   scaffold_split, random_split, label -- `label` is one or more
#   whitespace-separated floats, in the benchmark's own documented order.
#   Each `psi4_grid0.4_cubes/{mol_index}/` is its own molecule's cube
#   directory even though the file inside is always literally named
#   "Mol1_Dt.cube" (a generic per-directory name, not molecule #1).
# ─────────────────────────────────────────────────────────────────────────────

EDBENCH_LEVEL_OF_THEORY = "DFT/b3lyp/6-31G**(+)/psi4"

# Target order/names for ED5-EC, ED5-OE and ED5-MM are transcribed directly
# from the paper's Table 2 (arXiv:2505.09262), not guessed. ED5-OCS/MER/EDP
# are intentionally absent: their labels are not per-molecule scalar
# targets in this same sense (binary class, retrieval pairing, or the
# density itself as supervision), so `load_edbench_records` falls back to
# generic `target_i` naming for any benchmark not listed here.
EDBENCH_TARGET_NAMES: Dict[str, List[str]] = {
    "ed_energy_5w": [
        "dft_final_energy", "nuclear_repulsion_energy", "one_electron_energy",
        "two_electron_energy", "dft_exchange_correlation_energy", "total_energy",
    ],
    "ed_homo_lumo_5w": [
        "homo_minus_2", "homo_minus_1", "homo_0",
        "lumo_plus_0", "lumo_plus_1", "lumo_plus_2", "lumo_plus_3",
    ],
    "ed_multipole_moments_5w": ["dipole_x", "dipole_y", "dipole_z", "dipole_magnitude"],
}


def load_edbench_records(
    n_samples: Optional[int] = None,
    edbench_root: Optional[str] = None,
    benchmark: str = "ed_energy_5w",
    cache_dir: Optional[str] = None,
    adopt_precomputed_cube: bool = True,
) -> List[Dict[str, Any]]:
    """
    Загружает один EDBench-бенчмарк (по умолчанию ed_energy_5w, ~50k молекул)
    из уже скачанного и распакованного `edbench_root`.

    Parameters
    ----------
    n_samples   : максимальное число записей (None = все)
    edbench_root: путь к распакованному EDBench (см. layout выше). Автозагрузка
                  не выполняется -- Dataverse-депозиты не универсально доступны
                  по прямой ссылке; скачать и распаковать нужно вручную.
    benchmark   : один из ed_energy_5w / ed_homo_lumo_5w / ed_multipole_moments_5w /
                  ed_open_shell_5w / ed_retrieval_5w / ed_prediction_5w
    adopt_precomputed_cube: если True (по умолчанию), каждая запись несёт
                  `precomputed_cube_path` + `precomputed_field_kind="density"` +
                  `level_of_theory=EDBENCH_LEVEL_OF_THEORY` -- сигнал даунстриму
                  адаптировать готовый куб вместо повторного DFT (см.
                  esm_master_analytics.py). Отдельно от cube кладётся ESP-куб
                  как `precomputed_esp_cube_path`, если он есть на диске.
    """
    import pandas as pd

    if not edbench_root:
        raise ValueError(
            "edbench_root is required -- download and extract a benchmark subset "
            "from the Harvard Dataverse deposit linked at "
            "https://github.com/HongxinXiang/EDBench first; this loader does not "
            "fetch it automatically"
        )
    root = pathlib.Path(edbench_root) / benchmark / "raw"
    csv_path = root / f"{benchmark}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"expected {csv_path} -- check `edbench_root`/`benchmark`, and that "
            f"the archive was extracted with its original folder layout intact"
        )
    df = pd.read_csv(csv_path, nrows=n_samples)
    missing = {"index", "smiles", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected column(s) {sorted(missing)}; found "
            f"{list(df.columns)} -- EDBench's CSV layout may differ from the "
            f"version this loader was written against, check raw/readme.md"
        )
    target_names = EDBENCH_TARGET_NAMES.get(benchmark)

    cube_dir = root / "psi4_grid0.4_cubes"
    records: List[Dict[str, Any]] = []
    n_missing_cube = 0
    for _, row in df.iterrows():
        smiles = str(row["smiles"]).strip()
        if not smiles:
            continue
        values = [float(v) for v in str(row["label"]).split()]
        targets = (
            dict(zip(target_names, values)) if target_names and len(target_names) == len(values)
            else {f"target_{i}": v for i, v in enumerate(values)}
        )
        record: Dict[str, Any] = {
            "smiles": smiles,
            "targets": targets,
            "dataset_record_index": int(row["index"]),
        }
        if adopt_precomputed_cube:
            mol_dir = cube_dir / str(row["index"])
            density_cube = mol_dir / "Mol1_Dt.cube"
            esp_cube = mol_dir / "Mol1_ESP.cube"
            if density_cube.exists():
                record["precomputed_cube_path"] = str(density_cube)
                record["precomputed_field_kind"] = "density"
                record["level_of_theory"] = EDBENCH_LEVEL_OF_THEORY
                if esp_cube.exists():
                    record["precomputed_esp_cube_path"] = str(esp_cube)
            else:
                n_missing_cube += 1
        records.append(record)

    if adopt_precomputed_cube and n_missing_cube:
        logger.info(
            f"[edbench] {n_missing_cube}/{len(records)} records had no cube at the "
            f"expected path -- those will fall through to ordinary DFT generation"
        )
    logger.info(f"[edbench] {benchmark}: loaded {len(records)} records")
    return records


def load_local_cube_records(
    path: str,
    level_of_theory: str,
    n_samples: Optional[int] = None,
    field_kind: str = "density",
    recursive: bool = False,
    smiles_from_sdf: bool = True,
) -> List[Dict[str, Any]]:
    """
    A folder of `.cube` files with no accompanying dataset/CSV at all --
    "just here are some cubes" rather than "here is a named benchmark".
    Every record carries `precomputed_cube_path` (see
    `_extract_record_worker`'s adoption path) and no SMILES unless an
    identically-named `.sdf` sits next to the cube AND `smiles_from_sdf` is
    left on (best-effort only -- a missing/unparseable one just leaves
    `smiles` empty, which is a clean, expected outcome here: if the cube
    cannot be adopted for some reason, that record fails as "invalid or
    unembeddable structure" rather than the worker inventing a geometry).

    `level_of_theory` is REQUIRED and not guessed: nothing about a `.cube`
    file reliably states what produced it, and adoption's whole point is
    refusing to mix an unlabelled level of theory into a run's
    analysis_signature (doc §15/§20.5) -- see `AnalysisConfig.level_of_theory`.

    Ordering is alphabetical by path and stable across calls: `mol_id` is
    derived from a record's POSITION in what a loader returns
    (`esm_master_analytics.py`: `mol_id = f"{ds_name}_{i:06d}"`), so
    resumability depends on this loader returning the same file at the
    same position every time the folder's contents haven't changed. Read-
    only throughout -- nothing here ever writes to or deletes from `path`.
    """
    root = pathlib.Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Local cube path not found: {root}")
    pattern = "**/*.cube" if recursive else "*.cube"
    cube_paths = sorted(root.glob(pattern))
    if n_samples is not None:
        cube_paths = cube_paths[:n_samples]

    records: List[Dict[str, Any]] = []
    for index, cube_path in enumerate(cube_paths):
        smiles = ""
        if smiles_from_sdf:
            sdf_path = cube_path.with_suffix(".sdf")
            if sdf_path.exists():
                try:
                    from rdkit import Chem  # type: ignore
                    mol = next(Chem.SDMolSupplier(str(sdf_path)), None)
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                except Exception:
                    pass  # best-effort only, per docstring
        records.append({
            "smiles": smiles,
            "targets": {},
            "dataset_record_index": index,
            "precomputed_cube_path": str(cube_path),
            "precomputed_field_kind": field_kind,
            "level_of_theory": level_of_theory,
        })
    logger.info(f"[local_cubes] {root}: found {len(records)} cube file(s)")
    return records


DATASET_REGISTRY: Dict[str, Callable] = {
    "qm9":      load_qm9_records,
    "oqmd":     load_oqmd_records,
    "oqmd_dump": load_oqmd_dump_records,
    "mp":       load_mp_records,
    "jarvis":   load_jarvis_records,
    "aflow":    load_aflow_records,
    "cod":      load_cod_records,
    "nomad":    load_nomad_records,
    "colabfit": load_colabfit_records,
    "hdf5":     load_hdf5_records,
    "csv":      load_csv_records,
    "json":     load_json_records,
    "rcsb":     load_rcsb_records,
    "omc25":    load_omc25_records,
    "edbench":  load_edbench_records,
    "local_cubes": load_local_cube_records,
}

DATASET_DESCRIPTIONS: Dict[str, str] = {
    "qm9":       "QM9: ~134k молекул, квантовые свойства (gap, dipole, …)",
    "oqmd":      "OQMD v1.8: >1M кристаллических структур, REST API",
    "oqmd_dump": "OQMD v1.8: локальный CSV-дамп (oqmd.org/download/)",
    "mp":        "Materials Project: ~160k структур (требует API-ключ)",
    "jarvis":    "JARVIS-DFT: 76k 3D-структур, оптика, магнетизм, упругость",
    "aflow":     "AFLOW: >3M записей, REST API без ключей",
    "cod":       "COD: Crystallography Open Database, REST API",
    "nomad":     "NOMAD Archive: вычислительные записи, REST API",
    "colabfit":  "ColabFit DS_u8strp7hm0cy_0 (JARVIS_CFID_OQMD зеркало)",
    "hdf5":      "HDF5: универсальный формат UMCS (локальный файл)",
    "csv":       "CSV: произвольный CSV-файл",
    "json":      "JSON/JSONL: произвольный JSON или JSON-Lines файл",
    "rcsb":      "RCSB PDB: PDB файлы по ID из CSV + XYZ координаты",
    "omc25":     "Facebook OMC25: HuggingFace датасет с точными XYZ (streaming)",
    "edbench":   "EDBench: ~50k молекул на подвыборку, готовые cube-файлы плотности (Psi4 B3LYP/6-31G**) — другой level_of_theory, чем локальный QM9/PySCF конвейер",
    "local_cubes": "Папка с .cube-файлами без датасета-обёртки — level_of_theory задаётся вручную, обязательно",
}

# Which dataset loaders take a local filesystem path (and any other
# critical-but-easy-to-mistype parameter) as a required or near-required
# argument. Drives a dedicated, labelled input per parameter in the
# control centre's dataset picker instead of requiring the raw JSON
# kwargs box for these -- read-only paths into the pipeline, never a
# location anything here writes to or deletes (see `ArtifactPaths.reset_run`,
# which only ever touches `run_dir`, never a path a loader was given).
# {dataset_name: {kwarg_name: (ui_label, default_value)}}
DATASET_LOCAL_INPUTS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "edbench": {
        "edbench_root": ("Local EDBench root (extracted benchmark folder)", ""),
        "benchmark": ("Benchmark subset", "ed_energy_5w"),
    },
    "local_cubes": {
        "path": ("Local .cube folder", ""),
        "level_of_theory": ("Level of theory that produced these cubes (required)", ""),
    },
    "hdf5": {"path": ("Local HDF5 file", "")},
    "csv": {"path": ("Local CSV file", "")},
    "json": {"path": ("Local JSON/JSON-Lines file", "")},
}


def load_dataset_records(
    name: str,
    n_samples: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Унифицированная точка входа: загружает датасет по имени из реестра.

    Parameters
    ----------
    name      : имя из DATASET_REGISTRY
    n_samples : максимальное число записей
    **kwargs  : передаются напрямую в загрузчик (api_key, cache_dir, …)

    Returns
    -------
    List[Dict[str, Any]]  — записи в формате UMCS normalize_training_records
    """
    if not kwargs.get("cache_dir"):
        # Dataset downloads are input caches, not source-tree artefacts.  A
        # caller may still set cache_dir per loader, while the common default
        # follows the same reusable/shared-data policy as ESM cube fields.
        default_cache = os.environ.get(
            "ESM_DATASET_CACHE_DIR",
            os.path.join(os.getcwd(), "esm_analysis", "shared_datasets"),
        )
        os.makedirs(default_cache, exist_ok=True)
        kwargs["cache_dir"] = default_cache

    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    loader = DATASET_REGISTRY[name]
    records = loader(n_samples=n_samples, **kwargs)
    logger.info(f"[datasets] {name}: {len(records)} records loaded")
    return records


def list_datasets() -> None:
    """Выводит таблицу доступных датасетов."""
    print("\nДоступные датасеты UMCS:")
    print(f"  {'Имя':<14} {'Описание'}")
    print(f"  {'-'*13} {'-'*55}")
    for name in sorted(DATASET_REGISTRY):
        print(f"  {name:<14} {DATASET_DESCRIPTIONS.get(name, '')}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI (автономный режим — для тестирования загрузчиков)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Тест загрузчиков датасетов UMCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python umcs_datasets.py --list\n"
            "  python umcs_datasets.py --dataset qm9    --n 10\n"
            "  python umcs_datasets.py --dataset oqmd   --n 5\n"
            "  python umcs_datasets.py --dataset jarvis --n 10 --jarvis-name dft_3d\n"
            "  python umcs_datasets.py --dataset mp     --n 5  --api-key KEY\n"
            "  python umcs_datasets.py --dataset colabfit --n 5\n"
            "  python umcs_datasets.py --dataset csv --csv data.csv --n 20\n"
        ),
    )
    parser.add_argument("--list", action="store_true", help="Вывести список датасетов")
    parser.add_argument("--dataset", default="qm9",
                        choices=list(DATASET_REGISTRY),
                        help="Источник данных")
    parser.add_argument("--n", type=int, default=10, help="Число записей")
    parser.add_argument("--api-key", default=None, help="API-ключ (для MP)")
    parser.add_argument("--csv", default=None, help="Путь к CSV/JSON/HDF5 файлу")
    parser.add_argument("--cache-dir", default=None, help="Папка кэша")
    parser.add_argument("--jarvis-name", default="dft_3d",
                        help="Имя датасета JARVIS (dft_3d, dft_2d, cfid_3d)")
    parser.add_argument("--output-json", default=None,
                        help="Сохранить записи в JSON-файл")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    else:
        kwargs: Dict[str, Any] = {"cache_dir": args.cache_dir}
        if args.api_key:
            kwargs["api_key"] = args.api_key
        if args.csv:
            kwargs["path" if args.dataset in ("hdf5", "csv", "json") else "csv_path"] = args.csv
        if args.dataset == "jarvis":
            kwargs["dataset_name"] = args.jarvis_name

        records = load_dataset_records(args.dataset, n_samples=args.n, **kwargs)
        print(f"\nЗагружено {len(records)} записей из '{args.dataset}':")
        for i, rec in enumerate(records[:5], 1):
            smiles = rec.get("smiles", "")[:40]
            targets_preview = {k: round(v, 4) for k, v in list(rec.get("targets", {}).items())[:4]}
            src = rec.get("source", "")
            print(f"  [{i}] smiles={smiles!r:<42} targets={targets_preview}  src={src}")
        if len(records) > 5:
            print(f"  … ещё {len(records) - 5} записей")

        if args.output_json:
            out = pathlib.Path(args.output_json)
            out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[output] saved {len(records)} records → {out}")

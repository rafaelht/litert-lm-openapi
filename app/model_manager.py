from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredModel:
    id: str
    path: Path
    created: int
    size_bytes: int


def _get_models_dir(custom_dir: str | Path | None = None) -> Path:
    if custom_dir:
        return Path(custom_dir).resolve()
    settings = get_settings()
    configured = Path(settings.models_dir)
    if configured.exists():
        return configured.resolve()

    # Si /models no existe (ej. en desarrollo local en Mac o durante tests),
    # comprobar si existe 'models' en el directorio de trabajo o en la raíz del proyecto.
    for candidate in [Path.cwd() / "models", Path(__file__).resolve().parents[1] / "models"]:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    return configured.resolve()


def discover_models(models_dir: str | Path | None = None) -> dict[str, DiscoveredModel]:
    """
    Escanea dinámicamente el directorio de modelos montado y detecta:
    1. Subcarpetas que contienen 'model.litertlm' (ej. /models/<name>/model.litertlm) -> ID = <name>
    2. Archivos directos con extensión '.litertlm' (ej. /models/<name>.litertlm) -> ID = <name>.litertlm

    Excluye explícitamente archivos de cache (*.xnnpack_cache), archivos temporales y carpetas ocultas.
    """
    dir_path = _get_models_dir(models_dir)
    discovered: dict[str, DiscoveredModel] = {}

    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning("Models directory does not exist or is not a directory: %s", dir_path)
        return discovered

    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: p.name)
    except Exception:
        logger.exception("Failed to list models directory: %s", dir_path)
        return discovered

    for entry in entries:
        if entry.name.startswith("."):
            continue

        # 1. Caso subdirectorio con model.litertlm dentro
        if entry.is_dir():
            model_file = entry / "model.litertlm"
            if model_file.is_file():
                try:
                    stat = model_file.stat()
                    model_id = entry.name
                    discovered[model_id] = DiscoveredModel(
                        id=model_id,
                        path=model_file.resolve(),
                        created=int(stat.st_mtime),
                        size_bytes=stat.st_size,
                    )
                except Exception:
                    logger.exception("Error reading model info for directory: %s", entry)
            continue

        # 2. Caso archivo directo .litertlm
        if entry.is_file() and entry.name.endswith(".litertlm"):
            try:
                stat = entry.stat()
                model_id = entry.name
                discovered[model_id] = DiscoveredModel(
                    id=model_id,
                    path=entry.resolve(),
                    created=int(stat.st_mtime),
                    size_bytes=stat.st_size,
                )
            except Exception:
                logger.exception("Error reading model info for file: %s", entry)

    # 3. Retrocompatibilidad: si MODEL_PATH apunta a un archivo válido y no fue detectado aún
    settings = get_settings()
    if settings.model_path:
        fallback_path = Path(settings.model_path).resolve()
        if fallback_path.is_file() and fallback_path.name.endswith(".litertlm"):
            fallback_id = (
                fallback_path.parent.name
                if fallback_path.name == "model.litertlm"
                else fallback_path.name
            )
            if fallback_id not in discovered:
                stat = fallback_path.stat()
                discovered[fallback_id] = DiscoveredModel(
                    id=fallback_id,
                    path=fallback_path,
                    created=int(stat.st_mtime),
                    size_bytes=stat.st_size,
                )

    return discovered


def resolve_model(
    model_id: str | None,
    models_dir: str | Path | None = None,
) -> DiscoveredModel | None:
    """
    Resuelve de forma robusta un identificador de modelo solicitado hacia su modelo descubierto.
    Soporta coincidencia exacta o coincidencia agregando/removiendo la extensión '.litertlm'.
    Si el model_id es None, vacío, 'default' o 'litert-model', recurre al modelo activo en RAM
    o al primer modelo disponible en models_dir.
    """
    models = discover_models(models_dir)
    if not models:
        return None

    cleaned_id = (model_id or "").strip()

    # Si viene vacío, "default" o "litert-model", resolver al modelo activo en RAM o al primero disponible
    if not cleaned_id or cleaned_id.lower() in {"default", "litert-model"}:
        try:
            from app.engine import get_current_model_id
            active_id = get_current_model_id()
            if active_id and active_id in models:
                return models[active_id]
        except Exception:
            pass

        first_key = next(iter(models.keys()))
        return models[first_key]

    # 1. Búsqueda exacta
    if cleaned_id in models:
        return models[cleaned_id]

    # 2. Si no tiene extensión .litertlm, buscar con extensión
    with_ext = f"{cleaned_id}.litertlm"
    if with_ext in models:
        return models[with_ext]

    # 3. Si termina en .litertlm, buscar sin extensión
    if cleaned_id.endswith(".litertlm"):
        without_ext = cleaned_id[:-9]
        if without_ext in models:
            return models[without_ext]

    # 4. Comprobación directa en sistema de archivos en caso de montaje dinámico reciente
    dir_path = _get_models_dir(models_dir)
    candidates = [
        dir_path / cleaned_id / "model.litertlm",
        dir_path / f"{cleaned_id}.litertlm" / "model.litertlm",
        dir_path / cleaned_id,
        dir_path / f"{cleaned_id}.litertlm",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.name.endswith(".litertlm"):
            stat = candidate.stat()
            resolved_id = (
                candidate.parent.name
                if candidate.name == "model.litertlm"
                else candidate.name
            )
            return DiscoveredModel(
                id=resolved_id,
                path=candidate.resolve(),
                created=int(stat.st_mtime),
                size_bytes=stat.st_size,
            )

    return None


def resolve_model_profile_path(model_id: str) -> Path:
    """
    Resuelve la ruta del perfil YAML para un modelo específico.
    1. profiles/<model_id>.yaml o profiles/<model_id>.yml
    2. Si model_id termina en '.litertlm', profiles/<model_id_without_ext>.yaml
    3. Fallback a settings.model_profile o profiles/default.yaml
    """
    base_dirs = [
        Path.cwd() / "profiles",
        Path(__file__).resolve().parents[1] / "profiles",
    ]

    candidate_names: list[str] = [
        f"{model_id}.yaml",
        f"{model_id}.yml",
    ]
    if model_id.endswith(".litertlm"):
        stem = model_id[:-9]
        candidate_names.extend([f"{stem}.yaml", f"{stem}.yml"])

    for base in base_dirs:
        if not base.exists():
            continue
        for name in candidate_names:
            candidate = (base / name).resolve()
            if candidate.is_file():
                return candidate

    # Fallback configurado o por defecto
    settings = get_settings()
    default_profile = Path(settings.model_profile)
    if default_profile.is_absolute() and default_profile.is_file():
        return default_profile

    for base in [Path.cwd(), Path(__file__).resolve().parents[1]]:
        candidate = (base / default_profile).resolve()
        if candidate.is_file():
            return candidate

    return (Path.cwd() / "profiles" / "default.yaml").resolve()

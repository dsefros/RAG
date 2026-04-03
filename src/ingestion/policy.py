from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from src.config.settings import Settings


class PolicyResolutionError(Exception):
    pass


def _domain_from_path(document_root: Path, path: Path) -> str:
    rel = path.relative_to(document_root)
    return rel.parts[0]


def resolve_document_policy(settings: Settings, file_path: Path) -> dict[str, Any]:
    root = Path(settings.document_root)
    domain = _domain_from_path(root, file_path)
    default_groups = settings.access_policy.folder_defaults.get(domain)
    if not default_groups:
        raise PolicyResolutionError(f"No folder default policy for domain '{domain}'")

    policy: dict[str, Any] = {
        "allowed_groups": default_groups,
        "status": "active",
        "domain": domain,
    }

    sidecar = file_path.with_suffix(".meta.yaml")
    if sidecar.exists():
        loaded = yaml.safe_load(sidecar.read_text()) or {}
        if "allowed_groups" in loaded:
            if not isinstance(loaded["allowed_groups"], list) or not loaded["allowed_groups"]:
                raise PolicyResolutionError(f"Invalid allowed_groups in {sidecar}")
            policy["allowed_groups"] = loaded["allowed_groups"]
        if "status" in loaded:
            status = loaded["status"]
            if status not in settings.access_policy.allowed_statuses:
                raise PolicyResolutionError(f"Invalid status '{status}' in {sidecar}")
            policy["status"] = status

    return policy


def build_doc_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()[:24]

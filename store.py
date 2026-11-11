"""Circuit storage for NeuronSteer.

Circuits are small JSON documents (model id + a sparse neuron list), kept
in a per-user directory so they persist across Hermes sessions and can be
steered by name. Location: $NEURON_STEER_HOME, else
get_hermes_home()/neuron-steer/circuits (falling back to
$HERMES_HOME or ~/.hermes when Hermes itself is not importable).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from .engine import Circuit
except ImportError:  # loaded as a flat module directory
    from engine import Circuit

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _default_base() -> Path:
    """Per-user circuit home: Hermes home, never the plugin install dir
    (install-dir storage is lost on plugin update and shared across
    profiles)."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "neuron-steer" / "circuits"
    except Exception:
        home = os.environ.get("HERMES_HOME")
        base = Path(home) if home else Path.home() / ".hermes"
        return base / "neuron-steer" / "circuits"


def circuits_dir() -> Path:
    home = os.environ.get("NEURON_STEER_HOME")
    base = Path(home) if home else _default_base()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path_for(name: str) -> Path:
    if not _NAME_RE.match(name or ""):
        raise ValueError(
            f"Invalid circuit name {name!r}: use letters, digits, '-' and '_'."
        )
    return circuits_dir() / f"{name}.json"


def save_circuit(name: str, circuit: Circuit, meta: dict | None = None) -> Path:
    path = _path_for(name)
    doc = {
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "circuit": circuit.to_dict(),
        "meta": meta or {},
    }
    path.write_text(json.dumps(doc, indent=2))
    return path


def load_circuit(name: str) -> Circuit:
    path = _path_for(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved circuit named {name!r}. Use neuron_steer_list to see "
            "what is saved, or neuron_steer_find to create one."
        )
    doc = json.loads(path.read_text())
    return Circuit.from_dict(doc["circuit"])


def load_circuit_doc(name: str) -> dict:
    path = _path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"No saved circuit named {name!r}.")
    return json.loads(path.read_text())


def list_circuits() -> list[dict]:
    out = []
    for path in sorted(circuits_dir().glob("*.json")):
        try:
            doc = json.loads(path.read_text())
            circuit = doc.get("circuit", {})
            out.append({
                "name": doc.get("name", path.stem),
                "model_id": circuit.get("model_id", ""),
                "neurons": len(circuit.get("neurons", [])),
                "saved_at": doc.get("saved_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return out


def delete_circuit(name: str) -> bool:
    path = _path_for(name)
    if path.exists():
        path.unlink()
        return True
    return False

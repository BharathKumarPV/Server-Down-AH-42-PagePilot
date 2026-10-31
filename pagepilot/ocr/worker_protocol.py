"""
PagePilot – Stage 6B OCR Worker Communication Protocol.

Defines the JSON-over-stdio protocol used between the main Python 3.14
application and the isolated Python 3.12 worker process.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

PROTOCOL_VERSION = "1.0.0"

CMD_PING              = "ping"
CMD_DETECT_LAYOUT     = "detect_layout"
CMD_RECOGNIZE_TEXT    = "recognize_text"
CMD_RECOGNIZE_FORMULA = "recognize_formula"
CMD_PROCESS_PAGE      = "process_page"
CMD_SHUTDOWN          = "shutdown"

STATUS_OK                   = "ok"
STATUS_ERROR                = "error"
STATUS_MODELS_NOT_INSTALLED = "models_not_installed"
STATUS_DEPENDENCY_MISSING   = "dependency_missing"

# Explicit inference mode classifications
MODE_REAL_INFERENCE     = "REAL_INFERENCE"
MODE_SIMULATED_RESULT   = "SIMULATED_RESULT"
MODE_ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"


def serialize_request(
    command: str,
    args: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> str:
    """Format a request dictionary into an NDJSON single-line string."""
    req_id = request_id or str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "protocol_version": PROTOCOL_VERSION,
        "command": command,
        "args": args or {},
    }
    return json.dumps(payload, ensure_ascii=False) + "\n"


def deserialize_request(line: str) -> Dict[str, Any]:
    """Parse a single-line JSON string into a request dictionary."""
    clean = line.strip()
    if not clean:
        raise ValueError("Empty request line")
    data = json.loads(clean)
    if "command" not in data:
        raise ValueError("Malformed request: missing 'command'")
    return data


def serialize_response(
    status: str,
    payload: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    request_id: Optional[str] = None,
    execution_ms: float = 0.0,
) -> str:
    """Format a response dictionary into an NDJSON single-line string."""
    resp = {
        "jsonrpc": "2.0",
        "id": request_id or "",
        "status": status,
        "execution_ms": round(execution_ms, 2),
        "payload": payload or {},
        "error": error or "",
    }
    return json.dumps(resp, ensure_ascii=False) + "\n"


def deserialize_response(line: str) -> Dict[str, Any]:
    """Parse a single-line JSON string into a response dictionary."""
    clean = line.strip()
    if not clean:
        raise ValueError("Empty response line")
    data = json.loads(clean)
    if "status" not in data:
        raise ValueError("Malformed response: missing 'status'")
    return data


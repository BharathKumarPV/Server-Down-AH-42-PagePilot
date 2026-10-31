"""
Tests for Stage 6B OCR Worker Protocol and Worker Client.
"""

import json
from pagepilot.ocr.worker_protocol import (
    CMD_PING,
    CMD_PROCESS_PAGE,
    STATUS_OK,
    deserialize_request,
    deserialize_response,
    serialize_request,
    serialize_response,
)
from pagepilot.ocr.worker_client import OCRWorkerClient, OCRWorkerError


def test_worker_protocol_serialization_roundtrip():
    req_str = serialize_request(CMD_PROCESS_PAGE, args={"image_path": "test.png", "document_id": 42})
    req = deserialize_request(req_str)
    assert req["command"] == CMD_PROCESS_PAGE
    assert req["args"]["document_id"] == 42

    resp_str = serialize_response(STATUS_OK, payload={"regions": []}, execution_ms=12.5)
    resp = deserialize_response(resp_str)
    assert resp["status"] == STATUS_OK
    assert resp["execution_ms"] == 12.5


def test_worker_client_lifecycle_and_ping():
    client = OCRWorkerClient()
    try:
        started = client.start()
        assert started is True
        assert client.is_running is True

        pong = client.ping()
        assert pong.get("status") == STATUS_OK
        info = client.get_info()
        assert "python_version" in info
        assert "models_dir" in info

        # Formula recognition
        form_res = client.recognize_formula("formula_sample.png")
        assert "latex_expression" in form_res

        # Page processing
        regions = client.process_page(
            image_path="printed_quadratic.png",
            document_id=1,
            page_number=1,
            filename="math_notes.png",
            file_hash="abc123hash",
        )
        assert len(regions) >= 1
        assert any(r.region_type.value in {"formula", "text", "heading"} for r in regions)

    finally:
        client.stop()
        assert client.is_running is False


def test_production_mode_engine_unavailable_without_models(tmp_path):
    from pagepilot.ocr.worker_process import OCRWorkerEngine
    from pagepilot.ocr.worker_protocol import MODE_ENGINE_UNAVAILABLE

    # Instantiate engine with simulation explicitly disabled
    engine = OCRWorkerEngine(models_dir=str(tmp_path / "empty_models"), simulation_mode=False)
    assert engine.simulation_mode is False

    # Verify that without simulation mode, missing models/failures return ENGINE_UNAVAILABLE
    res_text = engine.recognize_text(str(tmp_path / "non_existent.png"), simulation_mode=False)
    assert res_text["inference_mode"] == MODE_ENGINE_UNAVAILABLE
    assert len(res_text["regions"]) == 0
    assert "error" in res_text

    res_form = engine.recognize_formula(str(tmp_path / "non_existent.png"), simulation_mode=False)
    assert res_form["inference_mode"] == MODE_ENGINE_UNAVAILABLE
    assert "error" in res_form

    res_page = engine.process_page(
        str(tmp_path / "non_existent.png"),
        document_id=1,
        page_number=1,
        filename="test.png",
        file_hash="hash1",
        simulation_mode=False,
    )
    assert res_page["inference_mode"] == MODE_ENGINE_UNAVAILABLE
    assert len(res_page["regions"]) == 0
    assert "error" in res_page


def test_production_mode_worker_client_rejects_simulation(ocr_production_mode):
    client = OCRWorkerClient(simulation_mode=False)
    assert client.simulation_mode is False


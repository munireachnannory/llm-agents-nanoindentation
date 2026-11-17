"""
sim_server.py
=============
Flask server exposing the nanoindenter simulator on the same routes as the real
InstrumentServer (``/ping``, ``/call_auto``, downloads, alignment).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_file

from .nanoindent_simulator import NanoindentSimulator

_simulator: Optional[NanoindentSimulator] = None


def get_simulator(data_dir: Optional[str] = None) -> NanoindentSimulator:
    global _simulator
    if _simulator is None:
        root = data_dir or os.environ.get("NANOINDENT_SIM_DATA_DIR", "./sim_instrument_data")
        _simulator = NanoindentSimulator(data_dir=root)
    return _simulator


def reset_simulator(data_dir: Optional[str] = None) -> NanoindentSimulator:
    global _simulator
    _simulator = NanoindentSimulator(data_dir=data_dir or os.environ.get("NANOINDENT_SIM_DATA_DIR", "./sim_instrument_data"))
    return _simulator


def _expected_token() -> str:
    return os.environ.get("NANOINDENT_TOKEN", os.environ.get("INSTRUMENT_TOKEN", ""))


def _check_token() -> Optional[tuple[Any, int]]:
    expected = _expected_token()
    if not expected:
        return None
    supplied = (
        request.headers.get("X-API-Token", "").strip()  # real InstrumentClient header
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        or (request.get_json(silent=True) or {}).get("token", "")
        or request.args.get("token", "")
    )
    if supplied != expected:
        return jsonify({"ok": False, "error": "invalid token"}), 401
    return None


def _extract_call(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = (
        payload.get("function_name")
        or payload.get("function")
        or payload.get("name")
        or payload.get("method")  # older InstrumentClient sends "method"
        or ""
    )
    kwargs = payload.get("kwargs") or payload.get("args") or {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    return str(name), kwargs


def create_app(data_dir: Optional[str] = None) -> Flask:
    app = Flask(__name__)
    sim = reset_simulator(data_dir) if data_dir is not None else get_simulator()

    @app.get("/ping")
    def ping():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        return jsonify(
            {
                "ok": True,
                "result": "pong",
                "simulated": True,
                "message": "Nanoindent instrument simulator",
            }
        )

    @app.post("/call_auto")
    def call_auto():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        payload = request.get_json(silent=True) or {}
        name, kwargs = _extract_call(payload)
        if not name:
            return jsonify({"ok": False, "error": "missing function_name"}), 400
        try:
            result = sim.dispatch(name, **kwargs)
            return jsonify({"ok": True, "result": result, "simulated": True})
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400

    @app.post("/restore_alignment")
    def restore_alignment():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        payload = request.get_json(silent=True) or {}
        try:
            result = sim.restore_alignment(**payload)
            return jsonify({"ok": True, "result": result, "simulated": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.post("/move_to_aligned_point")
    def move_to_aligned_point():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        payload = request.get_json(silent=True) or {}
        try:
            result = sim.move_to_aligned_point(**payload)
            return jsonify({"ok": True, "result": result, "simulated": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.get("/last_export")
    def last_export():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        return jsonify({"ok": True, "result": sim.state.last_export, "simulated": True})

    @app.post("/download_file")
    def download_file():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        payload = request.get_json(silent=True) or {}
        directory = payload.get("directory") or request.args.get("directory", "")
        filename = payload.get("filename") or request.args.get("filename", "")
        if not filename:
            return jsonify({"ok": False, "error": "filename required"}), 400
        if not directory:
            directory = "captures"
        # Clients pass the remote directory as an absolute Windows path; the
        # simulator only serves from its data dir, so fall back to the basename.
        try:
            src = sim.resolve_download_path(directory, filename)
        except (FileNotFoundError, ValueError):
            try:
                src = sim.resolve_download_path(Path(directory).name or "captures", filename)
            except (FileNotFoundError, ValueError) as e:
                return jsonify({"ok": False, "error": str(e)}), 404
        # Stream file bytes, matching the real InstrumentServer contract that
        # InstrumentClient.download_file and the observe agent rely on.
        return send_file(src, as_attachment=True, download_name=filename)

    @app.get("/download_file")
    def download_file_get():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        directory = request.args.get("directory", "")
        filename = request.args.get("filename", "")
        try:
            src = sim.resolve_download_path(directory, filename)
        except (FileNotFoundError, ValueError) as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        return send_file(src, as_attachment=True, download_name=filename)

    @app.post("/download_last_export")
    def download_last_export():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        info = sim.state.last_export
        if not info:
            return jsonify({"ok": False, "error": "no export recorded"}), 404
        payload = {
            "directory": info.get("directory", "exports"),
            "filename": info.get("filename", ""),
            "local_dir": (request.get_json(silent=True) or {}).get("local_dir", "downloads"),
        }
        with app.test_request_context(json=payload):
            return download_file()

    @app.get("/sim/state")
    def sim_state():
        """Debug endpoint: current simulator state snapshot."""
        auth_err = _check_token()
        if auth_err:
            return auth_err
        st = sim.state
        return jsonify(
            {
                "ok": True,
                "x_um": st.x_um,
                "y_um": st.y_um,
                "extension_um": st.extension_um,
                "engaged": st.engaged,
                "method_type": st.method_type,
                "test": {
                    "sample_name": st.test.sample_name,
                    "status": st.test.status,
                    "test_number": st.test.test_number,
                    "expected_indents": st.test.expected_indents,
                },
                "alignments": list(st.alignments.keys()),
                "command_count": len(st.command_log),
                "simulated": True,
            }
        )

    @app.post("/sim/reset")
    def sim_reset():
        auth_err = _check_token()
        if auth_err:
            return auth_err
        sim.reset()
        return jsonify({"ok": True, "message": "simulator reset", "simulated": True})

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Nanoindent instrument simulator server")
    parser.add_argument("--host", default=os.environ.get("NANOINDENT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NANOINDENT_PORT", "5000")))
    parser.add_argument("--data-dir", default=os.environ.get("NANOINDENT_SIM_DATA_DIR", "./sim_instrument_data"))
    args = parser.parse_args()

    app = create_app(data_dir=args.data_dir)
    print(f"Nanoindent SIM server on http://{args.host}:{args.port}  (data: {args.data_dir})")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

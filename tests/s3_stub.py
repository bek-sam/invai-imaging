"""A tiny stub S3 server (python -I subprocess on a free port) that records PUT headers.

Never MinIO, never a real bucket: the test starts it itself and stops it.
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

_STUB = r"""
import http.server, json, sys
LOG = sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _read(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if self.headers.get("Expect") == "100-continue":
            pass
        return body
    def do_PUT(self):
        body = self._read()
        with open(LOG, "a") as f:
            f.write(json.dumps({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "n": len(body)}) + "\n")
        self.send_response(200)
        self.send_header("ETag", '"abc"')
        self.send_header("Content-Length", "0")
        self.end_headers()
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StubS3:
    def __init__(self, tmp: Path):
        self.port = _free_port()
        self.log = tmp / "stub-puts.jsonl"
        self.script = tmp / "stub_s3_server.py"
        self.script.write_text(_STUB)
        self.proc: subprocess.Popen | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(self.script), str(self.port), str(self.log)]
        )
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return self
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("stub S3 did not start")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)

    def puts(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line]

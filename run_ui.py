#!/usr/bin/env python3
"""Run the ReceiptGuard UI with mock backend."""
import json
import os
import hashlib
import http.server
import socketserver
import tempfile
from pathlib import Path

PORT = 8080
FRONTEND_DIR = Path(__file__).parent / "frontend"

# In-memory "database" of known legitimate receipts (simulates 2-database setup)
known_receipts: dict[str, dict] = {}


class MockAPIHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Parse multipart to extract filename and content for fingerprinting
            boundary = None
            ct = self.headers.get("Content-Type", "")
            if "boundary=" in ct:
                boundary = ct.split("boundary=")[1].split(";")[0].strip()

            filename = "receipt.jpg"
            content_hash = None

            if boundary:
                try:
                    parts = body.split(b"--" + boundary.encode())
                    for part in parts:
                        if b"Content-Disposition" in part and b"name=\"image\"" in part:
                            headers_end = part.find(b"\r\n\r\n") + 4
                            file_data = part[headers_end:]
                            file_data = file_data.rstrip(b"\r\n--")
                            content_hash = hashlib.md5(file_data).hexdigest()
                            # Extract filename from disposition header
                            for line in part.split(b"\r\n"):
                                if b"filename=" in line:
                                    fn = line.split(b"filename=")[1].strip(b'"').decode()
                                    if fn:
                                        filename = fn
                                    break
                            break
                except Exception:
                    content_hash = hashlib.md5(body).hexdigest()

            if not content_hash:
                content_hash = hashlib.md5(body).hexdigest()

            # 2-database formula: if receipt fingerprint is already known, it's FRAUD
            if content_hash in known_receipts:
                existing = known_receipts[content_hash]
                response = {
                    "filename": filename,
                    "verdict": "FRAUD",
                    "confidence": 0.97,
                    "similarity": 0.94,
                    "detail": f"Duplicate receipt detected! Previously seen as '{existing['filename']}'. This is likely a fraud attempt using a copied receipt.",
                }
            else:
                known_receipts[content_hash] = {"filename": filename}
                response = {
                    "filename": filename,
                    "verdict": "LEGITIMATE",
                    "confidence": 0.89,
                    "similarity": 0.08,
                    "detail": "New receipt registered in legitimate database. First time seeing this receipt.",
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return
        self.send_error(405)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path == "/predict/batch":
            response = {
                "results": [],
                "total_processed": 0,
                "fraud_count": 0,
                "suspicious_count": 0,
                "legitimate_count": 0,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return
        if self.path == "/health":
            response = {
                "status": "healthy",
                "models_loaded": True,
                "known_receipts": len(known_receipts),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return
        super().do_GET()

    def translate_path(self, path):
        if path == "/" or path == "":
            path = "/index.html"
        return str(FRONTEND_DIR / path.lstrip("/"))


if __name__ == "__main__":
    os.chdir(str(FRONTEND_DIR))
    with socketserver.TCPServer(("", PORT), MockAPIHandler) as httpd:
        print(f"🏪 ReceiptGuard UI running at http://localhost:{PORT}")
        print(f"📁 2-Database Mode: 1st upload = LEGITIMATE, 2nd upload (same image) = FRAUD")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()

#!/usr/bin/env bash
# Container entry point: the API on localhost only, then the UI on the public port.
set -euo pipefail

uvicorn query_tool.api.main:app --host 127.0.0.1 --port 8000 &

# Wait for the API, so the first page view doesn't report it offline.
python - <<'EOF'
import time
import urllib.request

for _ in range(180):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/api/v1/health", timeout=2)
        break
    except Exception:
        time.sleep(1)
EOF

# Spaces show the app inside an iframe on huggingface.co, where Streamlit's XSRF check
# blocks file uploads, so it is turned off here (and only here).
exec streamlit run frontend/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.enableXsrfProtection false \
  --server.enableCORS false

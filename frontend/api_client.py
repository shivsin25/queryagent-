"""Thin HTTP client for the Query Tool API. The UI never imports backend code directly."""

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class ApiError(Exception):
    pass


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(item.get("msg", str(item)) for item in detail)
    return str(detail)


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        base_url = base_url or os.getenv("QT_API_URL", DEFAULT_API_URL)
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            kwargs.setdefault("timeout", self._timeout)
            response = httpx.request(method, self._url(path), **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url} ({exc}).") from exc
        if response.status_code >= 400:
            raise ApiError(_detail(response))
        return response

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5).json()

    def usage(self) -> dict:
        return self._request("GET", "/usage", timeout=5).json()

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents").json()

    def upload(self, files: list[tuple[str, bytes]]) -> dict:
        payload = [("files", (name, data)) for name, data in files]
        return self._request("POST", "/documents", files=payload).json()

    def delete(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}")

    def feedback(self, request_id: str, rating: str) -> None:
        self._request("POST", "/feedback", json={"request_id": request_id, "rating": rating})

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}")

    def query_stream(
        self,
        question: str,
        top_k: int,
        max_revisions: int,
        session_id: str | None = None,
        doc_ids: list[str] | None = None,
        retry_of: str | None = None,
        retry_feedback: str | None = None,
    ) -> Iterator[dict]:
        body = {"question": question, "top_k": top_k, "max_revisions": max_revisions,
                "session_id": session_id, "doc_ids": doc_ids or None, "retry_of": retry_of,
                "retry_feedback": retry_feedback}
        try:
            with httpx.stream(
                "POST", self._url("/query/stream"), json=body, timeout=self._timeout
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(_detail(response))
                for line in response.iter_lines():
                    if line.strip():
                        yield json.loads(line)
        except httpx.HTTPError as exc:
            raise ApiError(f"Connection to the API failed ({exc}).") from exc

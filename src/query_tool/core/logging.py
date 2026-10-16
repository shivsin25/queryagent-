import logging
import re
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Credentials that provider libraries can echo in error messages and tracebacks (an HTTP
# client once logged a full "Bearer sk-..." header). Logs are shared and kept, so they never
# hold a key.
_SECRET_RE = re.compile(
    r"(Bearer\s+)[^\s'\"]+"
    r"|sk-[A-Za-z0-9_-]{16,}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|gsk_[A-Za-z0-9]{16,}"
    r"|hf_[A-Za-z0-9]{16,}"
)


def redact(text: str) -> str:
    return _SECRET_RE.sub(lambda m: f"{m.group(1) or ''}[redacted]", text)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class RedactingFormatter(logging.Formatter):
    """Formats as usual, then removes credentials, including from tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

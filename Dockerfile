# One container for Hugging Face Spaces (Docker SDK), or any container host.
# The Streamlit UI listens on 7860, the port Spaces exposes. The API runs on localhost:8000
# inside the container and is not reachable from outside.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/app/.venv/bin:${PATH}" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    QT_API_URL=http://127.0.0.1:8000

# Dependencies first, so a code change doesn't reinstall them.
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project --extra gemini --extra openai

COPY --chown=user . .
RUN uv sync --frozen --no-dev --extra gemini --extra openai

# Bake the search models and the indexed sample documents into the image, so the app
# starts without downloading anything.
RUN python scripts/warm_models.py && qt seed

EXPOSE 7860
CMD ["bash", "scripts/start_space.sh"]

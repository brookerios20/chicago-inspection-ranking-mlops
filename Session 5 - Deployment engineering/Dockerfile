# ---- 1. Base image -------------------------------------------------------
# Pin a specific version, not "latest". "slim" drops compilers/docs (~5x smaller).
FROM python:3.13-slim

# ---- 2. Environment settings ---------------------------------------------
# Don't write .pyc files; send print/log output straight to `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# All following paths are relative to this directory inside the image.
WORKDIR /serving

# ---- 3. System packages (only if needed) ---------------------------------
# Example: curl for the HEALTHCHECK below. Clean the apt cache in the same
# RUN so it doesn't bloat the layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ---- 4. Python dependencies (before code, for caching) --------------------
# Docker caches each step. Copying ONLY requirements.txt first means editing
# your code later won't re-run the slow pip install.
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 5. Application code -------------------------------------------------
# Option A: copy the whole serving folder (app.py + model/ + requirements.txt)
COPY serving/ .

# ---- 6. Run as a non-root user -------------------------------------------
RUN useradd --create-home appuser
USER appuser

# ---- 7. Document the port and add a health check -------------------------
# EXPOSE is documentation; you still publish the port with `docker run -p`.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# ---- 8. Start command ----------------------------------------------------
# Exec form (JSON array) so the server receives stop signals properly.
# Bind 0.0.0.0, or nothing outside the container can reach it.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

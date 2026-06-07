# Dockerfile for the README "Quick Start" workflow.
# Builds an image with the study toolkit's dependencies and scripts so you can
# run benchmarks against Baseten-hosted models.

FROM python:3.11-slim

# Install dependencies from the Quick Start section.
# `uvx` (from the `uv` toolchain) is used to run `truss login`.
RUN pip install --no-cache-dir \
        truss \
        openai \
        requests \
        numpy \
        tabulate \
        matplotlib \
        uv

WORKDIR /app

# Bring in the scripts the Quick Start runs.
COPY scripts/ ./scripts/

# Authenticate at runtime (interactive), e.g.:
#   docker run -it --rm baseten-study uvx truss login
#
# Then run your first benchmark, e.g.:
#   docker run --rm baseten-study python scripts/benchmark.py --model "deepseek-v3" --requests 100
#
# Default command runs the Quick Start benchmark. Override as needed.
CMD ["python", "scripts/benchmark.py", "--model", "deepseek-v3", "--requests", "100"]

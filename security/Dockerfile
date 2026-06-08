# Builds an image with the OSS components that are part of the Quick Start

FROM python:3.11-slim

# Install dependencies from the Quick Start section.
RUN pip install --no-cache-dir \
        truss \
        openai \
        requests \
        numpy \
        tabulate \
        matplotlib \
        uv

WORKDIR /app

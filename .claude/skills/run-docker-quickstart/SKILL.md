---
name: run-docker-quickstart
description: Build and run the README "Quick Start" benchmark toolkit in Docker. Use when asked to build the Dockerfile, run the benchmark scripts in a container, or set up a container runtime on this machine. Covers the sudo-free Colima/Lima/Docker-CLI install for Apple Silicon Macs that lack Homebrew/Docker.
---

# Run the Quick Start toolkit in Docker

The `Dockerfile` at the repo root packages the README "Quick Start" workflow:
`pip install` of the SA toolkit deps, `uvx truss login` (interactive, runtime
only), and `python scripts/benchmark.py`. This skill builds that image and
drives it far enough to prove it works.

## Prerequisites: container runtime (Apple Silicon, no Homebrew/sudo)

This machine has **no Docker, no Homebrew, and `sudo` cannot read a password**
(it only pops a GUI dialog that fails non-interactively). Do NOT try to install
Homebrew or Docker Desktop — both need `sudo` to write `/opt/homebrew` or
`/Applications` and will fail.

Instead install a **sudo-free** stack into `~/.local`. Colima + Lima run on
macOS's built-in Virtualization framework (`--vm-type vz`), so no QEMU and no
admin rights are needed. Pin to known-good versions; bump if a download 404s.

```bash
mkdir -p ~/.local/bin ~/.local && cd /tmp

# Lima (bundles limactl + guest agents; supports native vz on Apple Silicon)
curl -fsSL -o lima.tar.gz https://github.com/lima-vm/lima/releases/download/v2.1.2/lima-2.1.2-Darwin-arm64.tar.gz
tar -xzf lima.tar.gz -C ~/.local

# Colima
curl -fsSL -o ~/.local/bin/colima https://github.com/abiosoft/colima/releases/download/v0.10.3/colima-Darwin-arm64
chmod +x ~/.local/bin/colima

# Docker CLI (host-side client only; Colima provides the daemon)
ver=$(curl -fsSL https://download.docker.com/mac/static/stable/aarch64/ | grep -oE 'docker-[0-9.]+\.tgz' | sort -V | tail -1)
curl -fsSL -o docker.tgz "https://download.docker.com/mac/static/stable/aarch64/$ver"
tar -xzf docker.tgz && cp docker/docker ~/.local/bin/docker && chmod +x ~/.local/bin/docker
```

`~/.local/bin` is NOT on PATH in a fresh shell — prefix every command in this
session with `export PATH="$HOME/.local/bin:$PATH"` (or add it to `~/.zshrc`).

Start the VM (one-time per boot; persists across builds):

```bash
export PATH="$HOME/.local/bin:$PATH"
colima start --vm-type vz --cpu 2 --memory 4 --disk 20
docker info --format '{{.ServerVersion}}'   # confirms daemon reachable
```

`colima start` auto-creates and selects the `colima` docker context. Stop with
`colima stop` to free resources; restart with `colima start --vm-type vz`.

> Heads-up: the VM is a process on the Mac. **Closing the laptop lid sleeps it
> and stalls any in-progress build.** Build on power / lid open.

## Build

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /private/tmp/baseten-study
docker build -t baseten-study .
```

The `pip install` layer pulls truss + matplotlib + numpy and friends (~several
minutes the first time, cached after). Final image is ~195 MB compressed.

## Drive it (verify, don't just launch)

The default `CMD` runs a 100-request benchmark that needs a real
`BASETEN_API_KEY` and network — so for a self-contained check, drive the parts
that don't need credentials:

```bash
export PATH="$HOME/.local/bin:$PATH"

# deps import + uvx present + scripts copied in
docker run --rm baseten-study sh -c '
  python -c "import openai, requests, numpy, tabulate, matplotlib, truss; print(truss.__version__)" &&
  uvx --version && ls scripts/'

# entrypoint + argparse actually run
docker run --rm baseten-study python scripts/benchmark.py --help
```

Both should exit 0. A real benchmark run is the one step only the user can
complete (it needs Baseten auth):

```bash
docker run -it --rm baseten-study uvx truss login          # interactive auth
docker run --rm -e BASETEN_API_KEY=sk-... baseten-study     # default CMD benchmark
```

## Gotchas

- **`uvx truss login` is NOT baked into the image** — interactive auth can't be
  a build step, and login state doesn't persist between `docker run` calls
  unless you mount a volume for truss's config. Run it interactively each
  session, or pass `-e BASETEN_API_KEY=...`.
- **Base image is `python:3.11-slim`**, not the host's 3.9 — truss and recent
  numpy/matplotlib drop 3.9 support.
- If a pinned release URL 404s, fetch the latest tag:
  `curl -fsSL https://api.github.com/repos/abiosoft/colima/releases/latest | grep tag_name`
  (same for `lima-vm/lima`).

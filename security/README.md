# Security
The intent is to give rightly skeptical folks the option to scan the tools in use by the quickstart.

## Method
Use Trivy (safe now, though mindful of [3/2026 supply chain attack](https://www.paloaltonetworks.com/blog/cloud-security/trivy-supply-chain-attack/)) to scan a Docker container isolating the utilities installed in the root of this repo, namely: uv, truss, openai, requests, numpy, tabulate,and matplotlib

## Reasoning
Open Source code, even when well-maintained by experts, is a common target for software supply chain attackers due to the wide distribution and innate trust that developers give even `:latest` tagged images.

## How
Build the image from the `Dockerfile` contained herein as follows:
`docker buildx build --platform linux/amd64,linux/arm64 \
 -t <your_docker_hub_account>/b10-utils --push .`

Scan it with Trivy (caveat above):
`docker run --rm \
  -v trivy-cache:/root/.cache/ \
  aquasec/trivy:latest image <your_docker_hub_account>/b10-utils`

You'll get a list of CVEs that look similar to this:
Total: 106 (UNKNOWN: 5, LOW: 63, MEDIUM: 29, HIGH: 7, CRITICAL: 2)

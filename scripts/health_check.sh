#!/bin/bash
# health_check.sh — Quick endpoint health and latency check
#
# Usage:
#   ./health_check.sh                          # Default Baseten endpoint
#   ./health_check.sh https://model.api.baseten.co/v1
#   ./health_check.sh http://localhost:8000/v1  # Local vLLM server

set -euo pipefail

ENDPOINT="${1:-https://bridge.baseten.co/v1/direct}"
API_KEY="${BASETEN_API_KEY:-}"
MODEL="${2:-meta-llama/Llama-3.1-70B-Instruct}"
NUM_REQUESTS=5

echo "======================================"
echo "  Endpoint Health Check"
echo "======================================"
echo "  Target:   $ENDPOINT"
echo "  Model:    $MODEL"
echo "  Requests: $NUM_REQUESTS"
echo ""

# Build auth header
AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
    AUTH_HEADER="-H \"Authorization: Bearer $API_KEY\""
fi

LATENCIES=()
ERRORS=0

for i in $(seq 1 $NUM_REQUESTS); do
    START_MS=$(python3 -c "import time; print(int(time.time() * 1000))")

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$ENDPOINT/chat/completions" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        --max-time 30 \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word.\"}],
            \"max_tokens\": 5
        }" 2>/dev/null) || HTTP_CODE="TIMEOUT"

    END_MS=$(python3 -c "import time; print(int(time.time() * 1000))")
    LATENCY_MS=$((END_MS - START_MS))

    if [ "$HTTP_CODE" = "200" ]; then
        STATUS="OK"
        LATENCIES+=($LATENCY_MS)
    else
        STATUS="FAIL ($HTTP_CODE)"
        ERRORS=$((ERRORS + 1))
    fi

    printf "  Request %d: %-15s %6d ms\n" "$i" "$STATUS" "$LATENCY_MS"
done

echo ""
echo "  Summary:"
echo "  --------"

if [ ${#LATENCIES[@]} -gt 0 ]; then
    # Calculate stats with python
    python3 -c "
import sys
latencies = [${LATENCIES[*]}]
latencies.sort()
n = len(latencies)
print(f'  Successful: {n}/$NUM_REQUESTS')
print(f'  Min:        {min(latencies)} ms')
print(f'  Max:        {max(latencies)} ms')
print(f'  Median:     {latencies[n//2]} ms')
print(f'  Mean:       {sum(latencies)//n} ms')
"
else
    echo "  All requests failed!"
fi

if [ $ERRORS -gt 0 ]; then
    echo "  Errors:     $ERRORS/$NUM_REQUESTS"
    echo ""
    echo "  Troubleshooting:"
    echo "  - Check API key: echo \$BASETEN_API_KEY"
    echo "  - Check endpoint URL"
    echo "  - Check if model is deployed and active"
fi

echo ""
echo "======================================"

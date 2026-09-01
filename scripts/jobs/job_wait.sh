#!/bin/bash
# Wait for a serving job's health endpoint. Usage: ./job_wait.sh <job-id> <port>
set -eu
JOB=$1; PORT=$2
TOKEN="${HF_TOKEN:-$(hf auth token 2>/dev/null)}"
for i in $(seq 1 60); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    "https://$JOB--$PORT.hf.jobs/health" -H "Authorization: Bearer $TOKEN" || true)
  if [ "$CODE" = "200" ]; then
    echo "UP: https://$JOB--$PORT.hf.jobs"
    exit 0
  fi
  sleep 20
done
echo "TIMEOUT waiting for $JOB (last HTTP $CODE)" >&2
exit 1

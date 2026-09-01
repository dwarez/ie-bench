#!/bin/bash
# Debug an Inference Endpoint: stored config, state, error, GGUF path validation, recent errors.
# Usage: ./endpoint_debug.sh <namespace>/<endpoint-name>
# Needs HF_TOKEN (or `hf auth token`) with read access to the namespace.
set -eu
NS=${1%%/*}; NAME=${1##*/}
TOKEN="${HF_TOKEN:-$(hf auth token 2>/dev/null)}"
API="https://api.endpoints.huggingface.cloud/v2/endpoint/$NS/$NAME"
CFG=$(curl -sf -H "Authorization: Bearer $TOKEN" "$API")

echo "=== config ==="
jq '{
  state: .status.state, message: .status.message, error: .status.errorMessage,
  fromCatalog: .model.fromCatalog, repository: .model.repository, revision: .model.revision,
  engine_image: .model.image,
  args: .model.args, command: .model.command, env: .model.env,
  compute: .compute | {instanceType, instanceSize}
}' <<<"$CFG"

# The stored config is a snapshot taken at creation and is never reconciled with the catalog.
# Upstream re-quantisation (renamed/deleted .gguf) leaves it pointing at a 404 while the catalog
# UI shows a corrected path, and the platform starts the container anyway. Verify every path.
echo "=== model files @ pinned revision ==="
REPO=$(jq -r '.model.repository // empty' <<<"$CFG")
REV=$(jq -r '.model.revision // "main"' <<<"$CFG")
FILES=$(jq -r '[.model.image[]? | .modelPath?, .mmprojModelPath?, .specModelPath?]
               | map(select(type == "string" and . != "")) | unique | .[]' <<<"$CFG")
if [ -z "$REPO" ] || [ -z "$FILES" ]; then
  echo "(no explicit model file paths in config — engine resolves them itself)"
else
  MISSING=0
  while IFS= read -r f; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer $TOKEN" \
      "https://huggingface.co/$REPO/resolve/$REV/$f")
    case "$code" in
      2*|3*) echo "  ok       $f" ;;
      *)     echo "  MISSING  $f  (HTTP $code)"; MISSING=1 ;;
    esac
  done <<<"$FILES"
  if [ "$MISSING" = 1 ]; then
    echo "  -- available .gguf in $REPO@${REV:0:8}:"
    curl -sf -H "Authorization: Bearer $TOKEN" \
      "https://huggingface.co/api/models/$REPO/revision/$REV?blobs=true" \
      | jq -r '.siblings[] | select(.rfilename | endswith(".gguf"))
               | "     \(.size / 1e9 | floor)GB\t\(.rfilename)"' | sort -k2
  fi
fi

# /logs returns text/plain (one "- <ts> <line>" per row) and ignores level/limit query params.
echo "=== recent error logs ==="
LOGS=$(curl -sf -H "Authorization: Bearer $TOKEN" "$API/logs" || true)
if [ -z "$LOGS" ]; then
  echo "(no logs — endpoint may be scaled to zero; resume it first)"
else
  ERRS=$(grep -E ' E{1,} | ERROR | CRITICAL |Traceback|[Ee]rror:' <<<"$LOGS" | tail -20 || true)
  [ -n "$ERRS" ] && echo "$ERRS" || { echo "(no error lines; last 20 of $(wc -l <<<"$LOGS") lines)"; tail -20 <<<"$LOGS"; }
fi

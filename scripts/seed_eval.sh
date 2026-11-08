#!/usr/bin/env bash
# Seed corpus for scripts/eval.sh (deterministic titles the eval set matches).
set -uo pipefail
BASE=http://localhost:8000
ADMIN_ACME=${ACME_ADMIN:-sk-acme-admin}
ADMIN_GLOBEX=${GLOBEX_ADMIN:-sk-globex-admin}

ingest() { # key, file, title
  local out id
  out=$(curl -s -m 120 -X POST "$BASE/ingest" \
    -H "Authorization: Bearer $1" -F "file=@$2" -F "title=$3" \
    -F "authority=official")
  echo "$out" >/tmp/eval_seed_last.json
  id=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('task_id',''))" "$out")
  # curl without -f still exits 0 on 4xx/5xx, so "did we get a task_id?" is
  # the actual success signal — don't rely on curl's exit code.
  if [ -n "$id" ]; then
    echo "ingest($3) → task $id"
    return 0
  fi
  echo "ingest($3) FAILED: ${out:0:200}" >&2
  return 1
}

poll() { # task_id key
  for _ in $(seq 1 40); do
    STATE=$(curl -s "$BASE/tasks/$1" -H "$2" | python3 -c "import json,sys;print(json.load(sys.stdin).get('state',''))" 2>/dev/null)
    [ "$STATE" = "SUCCESS" ] && echo "done" && return 0
    sleep 2
  done
  echo "TIMEOUT waiting for $1"; return 1
}

ingest "$ADMIN_ACME" "docs/acme_runbook.md" "Acme Runbook"
INGESTED=$?
[ "$INGESTED" != "0" ] && exit 1
T=$(python3 -c "import json;print(json.load(open('/tmp/eval_seed_last.json'))['task_id'])")
poll "$T" "X-API-Key: sk-acme-reader" || exit 1

ingest "$ADMIN_GLOBEX" "docs/globex_handbook.md" "Globex Handbook"
[ $? != 0 ] && exit 1
T=$(python3 -c "import json;print(json.load(open('/tmp/eval_seed_last.json'))['task_id'])")
poll "$T" "X-API-Key: sk-globex-reader" || exit 1

echo "Seed complete — run scripts/eval.py to score it or rerun the eval gate."

#!/usr/bin/env bash
# Manual test battery for the RAG engine (auth/RBAC, errors, tenancy).
# Usage: bash scripts/manual_test.sh   (API on :8000, worker running)
# NOTE: the query tests need a real LLM_API_KEY; they still run and are
# expected to fail with 502 while the key is placeholder.
set -u
BASE=${BASE:-http://localhost:8000}
PASS=0; FAIL=0

check() { # name, expected-substring, actual body (raw match)
  local name=$1 expected=$2 body=$3
  if [[ "$body" == *"$expected"* ]]; then
    echo "✅ $name"; PASS=$((PASS+1))
  else
    echo "❌ $name — expected ~'$expected', got: ${body:0:200}"; FAIL=$((FAIL+1))
  fi
}

not_in() { # name, forbidden-substring, actual body
  local name=$1 bad=$2 body=$3
  if [[ "$body" != *"$bad"* ]]; then
    echo "✅ $name"; PASS=$((PASS+1))
  else
    echo "❌ $name — found forbidden '$bad' in: ${body:0:200}"; FAIL=$((FAIL+1))
  fi
}

jqfield() { python3 -c "import json,sys;print(json.load(open('/tmp/last_body')).get(sys.argv[1],''))" "$1" 2>/dev/null; }

# poll_task <task_id> <api-key-header> — waits until the task leaves
# PENDING/STARTED (max ~45 s), echoing the final status JSON.
poll_task() {
  local id=$1 hdr=$2 i state out=""
  for i in $(seq 1 15); do
    curl -s "$BASE/tasks/$id" -H "$hdr" > /tmp/task_body 2>/dev/null
    state=$(python3 -c "import json;print(json.load(open('/tmp/task_body')).get('state',''))" 2>/dev/null)
    if [[ "$state" != "PENDING" && "$state" != "STARTED" ]]; then break; fi
    sleep 3
  done
  cat /tmp/task_body
}

ingest() { # admin-key, file, title, extra-form-args...
  local key=$1 file=$2 title=$3; shift 3
  curl -s -X POST "$BASE/ingest" -H "Authorization: Bearer $key" \
    -F "file=@$file" -F "title=$title" "$@" > /tmp/last_body
}

echo "== 0. Health (no key) =="
H=$(curl -s "$BASE/health")
check "health is keyless" '"status":"ok"' "$H"

echo "== 1/2. Authn: missing + unknown key =="
B=$(curl -s -X POST "$BASE/query" -H "Content-Type: application/json" -d '{"question":"x"}')
check "no key → 401 message" "Missing credentials" "$B"
B=$(curl -s -X POST "$BASE/query" -H "Authorization: Bearer sk-nope" -H "Content-Type: application/json" -d '{"question":"x"}')
check "unknown key → 401 message" "Invalid or unknown API key" "$B"
B=$(curl -s -X POST "$BASE/query" -H "Authorization: Peanut sk-acme-admin" -H "Content-Type: application/json" -d '{"question":"x"}')
check "wrong scheme → message" "Bearer <api-key>" "$B"

echo "== 3. RBAC: reader cannot ingest =="
C1=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/ingest" -H "Authorization: Bearer sk-acme-reader" -F "file=@docs/acme_runbook.md" -F "title=x")
C2=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/ingest" -H "X-API-Key: sk-acme-reader" -F "file=@docs/acme_runbook.md" -F "title=x")
if [[ "$C1" == "403" ]]; then echo "✅ reader ingest → 403 (Bearer)"; PASS=$((PASS+1)); else echo "❌ expected 403 got $C1"; FAIL=$((FAIL+1)); fi
if [[ "$C2" == "403" ]]; then echo "✅ reader ingest → 403 (X-API-Key)"; PASS=$((PASS+1)); else echo "❌ expected 403 got $C2"; FAIL=$((FAIL+1)); fi

echo "== 4. Upload validation =="
echo "not a real pdf" > /tmp/evil.txt
B=$(curl -s -X POST "$BASE/ingest" -H "Authorization: Bearer sk-acme-admin" -F "file=@/tmp/evil.txt" -F "title=t")
check "txt upload → 415" "Unsupported file type" "$B"

ingest sk-acme-admin docs/acme_runbook.md "Acme Runbook" -F "authority=official"
check "admin ingest → 202 with status_url" "status_url" "$(cat /tmp/last_body)"
ACME_TASK=$(jqfield task_id)

echo "== 5. Task status =="
B=$(poll_task "$ACME_TASK" "X-API-Key: sk-acme-reader")
if [[ "$B" == *'"status":"indexed"'* ]]; then
  echo "✅ task indexed"; PASS=$((PASS+1))
  if [[ "$B" == *'"chunks"'* ]]; then echo "✅ result has chunks count"; PASS=$((PASS+1)); else echo "❌ no chunks count: ${B:0:200}"; FAIL=$((FAIL+1)); fi
elif [[ "$B" == *'"status":"skipped"'* ]]; then
  # Re-run against existing data: same content already indexed.
  echo "✅ task skipped (content already indexed — idempotent)"; PASS=$((PASS+1))
else
  echo "❌ task did not succeed: ${B:0:200}"; FAIL=$((FAIL+1))
fi

echo "== 6. Idempotent re-ingest (same bytes, racing task) =="
ingest sk-acme-admin docs/acme_runbook.md "Acme Runbook"
T2=$(jqfield task_id)
B=$(poll_task "$T2" "X-API-Key: sk-acme-reader")
check "identical re-upload → skipped" '"status":"skipped"' "$B"

echo "== 7. Globex ingest =="
ingest sk-globex-admin docs/globex_handbook.md "Globex Handbook"
GT=$(jqfield task_id)
B=$(poll_task "$GT" "X-API-Key: sk-globex-reader")
# First run → "indexed"; battery re-runs → "skipped" (same content hash).
if [[ "$B" == *'"status":"indexed"'* || "$B" == *'"status":"skipped"'* ]]; then
  echo "✅ globex task succeeded (indexed/skipped)"; PASS=$((PASS+1))
else
  echo "❌ globex task failed: ${B:0:200}"; FAIL=$((FAIL+1))
fi

echo "== 8. Cross-workspace isolation =="
B=$(curl -s -m 90 -X POST "$BASE/query" -H "Authorization: Bearer sk-globex-reader" -H "Content-Type: application/json" -d '{"question":"What is the springfield calibration phrase?"}')
check "globex query answer contains ZEBRACAKE" "ZEBRACAKE" "$B"
not_in "globex query does not leak BANANA" "BANANA-RHUBARB" "$B"

B=$(curl -s -m 90 -X POST "$BASE/query" -H "Authorization: Bearer sk-acme-reader" -H "Content-Type: application/json" -d '{"question":"What is the springfield calibration phrase?"}')
not_in "acme query never sees ZEBRACAKE" "ZEBRACAKE" "$B"

B=$(curl -s -m 90 -X POST "$BASE/query" -H "Authorization: Bearer sk-acme-reader" -H "Content-Type: application/json" -d '{"question":"What is the maximum operating temperature?"}')
check "acme query sees its own doc" "85" "$B"

echo "== 9. Streaming: citations-first, then tokens =="
SSE=$(curl -s -m 90 -N -X POST "$BASE/query/stream" -H "Authorization: Bearer sk-acme-reader" -H "Content-Type: application/json" -d '{"question":"What is the deployment rollback procedure?"}')
echo "$SSE" | head -8
SSE_LINES=$(echo "$SSE" | tr -d '\n')
check "citations event present" "event: citations" "$SSE_LINES"
check "token events present" "event: token" "$SSE_LINES"
check "done event present" "event: done" "$SSE_LINES"
not_in "no error event" "event: error" "$SSE_LINES"
FIRST_EVT=$(echo "$SSE" | grep '^event:' | head -1)
check "first event is citations" "event: citations" "$FIRST_EVT"

echo "== 10. Malformed PDF: rejected at the API or cleanly failed in the worker =="
# (a) Random bytes as .pdf: the magic-byte check rejects at upload (415), no
# task is ever created, the worker never sees it.
head -c 4096 /dev/urandom > /tmp/corrupt.pdf
C=$(curl -s -X POST "$BASE/ingest" -H "Authorization: Bearer sk-acme-admin" -F "file=@/tmp/corrupt.pdf" -F "title=corrupt")
check "random bytes renamed .pdf → 415 magic-byte mismatch" "magic-byte mismatch" "$C"
# (b) A valid %PDF header followed by garbage passes the sniff (it IS pdf-
# shaped) and must die as a clean permanent FAILURE in the worker — with a
# short error message, never a traceback.
{ printf '%%PDF-1.4\n'; head -c 4096 /dev/urandom; } > /tmp/garbage.pdf
curl -s -X POST "$BASE/ingest" -H "Authorization: Bearer sk-acme-admin" -F "file=@/tmp/garbage.pdf" -F "title=garbage" > /tmp/last_body
CT=$(jqfield task_id)
B=$(poll_task "$CT" "X-API-Key: sk-acme-reader")
STATE=$(python3 -c "import json;print(json.load(open('/tmp/task_body')).get('state',''))" 2>/dev/null)
if [[ "$STATE" == "FAILURE" && "$B" == *'"error"'* ]]; then
  echo "✅ garbage pdf → FAILURE with {'error'}"; PASS=$((PASS+1))
else
  echo "❌ corrupt handling — state=$STATE body=${B:0:200}"; FAIL=$((FAIL+1))
fi
not_in "no traceback leaked to client" "traceback" "$B"

echo
echo "== 11. Task ownership: cross-workspace polling =="
# $GT is the globex-owned task from section 7; acme callers must not see it.
C=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/tasks/$GT" -H "X-API-Key: sk-acme-admin")
if [[ "$C" == "404" ]]; then
  echo "✅ acme polling globex task → 404"; PASS=$((PASS+1))
else
  echo "❌ acme polling globex task → $C (want 404)"; FAIL=$((FAIL+1))
fi
C=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/tasks/$GT" -H "X-API-Key: sk-globex-reader")
if [[ "$C" == "200" ]]; then
  echo "✅ globex polling its own task → 200"; PASS=$((PASS+1))
else
  echo "❌ globex polling own task → $C (want 200)"; FAIL=$((FAIL+1))
fi
C=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/tasks/00000000-0000-4000-8000-000000000000" -H "X-API-Key: sk-acme-admin")
if [[ "$C" == "404" ]]; then
  echo "✅ unknown task id → 404 (no existence leak)"; PASS=$((PASS+1))
else
  echo "❌ unknown task id → $C (want 404)"; FAIL=$((FAIL+1))
fi

echo "== 12. Metrics endpoint: admin-only + exposition =="
C=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics" -H "X-API-Key: sk-acme-admin")
check "admin scrape → 200" "200" "$C"
C=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics" -H "X-API-Key: sk-acme-reader")
if [[ "$C" == "403" ]]; then echo "✅ reader scrape → 403"; PASS=$((PASS+1)); else echo "❌ reader scrape → $C (want 403)"; FAIL=$((FAIL+1)); fi
C=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
if [[ "$C" == "401" ]]; then echo "✅ anon scrape → 401"; PASS=$((PASS+1)); else echo "❌ anon scrape → $C (want 401)"; FAIL=$((FAIL+1)); fi
M=$(curl -s "$BASE/metrics" -H "X-API-Key: sk-acme-admin")
not_in "exposition never leaks key material" "sk-" "$M"
grep -q "rag_engine_requests_total" <<<"$M" && { echo "✅ exposition has request counters"; PASS=$((PASS+1)); } || { echo "❌ no request counter"; FAIL=$((FAIL+1)); }

echo "== 12b. Magic-byte sniffing: mislabeled uploads rejected =="
printf '#!/bin/sh\nrm -rf /\n' > "/tmp/mt_evil.pdf"
C=$(curl -s -m 30 -X POST "$BASE/ingest" -H "X-API-Key: sk-acme-admin" -F "file=@/tmp/mt_evil.pdf" -F "title=Evil")
check "shell script renamed .pdf → 415" "magic-byte mismatch" "$C"
printf 'MZ\x90\x00\x03\x00\x00\x00binary' > "/tmp/mt_evil.md"
C=$(curl -s -m 30 -X POST "$BASE/ingest" -H "X-API-Key: sk-acme-admin" -F "file=@/tmp/mt_evil.md" -F "title=Evil")
check "PE binary renamed .md → 415" "magic-byte mismatch" "$C"
rm -f /tmp/mt_evil.pdf /tmp/mt_evil.md

echo "== 13. Relevance gate: off-corpus question refuses before the LLM =="
# Uses the acme-reader key BEFORE the burst below drains its minute budget.
B=$(curl -s -m 30 -X POST "$BASE/query" -H "Authorization: Bearer sk-acme-reader" -H "Content-Type: application/json" -d '{"question":"How long do I boil an egg for?"}')
if grep -q "no relevant material" <<<"$B"; then
  echo "✅ off-corpus /query refused (no LLM call)"; PASS=$((PASS+1))
else
  echo "❌ off-corpus query not refused: ${B:0:200}"; FAIL=$((FAIL+1))
fi
SSE=$(curl -s -m 30 -N -X POST "$BASE/query/stream" -H "Authorization: Bearer sk-acme-reader" -H "Content-Type: application/json" -d '{"question":"What is the capital of France?"}')
check "off-corpus stream → error event" "event: error" "$SSE"
check "stream refusal says no relevant material" "no relevant material" "$SSE"
not_in "refused stream produced no tokens" "event: token" "$SSE"

echo "== 14. Rate limiting: 429 on burst =="
# 65 fast authed polls as one reader: per-minute limit is 60 → the tail must
# start answering 429 with Retry-After. (Do this last: it exhausts the key's
# minute budget and would pollute later sections.)
FOUR29S=0
RETRY=""
for _ in $(seq 1 65); do
  R=$(curl -s -m 5 -D - -o /dev/null "$BASE/tasks/00000000-0000-4000-8000-000000000000" -H "X-API-Key: sk-acme-reader" 2>/dev/null | tr -d '\r')
  grep -q "^HTTP/ 429\|^HTTP/1.1 429\|^HTTP/1.0 429\|^HTTP/2 429" <<<"$R" && { FOUR29S=$((FOUR29S+1)); [ -z "$RETRY" ] && RETRY=$(grep -i '^retry-after:' <<<"$R"); }
done
if [[ "$FOUR29S" -ge 1 ]]; then
  echo "✅ burst throttled: $FOUR29S/65 requests got 429 ($([ -n "$RETRY" ] && echo "Retry-After present" || echo "note: no Retry-After header"))"; PASS=$((PASS+1))
else
  echo "❌ burst got zero 429s (limit multiplier or limiter broken?)"; FAIL=$((FAIL+1))
fi

echo
echo "=== $PASS passed, $FAIL failed ==="

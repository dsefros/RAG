#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASS="${ADMIN_PASS:-admin123}"
QUESTION="${QUESTION:-Какие документы нужны для онбординга мерчанта?}"

echo "[1] health"
curl -fsS "$BASE_URL/health"

echo "\n[2] login"
TOKEN=$(curl -fsS -X POST "$BASE_URL/api/v1/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"$ADMIN_USER\",\"password\":\"$ADMIN_PASS\"}" | python -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')

echo "[3] trigger reindex"
JOB_ID=$(curl -fsS -X POST "$BASE_URL/api/v1/admin/reindex" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{}' | python -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')

echo "job_id=$JOB_ID"

echo "[4] poll job"
for i in {1..60}; do
  STATUS=$(curl -fsS "$BASE_URL/api/v1/admin/reindex/$JOB_ID" -H "Authorization: Bearer $TOKEN" | python -c 'import json,sys;print(json.load(sys.stdin)["status"])')
  echo "status=$STATUS"
  if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]]; then
    break
  fi
  sleep 2
done

echo "[5] query"
curl -fsS -X POST "$BASE_URL/api/v1/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"question\":\"$QUESTION\"}"

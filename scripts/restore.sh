#!/usr/bin/env bash
# Replay a backup produced by scripts/backup.sh into the dockerized Postgres.
# The database is *removed and rebuilt* — web servers will 503 briefly.
# Usage: scripts/restore.sh backups/rag-20261108T033101Z.sql.gz
set -euo pipefail

[ -f "${1:-}" ] || {
  echo "Usage: scripts/restore.sh <backup.sql.gz>" >&2
  exit 1
}

echo "Restoring $1 → docker compose service 'postgres' (rag)... Executing in your local Docker daemon."
gunzip -c "$1" | docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U rag -d rag

echo "Restored. Watch /health for 503 recovery: curl http://localhost:8000/health"

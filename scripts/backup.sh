#!/usr/bin/env bash
# Dump the RAG database to a gzip'd SQL archive in ./backups/.
# All tenanted state lives in one Postgres (chunks + embeddings + metadata),
# so a single dump is a consistent DR snapshot: no vector-store sync to fear.
# Restore: scripts/restore.sh backups/rag-<timestamp>.sql.gz
set -euo pipefail

OUT_DIR="${1:-backups}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/rag-$STAMP.sql.gz"

echo "Dumping docker compose service 'postgres' (rag database) → $OUT"
docker compose exec -T postgres pg_dump -U rag --clean --if-exists rag |
  gzip > "$OUT"

LINES=$(gunzip -c "$OUT" | wc -l | tr -d ' ')
echo "Done: $(du -h "$OUT" | cut -f1), $LINES statements."
echo "Verify a restore regularly by replaying with scripts/restore.sh."

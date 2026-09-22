#!/bin/bash
# Idempotent fix-up: re-download any partial/corrupt files with resume.
# .dat must be exactly 120000 bytes (5000 x 12 x int16); .hea must be >100 bytes.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REC="$REPO/data/records"
BASE="https://physionet.org/files/ptb-xl/1.0.3/records500/00000/"
JOBS="${1:-6}"
tmp=$(mktemp)
tail -n +2 "$REPO/data/sample_manifest.csv" | cut -d, -f3 | while IFS= read -r f; do
  name=$(basename "$f")
  sz=$(stat -c%s "$REC/$name.dat" 2>/dev/null || echo 0)
  [ "$sz" -eq 120000 ] || echo "$BASE$name.dat"
  sz=$(stat -c%s "$REC/$name.hea" 2>/dev/null || echo 0)
  [ "$sz" -gt 100 ] || echo "$BASE$name.hea"
done > "$tmp"
total=$(wc -l < "$tmp")
if [ "$total" -eq 0 ]; then echo "all files complete"; rm -f "$tmp"; exit 0; fi
echo "fixing $total files with $JOBS parallel jobs (resume)"
cat "$tmp" | xargs -P "$JOBS" -I{} curl -sS -C - --retry 3 --retry-delay 2 --max-time 300 -O --output-dir "$REC" {}
echo "done"
rm -f "$tmp"

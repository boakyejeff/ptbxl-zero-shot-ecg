#!/bin/bash
# Parallel download of the sampled records. Usage: bash scripts/download_parallel.sh [jobs]
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REC="$REPO/data/records"
BASE="https://physionet.org/files/ptb-xl/1.0.3/records500/00000/"
JOBS="${1:-6}"
mkdir -p "$REC"
# build URL list for missing files
tmp=$(mktemp)
tail -n +2 "$REPO/data/sample_manifest.csv" | cut -d, -f3 | while IFS= read -r f; do
  name=$(basename "$f")
  for ext in .dat .hea; do
    [ -s "$REC/$name$ext" ] || echo "$BASE$name$ext"
  done
done > "$tmp"
total=$(wc -l < "$tmp")
echo "downloading $total files with $JOBS parallel jobs"
cat "$tmp" | xargs -P "$JOBS" -I{} curl -sS --retry 3 --retry-delay 2 --max-time 300 -O --output-dir "$REC" {}
echo "done; files now in $REC: $(ls "$REC" | wc -l)"
rm -f "$tmp"

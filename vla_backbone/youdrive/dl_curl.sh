#!/bin/bash
REPO="$1"; DEST="$2"
mkdir -p "$DEST"
PY=/root/workspace/miniconda3/envs/vqa_gen/bin/python
echo "[list] fetching file tree for $REPO ..."
while true; do
  curl -s "https://huggingface.co/api/models/$REPO/tree/main?recursive=true" -o "$DEST/.tree.json" 2>/dev/null
  if $PY -c "import json,sys; d=json.load(open('$DEST/.tree.json')); assert isinstance(d,list) and len(d)>0" 2>/dev/null; then
    echo "[list] OK"; break
  fi
  echo "[list] retry in 5s ($(date '+%T'))"; sleep 5
done
mapfile -t FILES < <($PY -c "import json;[print(x['path']) for x in json.load(open('$DEST/.tree.json')) if x['type']=='file']")
echo "[list] ${#FILES[@]} files"
for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  for t in $(seq 1 500); do
    code=$(curl -sL -C - "https://huggingface.co/$REPO/resolve/main/$f" -o "$DEST/$f" -w "%{http_code}" 2>/dev/null)
    if [ "$code" = "200" ] || [ "$code" = "206" ] || [ "$code" = "416" ]; then
      echo "[ok] $f ($code)"; break
    fi
    sleep 4
  done
done
echo "ALL_DONE $REPO"

export HF_ENDPOINT=https://hf-mirror.com
HF=/root/workspace/miniconda3/envs/vqa_gen/bin/hf
REPO="$1"; DEST="$2"
for i in $(seq 1 200); do
  echo "[try $i] $(date '+%T') downloading $REPO ..."
  $HF download "$REPO" --local-dir "$DEST" --quiet && { echo "DONE_OK $REPO"; break; }
  echo "[try $i] failed, retry in 10s"; sleep 10
done

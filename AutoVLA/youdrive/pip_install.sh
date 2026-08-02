set -x
export PIP_CACHE_DIR=/mnt/pfs/zhengguantian/autovla/pip_cache
export TMPDIR=/mnt/pfs/zhengguantian/autovla/tmp
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
PIP="/root/workspace/miniconda3/envs/autovla/bin/pip install -i https://pypi.org/simple --timeout 60 --retries 10"
$PIP torch==2.4.0 torchvision==0.19.0
$PIP -r requirements.txt
$PIP -e navsim
$PIP flash-attn==2.7.4.post1 --no-build-isolation || echo "FLASHATTN_FAILED_fallback_sdpa"
/root/workspace/miniconda3/envs/autovla/bin/python -c "from transformers import Qwen2_5_VLForConditionalGeneration; import navsim, torch; print('IMPORT_OK torch', torch.__version__)"
echo "PIP_DONE"

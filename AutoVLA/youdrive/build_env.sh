set -x
export PIP_CACHE_DIR=/mnt/pfs/zhengguantian/autovla/pip_cache
export TMPDIR=/mnt/pfs/zhengguantian/autovla/tmp
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
source /root/workspace/miniconda3/etc/profile.d/conda.sh
# 用官方源(走代理通),绕开挂掉的清华镜像
conda create -n autovla python=3.9 -y --override-channels -c https://repo.anaconda.com/pkgs/main || { echo CONDA_CREATE_FAILED; exit 1; }
PIP=/root/workspace/miniconda3/envs/autovla/bin/pip
$PIP install --upgrade pip
$PIP install torch==2.4.0 torchvision==0.19.0
$PIP install -r requirements.txt
$PIP install -e navsim
$PIP install flash-attn==2.7.4.post1 --no-build-isolation || echo "FLASHATTN_FAILED_fallback_sdpa"
/root/workspace/miniconda3/envs/autovla/bin/python -c "from transformers import Qwen2_5_VLForConditionalGeneration; import navsim; print('IMPORT_OK')"
echo "ENV_BUILD_DONE"

#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

CONFIG=configs/vitb_384_OVRSIS95K.yaml
WEIGHT=pretrained_weights/VIT-B_noise_img_gaussian_txt_gaussian.pth

GPUS=4
GPU_IDS=0,1,2,3

LAMBDAS=("0" "-0.1" "-0.2" "-0.3" "-0.5")

for L in "${LAMBDAS[@]}"; do
  TAG=$(echo "$L" | sed 's/-/m/g; s/\./p/g')

  OUT_DIR=output_eval_vitb_noise_4gpu_segearth_gba_lambda_${TAG}
  LOG_FILE=logs/eval_all_pretrained_4gpu_segearth_gba_lambda_${TAG}.log

  echo "============================================================"
  echo "[START] lambda=${L}"
  echo "[OUT]   ${OUT_DIR}"
  echo "[LOG]   ${LOG_FILE}"
  echo "============================================================"

  PISEG_SEGEARTH_GBA=1 \
  PISEG_SEGEARTH_LAMBDA="${L}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=${GPU_IDS} \
  bash tools/eval_all_pretrained.sh \
    "${CONFIG}" \
    "${GPUS}" \
    "${OUT_DIR}" \
    "${WEIGHT}" \
    > "${LOG_FILE}" 2>&1

  echo "[DONE] lambda=${L}"
done

echo "[ALL DONE] all lambdas finished."
#!/usr/bin/env bash
# Chạy thử toàn bộ pipeline (2 epoch) trên data GIẢ để chắc môi trường/code ổn trước khi train thật.
#   bash scripts/smoke_test.sh [thư_mục_tạm]
set -e
OUT=${1:-/tmp/fake_emotion}
python scripts/make_synthetic_data.py --out "$OUT"
COMMON="data_root=$OUT train.epochs=2 train.batch_size=16 train.warmup_epochs=1 train.num_workers=0 model.pretrained=false logging.out_dir=$OUT/runs logging.log_every_steps=2 logging.param_every_steps=1"
python train.py --config configs/face.yaml  --set $COMMON
python train.py --config configs/audio.yaml --set $COMMON
FACE=$(ls -d "$OUT"/runs/face_resnet18_* | tail -1)
AUDIO=$(ls -d "$OUT"/runs/audio_cnn_* | tail -1)
python fuse_eval.py --face-run "$FACE" --audio-run "$AUDIO" --seeds 3
python scripts/plot_logs.py --run-dir "$FACE"
echo "SMOKE TEST OK"

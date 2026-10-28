#!/bin/bash
set -euo pipefail

if [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  USER_HOME="$HOME"
fi

APP_DIR="$USER_HOME/aries-waybill-demo"
cd "$APP_DIR"

mkdir -p tmp

./build/waybill_demo_kr_cpp live \
  --yolo-model yolo26m \
  --yolo-mode global4 \
  --yolo-clusters 0 \
  --ocr-det-mode global4 \
  --ocr-det-clusters 1 \
  --ocr-rec-mode multi \
  --ocr-rec-lanes 2 \
  --camera-width 1280 \
  --camera-height 720
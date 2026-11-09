# waybill-demo-kr-cpp Skill Guide

Use this guide when implementing or reviewing changes in this repo.

## Source Of Truth

Read these first:

- `/workspace/waybill-demo-kr/yolo_inference/pipeline_mxq_ocr.py`
- `/workspace/waybill-demo-kr/yolo_inference/demo_video_mxq_ocr.py`
- `/workspace/waybill-demo-kr/yolo_inference/visualize.py`
- `/workspace/waybill-demo-kr/yolo_inference/postprocess.py`
- `/workspace/waybill-demo-kr/ppocr_inference/tools/infer/predict_system.py`

For C++ structure reference only:

- `/workspace/waybill-demo-en-cpp/src/`

Do not treat the English repo as the behavior source of truth when it conflicts
with the Korean Python demo.

## Core Contract

This repo is a single-image pipeline plus demo-image video generator.

Keep the main inference path as:

1. one source image into YOLO
2. one best waybill crop out
3. OCR detection on that crop
4. batched OCR recognition over the text crops from that one waybill

Only recognition is batched by design.

## Defaults

- YOLO options: `yolo11n/s/m/l/x`, `yolo26n/s/m/l/x`
- default YOLO model: `yolo26x`
- default YOLO mode: `global8`
- default OCR detection mode: `global8`
- default OCR recognition mode: `multi`
- fixed OCR recognition orientation mode: `parallel-all`
- default OCR det model: `PP-OCRv5_mobile_det.mxq`
- default OCR rec model: `korean_PP-OCRv5_mobile_rec_w.mxq`
- default OCR dict: `ppocrv5_korean_dict.txt`
- default font: `Font-PretendardVariable.ttf`
- validation input directory: `waybill_val_dataset/`

## Rendering Rules

The expected result layout is fixed:

- `Original Image + Waybill Detection`
- `Cropped Waybill + OCR Boxes`
- `Recognized Text`

The recognized-text panel is sensitive:

- preserve OCR sorting order
- preserve the visual correspondence between text boxes and recognized text
- preserve fixed `parallel-all` recognition for rotated and opposite-side
  validation cases
- do not add another image-level rotation correction step for those cases
- avoid generic dashboard rewrites

## Build And Runtime Dependencies

Assume these are required unless the repo changes:

- C++17
- CMake
- OpenCV dev package
- OpenMP
- `qbruntime`

If the build fails, first check:

1. `OpenCV_DIR`
2. linker visibility for `qbruntime`
3. presence of OpenMP runtime

## Review Checklist

- Does the repo still expose both `image` and `video` commands?
- Does the repo expose `batch` for multi-image inference?
- Is default YOLO still `yolo26x`?
- Are YOLO and OCR-det still `global8` by default?
- Is OCR-rec still `multi` by default?
- Is OCR-rec orientation mode fixed to `parallel-all`?
- Is batching still limited to recognition?
- Does validation still run correctly from `waybill_val_dataset`?
- Does the recognized-text panel still preserve ordering and placement style?
- Do README instructions still include system package installation and `qbruntime` notes?

## Minimum Validation

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
./build/waybill_demo_kr_cpp --help
./build/waybill_demo_kr_cpp image --image-path waybill_val_dataset/PXL_20260612_043730575.jpg
./build/waybill_demo_kr_cpp batch --image-dir waybill_val_dataset --output-dir tmp/batch_inference --max-images 8
./build/waybill_demo_kr_cpp video --image-dir waybill_val_dataset --max-images 8
```

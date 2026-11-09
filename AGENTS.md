# Agent Instructions

## Scope

These instructions apply to the whole `aries-waybill-demo` repo.

## Primary Goal

Maintain the Korean C++ demo as a faithful port of `/workspace/waybill-demo-kr`.

The important contract is not generic OCR throughput. It is preserving the
Korean demo behavior and output style.

## Fixed Runtime Contract

Keep this processing order:

1. one input image into YOLO
2. choose the best waybill detection
3. crop one waybill region
4. run OCR detection on that crop
5. run batched OCR recognition on the text crops from that one waybill
6. render the same 3-panel result layout

Do not change this to batch YOLO or batch OCR detection unless the user
explicitly asks for it.

## Execution Defaults

- default YOLO model: `mxq/yolo26x.mxq`
- user must be able to choose `yolo11*` or `yolo26*`
- default OCR det model: `mxq/PP-OCRv5_mobile_det.mxq`
- default OCR rec model: `mxq/korean_PP-OCRv5_mobile_rec_w.mxq`
- default OCR dict: `assets/dict/ppocrv5_korean_dict.txt`
- default font: `fonts/Font-PretendardVariable.ttf`
- default YOLO mode: `global8`
- default OCR detection mode: `global8`
- default OCR recognition mode: `multi`
- fixed OCR recognition orientation mode: `parallel-all`
- validation input set for image/video checks: top-level `waybill_val_dataset/`

## Rendering Contract

Preserve the Korean output layout:

- left panel: original image with waybill detection
- top-right panel: cropped waybill with OCR boxes
- bottom-right panel: recognized text panel

Important rendering rules:

- keep the sorting/order of OCR results stable
- keep the recognized-text placement behavior intentional
- keep fixed `parallel-all` recognition enabled for rotated and opposite-side
  validation cases
- do not add another image-level rotation step
- do not replace the recognized-text panel with a generic legend unless the
  user asks for a different style
- do not regress the relation between OCR box order and displayed text order

## Documentation Rules

When updating docs, keep these items explicit:

- required system packages
- OpenCV dev dependency
- OpenMP dependency
- `qbruntime` dependency
- single-image command
- batch inference command
- validation video command
- default model choices and execution modes
- fixed OCR recognition orientation strategy

## Validation

Preferred validation sequence:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
./build/waybill_demo_kr_cpp --help
```

Optional smoke checks:

```bash
./build/waybill_demo_kr_cpp image --image-path waybill_val_dataset/...
./build/waybill_demo_kr_cpp batch --image-dir waybill_val_dataset --output-dir tmp/batch_inference --max-images 8
./build/waybill_demo_kr_cpp video --image-dir waybill_val_dataset --max-images 8
```

## What To Avoid

- do not reintroduce the English multi-video dashboard assumptions here
- do not change defaults from `yolo26x`, `global8`, `global8`, `multi`
- do not make OCR detection batched across multiple waybills in this repo
- do not silently swap Korean assets for English assets
- do not simplify the recognized-text arrangement if it changes the visible style

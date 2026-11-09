# waybill-demo-kr-cpp

Standalone C++ port of the Korean waybill demo.

This repo follows the Korean runtime contract:

- YOLO takes one image per call and returns the best waybill candidate.
- OCR detection takes one waybill crop per call and returns multiple text boxes.
- OCR recognition runs on the text crops from that one waybill.
- Batched inference is used only inside OCR recognition.

Default execution modes tuned from live camera measurements:

- YOLO: `global4` on cluster `0`
- OCR detection: `global4` on cluster `1`
- OCR recognition: `multi`
- OCR recognition orientation strategy: fixed `parallel-all`
- Default YOLO model: `yolo26m`

## Runtime Assets

Bundled assets:

- YOLO models in `mxq/`
- OCR detection model: `mxq/PP-OCRv5_mobile_det.mxq`
- OCR recognition model: `mxq/korean_PP-OCRv5_mobile_rec_w.mxq`
- OCR dictionary: `assets/dict/ppocrv5_korean_dict.txt`
- font: `fonts/Font-PretendardVariable.ttf`
- validation inputs: `waybill_val_dataset/`

## Package Installation

This project needs:

- a C++17 compiler
- CMake
- OpenCV development headers and libraries
- OpenMP
- the Mobilint MXQ runtime library `qbruntime`

### Ubuntu

Install the standard build dependencies:

```bash
apt update
apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libopencv-dev \
  libomp-dev
```

Install `qbruntime` from the Mobilint APT repository:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc
sudo chmod a+r /etc/apt/keyrings/mblt.asc

printf "%s\n" \
  "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt stable multiverse" | \
  sudo tee /etc/apt/sources.list.d/mobilint.list > /dev/null

sudo apt update
sudo apt install -y mobilint-qb-runtime
dpkg -L mobilint-qb-runtime
```

If `qbruntime` is installed in a nonstandard location instead of the APT package, export its paths before building:

```bash
export CMAKE_PREFIX_PATH=/path/to/qbruntime:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/path/to/qbruntime/lib:$LD_LIBRARY_PATH
```

## Build

### Linux

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

Output binary:

```bash
./build/waybill_demo_kr_cpp
```

If CMake cannot find OpenCV, point it explicitly at your OpenCV installation:

```bash
cmake -S . -B build -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
```

If `qbruntime` is installed in a nonstandard location, pass its package root:

```bash
cmake -S . -B build -DQBRUNTIME_PATH=/path/to/qbruntime
```

You can also run the Linux updater, which installs required system packages,
updates this repository, builds the executable, and installs a desktop shortcut
under `~/.local/share/applications/`:

```bash
chmod +x update.sh run.sh
./update.sh
```

The shortcut launches `./run.sh`, which starts live OpenCV camera display mode.
Live mode is camera-only and searches available OpenCV camera indices when
`--camera-index` is not provided after the user presses Enter. If the camera is
disconnected during live mode, reconnect is attempted only after Enter is
pressed. In live mode, camera capture, YOLO object detection, OCR, and rendering are decoupled with bounded latest-frame
handoff so slow OCR does not directly stall camera/display responsiveness.

### Windows

On Windows, provide OpenCV through `OpenCV_DIR`. The Mobilint QB Runtime package
root is detected automatically from `%USERPROFILE%\qbruntime` (or
`%QBRUNTIME_PATH%` when the environment variable is set). The qbruntime package
root must contain `include/`, `lib/`, and `bin/`.

```bat
cmake -S . -B build ^
  -DOpenCV_DIR=C:\path\to\opencv\build\x64\vc17\lib
cmake --build build --config Release
```

If qbruntime is not under your home folder, pass it explicitly:

```bat
cmake -S . -B build ^
  -DOpenCV_DIR=C:\path\to\opencv\build\x64\vc17\lib ^
  -DQBRUNTIME_PATH=C:\path\to\qbruntime
```

Run from the repository root so the default `mxq/`, dictionary, font, and
validation paths resolve correctly:

```bat
.\build\Release\waybill_demo_kr_cpp.exe --help
.\build\Release\waybill_demo_kr_cpp.exe image ^
  --image-path waybill_val_dataset\PXL_20260612_043730575.jpg ^
  --yolo-model yolo26m
```

CMake copies qbruntime/OpenCV runtime DLLs that are exposed through CMake target
metadata into the executable directory after build. It also searches common
OpenCV locations for `opencv_videoio_ffmpeg*.dll` for video mode support.

## Commands

Show help:

```bash
./build/waybill_demo_kr_cpp --help
```

Single-image inference:

```bash
./build/waybill_demo_kr_cpp image \
  --image-path waybill_val_dataset/PXL_20260612_043730575.jpg \
  --yolo-model yolo26m
```

Validation video from `waybill_val_dataset`:

```bash
./build/waybill_demo_kr_cpp video \
  --image-dir waybill_val_dataset \
  --yolo-model yolo26m \
  --fps 30
```

Live display from an automatically detected camera:

```bash
./build/waybill_demo_kr_cpp live \
  --yolo-model yolo26m \
  --yolo-mode global4 \
  --yolo-clusters 0 \
  --ocr-det-mode global4 \
  --ocr-det-clusters 1 \
  --display-scale 1.0
```

Live display from a specific camera index:

```bash
./build/waybill_demo_kr_cpp live \
  --camera-index 0 \
  --yolo-model yolo26m \
  --yolo-mode global4 \
  --yolo-clusters 0 \
  --ocr-det-mode global4 \
  --ocr-det-clusters 1 \
  --display-scale 1.0
```

Live mode hotkeys:

- `Enter`: retry camera detection or reconnect immediately
- `q` / `Esc`: quit
- `m`: toggle fullscreen

Low-light cameras may lengthen their auto-exposure time and return frames more slowly.
Live mode logs slow camera reads but does not treat them as a disconnect; reconnect
handling is reserved for a closed capture or repeated empty frames. Camera detection
and reconnect attempts run automatically every two seconds, while Enter triggers an
immediate retry.

Live rendering keeps the Korean 3-panel layout. While OCR is pending, the latest
YOLO-only detection state may be displayed. When OCR completes, its result is
rendered with the exact camera frame snapshot and waybill crop that produced that
OCR result, not overlaid onto a newer camera frame.

Multi-image inference from `waybill_val_dataset`:

```bash
./build/waybill_demo_kr_cpp batch \
  --image-dir waybill_val_dataset \
  --output-dir tmp/batch_inference \
  --output-json-path tmp/batch_inference_summary.json
```

Useful options:

- `--yolo-model yolo11s`
- `--yolo-model yolo11x`
- `--yolo-model yolo26m`
- `--yolo-model yolo26x`
- `--yolo-model-path /custom/model.mxq`
- `--ocr-det-model-path /custom/PP-OCRv5_mobile_det.mxq`
- `--ocr-rec-model-path /custom/korean_PP-OCRv5_mobile_rec_w.mxq`
- `--ocr-rec-orientation-mode parallel-all`
- `--output-image-path tmp/result.jpg`
- `--output-dir tmp/batch_inference`
- `--output-video-path tmp/demo.mp4`
- `--output-json-path tmp/result.json`
- `--camera-index 0`
- `--camera-width 1280`
- `--camera-height 720`
- `--camera-fps 30`
- `--fullscreen 1`
- `--display-scale 1.0`
- `--max-frames 30`

## Outputs

The rendered layout keeps the Korean Python demo structure:

- left: `Original Image + Waybill Detection`
- top-right: `Cropped Waybill + OCR Boxes`
- bottom-right: `Recognized Text`

Single-image mode writes:

- rendered image
- JSON summary with latency and recognized texts

Video mode writes:

- MP4 built from `waybill_val_dataset`
- JSON summary with per-frame latency and OCR counts

Batch mode writes:

- one rendered JPG per input image
- one JSON result per input image
- one batch summary JSON

## Notes

- YOLO model choice supports both `yolo11*` and `yolo26*`; default is `yolo26m`.
- The live-tuned default NPU placement is YOLO `global4` on cluster `0`, OCR detection `global4` on cluster `1`, and OCR recognition `multi` with two lanes.
- Live mode is camera-only. If `--camera-index` is omitted, OpenCV camera indices `0..9` are searched automatically every two seconds and the first camera that returns a valid frame is used. Enter triggers an immediate detection or reconnect attempt.
- `run.sh` uses the live-tuned default model/mode placement and explicitly requests `1920x1080` camera input. Override camera width/height only when a target camera needs a specific mode.
- OCR recognition orientation is fixed to `parallel-all`: horizontal text crops are evaluated in both upright and 180-degree reversed directions, and vertical text crops are evaluated in both 90-degree directions with two recognition model instances. `--ocr-rec-orientation-mode` is optional and only accepts `parallel-all`.
- The current code uses OpenCV text drawing. If Hangul rendering is incomplete on your target system, the next step is integrating a real font renderer such as OpenCV FreeType.
- `qbruntime` must already be installed and linkable on the target machine.
- **Latency Optimizations**:
  - Precomputed inverse sigmoid thresholds in the YOLO decoding loops (`DecodeYolo26`, `DecodeBestYolo26`, `DecodeYolo11`, `DecodeBestYolo11`) to bypass millions of `std::exp` (Sigmoid) function calls for grid cells scoring below the target confidence threshold.
  - Fused normalization and standardization passes (`convertTo`, `subtract`, `multiply`) in `OcrRuntime::FillDetectorInput` into a single manually-optimized loop parallelized with OpenMP to improve preprocessing latency.

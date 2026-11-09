#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cctype>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include "ocr_runtime.h"
#include "yolo_runtime.h"

namespace fs = std::filesystem;

namespace waybill {
namespace {

constexpr int kLayoutHeight = 1080;
constexpr int kLayoutWidth = 1920;
constexpr int kHeaderHeight = 108;
constexpr int kSectionLabelHeight = 34;
constexpr int kLayoutOuterMargin = 40;
constexpr int kLayoutGutter = 28;
constexpr int kPanelCornerRadius = 22;
const cv::Scalar kColorCanvasBg(10, 9, 8);
const cv::Scalar kColorPanelBg(26, 24, 22);
const cv::Scalar kColorPanelBgAlt(34, 32, 30);
const cv::Scalar kColorPanelBorder(58, 54, 50);
const cv::Scalar kColorTextPrimary(240, 240, 240);
const cv::Scalar kColorTextSecondary(200, 200, 205);
const cv::Scalar kColorTextMuted(150, 148, 145);
const cv::Scalar kColorBrandPrimary(246, 130, 59);
const cv::Scalar kColorBrandAccent(255, 175, 120);
const cv::Scalar kColorStatusPositive(120, 190, 100);
const cv::Scalar kColorStatusWarn(80, 190, 240);
const cv::Scalar kColorStatusBad(96, 96, 240);
constexpr float kOcrLayoutPaddingXRatio = 0.05f;
constexpr float kOcrLayoutPaddingYRatio = 0.08f;
constexpr float kOcrLayoutProjectionTrimRatio = 0.01f;
constexpr size_t kOcrLayoutOutlierMinItems = 12;
constexpr float kOcrLayoutOutlierNeighborScale = 4.0f;
constexpr float kOcrLayoutOutlierMinNeighborDistance = 180.0f;
constexpr float kOcrLayoutOutlierMinKeepRatio = 0.65f;
constexpr float kOcrLayoutOutlierMaxDropRatio = 0.25f;
// Stable recognized-text canvas ratio; update here if the physical template ratio changes.
constexpr float kCanonicalWaybillAspectRatio = 1.64f;
constexpr int kMaxCameraAutoSearchIndex = 9;
constexpr int kCameraPresenceProbeWidth = 320;
constexpr int kCameraPresenceProbeHeight = 240;
constexpr size_t kMaxFontCodepointCacheEntries = 512;
constexpr size_t kMaxFontMetricsCacheEntries = 1024;
constexpr size_t kMaxFontGlyphCacheEntries = 2048;

struct Config {
    bool video_mode = false;
    bool benchmark_mode = false;
    bool batch_mode = false;
    bool live_mode = false;
    std::string image_path;
    std::string image_dir;
    std::string output_image_path;
    std::string output_video_path;
    std::string output_json_path;
    std::string output_dir;
    std::string yolo_model_name = "yolo26m";
    std::string yolo_model_path;
    std::string ocr_det_model_path;
    std::string ocr_rec_model_path;
    std::string ocr_dict_path;
    std::string font_path;
    float conf_thres = 0.25f;
    float iou_thres = 0.45f;
    float crop_pad_ratio = 0.05f;
    float ocr_drop_score = 0.6f;
    float live_crop_smoothing_alpha = 0.3f;
    bool live_ocr_stabilizer = true;
    int live_ocr_stabilizer_window = 5;
    int live_ocr_stabilizer_min_agreement = 3;
    int live_ocr_stabilizer_freeze_frames = 5;
    float live_ocr_stabilizer_freeze_iou = 0.7f;
    float live_ocr_stabilizer_freeze_fraction = 0.8f;
    int live_ocr_stabilizer_freeze_min_slots = 3;
    int live_ocr_stabilizer_freeze_max_frames = 300;
    float live_ocr_stabilizer_freeze_divergence = 0.3f;
    int live_ocr_stabilizer_detection_miss_grace = 15;
    double fps = 30.0;
    double camera_fps = 30.0;
    double display_scale = 1.0;
    bool live_perf_log = true;
    double live_perf_log_interval = 1.0;
    int max_images = 0;
    int max_frames = 0;
    int shuffle_seed = 26;
    int camera_index = -1;
    int camera_width = 0;
    int camera_height = 0;
    int camera_focus = -1;
    std::string camera_backend = "auto";
    bool fullscreen = true;
    std::string window_name = "Waybill Demo KR Live";
    ExecMode yolo_mode = ExecMode::kGlobal4;
    ExecMode ocr_det_mode = ExecMode::kGlobal4;
    ExecMode ocr_rec_mode = ExecMode::kMulti;
    int ocr_rec_lanes = 2;
    std::vector<int> yolo_clusters = {0};
    std::vector<int> yolo_cores = {0, 1, 2, 3};
    std::vector<int> ocr_det_clusters = {1};
    std::vector<int> ocr_det_cores = {0, 1, 2, 3};
    std::vector<int> ocr_rec_clusters = {0, 1};
    std::vector<int> ocr_rec_cores = {0, 1, 2, 3};
};

struct PipelineResult {
    bool detected = false;
    Detection detection;
    cv::Rect crop_rect;
    cv::Mat crop;
    std::vector<TextBox> boxes;
    std::vector<OcrResult> results;
    YoloTiming yolo_timing;
    OcrTiming ocr_timing;
    double overall_latency_sec = 0.0;
};

struct LiveFramePacket {
    int64_t frame_id = 0;
    cv::Mat frame;
    std::chrono::steady_clock::time_point captured_at;
};

struct LiveDetectionPacket {
    int64_t frame_id = 0;
    cv::Mat frame;
    bool detected = false;
    Detection detection;
    cv::Rect crop_rect;
    cv::Mat crop;
    YoloTiming yolo_timing;
    std::chrono::steady_clock::time_point captured_at;
    std::chrono::steady_clock::time_point yolo_started_at;
    std::chrono::steady_clock::time_point detected_at;
};

struct LivePerfStats {
    std::chrono::steady_clock::time_point window_start;
    int64_t captured_window = 0;
    int64_t yolo_window = 0;
    int64_t ocr_window = 0;
    int64_t render_window = 0;
    double capture_read_ms = 0.0;
    double yolo_wait_ms = 0.0;
    double yolo_preprocess_ms = 0.0;
    double yolo_infer_ms = 0.0;
    double yolo_decode_ms = 0.0;
    double yolo_e2e_ms = 0.0;
    double ocr_wait_ms = 0.0;
    double ocr_det_preprocess_ms = 0.0;
    double ocr_det_infer_ms = 0.0;
    double ocr_det_postprocess_ms = 0.0;
    double ocr_rec_crop_ms = 0.0;
    double ocr_rec_preprocess_ms = 0.0;
    double ocr_rec_infer_ms = 0.0;
    double ocr_rec_postprocess_ms = 0.0;
    double ocr_e2e_ms = 0.0;
    int64_t ocr_rec_box_count_sum = 0;
    int64_t ocr_rec_crop_count_sum = 0;
    int64_t ocr_rec_batch_size_sum = 0;
    int64_t ocr_rec_bucket_width_sum = 0;
    int ocr_rec_box_count_max = 0;
    int ocr_rec_crop_count_max = 0;
    int ocr_rec_batch_size_max = 0;
    int ocr_rec_bucket_width_max = 0;
    double ocr_rec_lane0_infer_ms = 0.0;
    double ocr_rec_lane1_infer_ms = 0.0;
    double ocr_rec_lane0_infer_max_ms = 0.0;
    double ocr_rec_lane1_infer_max_ms = 0.0;
    int64_t ocr_rec_lane0_batch_size_sum = 0;
    int64_t ocr_rec_lane1_batch_size_sum = 0;
    int64_t ocr_rec_lane0_bucket_width_sum = 0;
    int64_t ocr_rec_lane1_bucket_width_sum = 0;
    int ocr_rec_lane0_batch_size_max = 0;
    int ocr_rec_lane1_batch_size_max = 0;
    int ocr_rec_lane0_bucket_width_max = 0;
    int ocr_rec_lane1_bucket_width_max = 0;
    double render_ms = 0.0;
    double display_ms = 0.0;
};

struct LiveRenderState {
    int64_t frame_id = 0;
    cv::Mat frame;
    PipelineResult result;
    bool has_result = false;
};

struct LiveSharedState {
    std::mutex mutex;
    std::condition_variable capture_cv;
    std::condition_variable frame_cv;
    std::condition_variable detection_cv;
    std::condition_variable render_cv;
    bool stop = false;
    bool error = false;
    bool camera_disconnected = false;
    bool camera_retry_requested = false;
    std::string error_message;
    std::optional<LiveFramePacket> latest_capture_frame;
    std::optional<LiveFramePacket> latest_frame;
    std::optional<LiveDetectionPacket> latest_detection;
    LiveRenderState latest_detection_render;
    LiveRenderState latest_ocr_render;
    int64_t captured_frames = 0;
    int64_t yolo_frames = 0;
    int64_t ocr_frames = 0;
    LivePerfStats perf;
};

struct DrawTextItem {
    std::vector<cv::Point2f> box;
    std::vector<cv::Point2f> canonical_box;
    std::string text;
    cv::Point2f canonical_center;
    cv::Rect2f canonical_bounds;
};

struct LayoutFrame {
    cv::Rect2f bounds;
    cv::Point2f x_axis = {1.0f, 0.0f};
    cv::Point2f y_axis = {0.0f, 1.0f};
    bool valid = false;
    bool oriented = false;
};

enum class NormalizedLayoutTransform {
    kIdentity,
    kRotateClockwise90,
    kRotateCounterClockwise90,
    kRotate180,
};

std::string JsonEscape(const std::string& value) {
    std::ostringstream oss;
    for (char ch : value) {
        switch (ch) {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << ch;
            break;
        }
    }
    return oss.str();
}

bool ParseFloat(const std::string& value, float* out) {
    char* end = nullptr;
    *out = std::strtof(value.c_str(), &end);
    return end != nullptr && *end == '\0';
}

bool ParseDouble(const std::string& value, double* out) {
    char* end = nullptr;
    *out = std::strtod(value.c_str(), &end);
    return end != nullptr && *end == '\0';
}

bool ParseInt(const std::string& value, int* out) {
    char* end = nullptr;
    *out = static_cast<int>(std::strtol(value.c_str(), &end, 10));
    return end != nullptr && *end == '\0';
}

bool ParseBool(const std::string& value, bool* out) {
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        *out = true;
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        *out = false;
        return true;
    }
    return false;
}

bool ParseIntList(const std::string& value, std::vector<int>* out) {
    std::vector<int> parsed;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            return false;
        }
        int parsed_item = 0;
        if (!ParseInt(item, &parsed_item)) {
            return false;
        }
        parsed.push_back(parsed_item);
    }
    if (parsed.empty()) {
        return false;
    }
    *out = std::move(parsed);
    return true;
}

bool ParseMode(const std::string& value, ExecMode* mode) {
    if (value == "single") {
        *mode = ExecMode::kSingle;
        return true;
    }
    if (value == "multi") {
        *mode = ExecMode::kMulti;
        return true;
    }
    if (value == "global4") {
        *mode = ExecMode::kGlobal4;
        return true;
    }
    if (value == "global8") {
        *mode = ExecMode::kGlobal8;
        return true;
    }
    return false;
}

bool IsSupportedOcrRecognitionOrientationMode(const std::string& value) {
    return value == "parallel-all";
}

std::string FirstExistingPath(const std::vector<fs::path>& paths) {
    for (const auto& path : paths) {
        std::error_code ec;
        if (fs::exists(path, ec)) {
            return path.string();
        }
    }
    return paths.empty() ? std::string() : paths.front().string();
}

std::string DefaultYoloModelPath(const fs::path& repo_root, const std::string& model_name) {
    return FirstExistingPath({repo_root / "mxq" / (model_name + ".mxq")});
}

std::string DefaultOcrDetModelPath(const fs::path& repo_root) {
    return FirstExistingPath({repo_root / "mxq" / "PP-OCRv5_mobile_det.mxq"});
}

std::string DefaultOcrRecModelPath(const fs::path& repo_root) {
    return FirstExistingPath({repo_root / "mxq" / "korean_PP-OCRv5_mobile_rec_w.mxq"});
}

std::string DefaultOcrDictPath(const fs::path& repo_root) {
    return FirstExistingPath({repo_root / "assets" / "dict" / "ppocrv5_korean_dict.txt",
                              repo_root.parent_path() / "waybill-demo-kr" / "ppocr_inference" /
                                  "ppocr" / "utils" / "dict" / "ppocrv5_korean_dict.txt"});
}

std::string DefaultFontPath(const fs::path& repo_root) {
    return FirstExistingPath({repo_root / "fonts" / "Font-PretendardVariable.ttf"});
}

std::string DefaultImagePath(const fs::path& repo_root) {
    const fs::path demo_images = repo_root / "demo_images";
    std::error_code ec;
    if (!fs::is_directory(demo_images, ec)) {
        return demo_images.string();
    }
    for (const auto& entry : fs::directory_iterator(demo_images)) {
        if (entry.is_regular_file()) {
            return entry.path().string();
        }
    }
    return demo_images.string();
}

std::string StemWithoutExtension(const std::string& path) { return fs::path(path).stem().string(); }

void PrintUsage(const char* argv0, const fs::path& repo_root) {
    std::cout
        << "Usage:\n"
        << "  " << argv0 << " image [options]\n"
        << "  " << argv0 << " batch [options]\n"
        << "  " << argv0 << " video [options]\n"
        << "  " << argv0 << " live [options]\n"
        << "  " << argv0 << " benchmark [options]\n\n"
        << "Common options:\n"
        << "  --yolo-model "
           "<yolo11n|yolo11s|yolo11m|yolo11l|yolo11x|yolo26n|yolo26s|yolo26m|yolo26l|yolo26x>\n"
        << "                               Default: yolo26m\n"
        << "  --yolo-model-path <path>\n"
        << "  --ocr-det-model-path <path>\n"
        << "  --ocr-rec-model-path <path>\n"
        << "  --ocr-dict-path <path>\n"
        << "  --font-path <path>\n"
        << "  --conf-thres <float>         Default: 0.25\n"
        << "  --iou-thres <float>          Default: 0.45\n"
        << "  --crop-pad-ratio <float>     Default: 0.05\n"
        << "  --ocr-drop-score <float>     Default: 0.6\n"
        << "  --yolo-mode <mode>           Default: global4\n"
        << "  --yolo-clusters <ids>        Comma-separated cluster ids. Default: 0\n"
        << "  --yolo-cores <ids>           Comma-separated core ids for single mode. Default: 0,1,2,3\n"
        << "  --ocr-det-mode <mode>        Default: global4\n"
        << "  --ocr-det-clusters <ids>     Comma-separated cluster ids. Default: 1\n"
        << "  --ocr-det-cores <ids>        Comma-separated core ids for single mode. Default: 0,1,2,3\n"
        << "  --ocr-rec-mode <mode>        Default: multi\n"
        << "  --ocr-rec-clusters <ids>     Comma-separated cluster ids. Default: 0,1\n"
        << "  --ocr-rec-cores <ids>        Comma-separated core ids for single mode. Default: 0,1,2,3\n"
        << "  --ocr-rec-lanes <1|2>        Recognition orientation lanes. Default: 2\n"
        << "  --ocr-rec-orientation-mode <parallel-all>\n"
        << "                               Only supported value; enabled by default\n\n"
        << "Image mode:\n"
        << "  --image-path <path>          Default: first file in demo_images\n"
        << "  --output-image-path <path>   Default: tmp/single_image_result.jpg\n"
        << "  --output-json-path <path>    Default: tmp/single_image_result.json\n\n"
        << "Batch mode:\n"
        << "  --image-dir <path>           Default: demo_images\n"
        << "  --output-dir <path>          Default: tmp/batch_inference\n"
        << "  --output-json-path <path>    Default: tmp/batch_inference_summary.json\n"
        << "  --shuffle-seed <int>         Default: 26\n"
        << "  --max-images <int>           Optional limit\n\n"
        << "Video mode:\n"
        << "  --image-dir <path>           Default: demo_images\n"
        << "  --output-video-path <path>   Default: tmp/demo_video_kr_cpp.mp4\n"
        << "  --output-json-path <path>    Default: tmp/demo_video_kr_cpp.json\n"
        << "  --fps <float>                Default: 30.0\n"
        << "  --shuffle-seed <int>         Default: 26\n"
        << "  --max-images <int>           Optional limit\n\n"
        << "Live mode:\n"
        << "  --camera-index <int>         Read frames from a specific camera index\n"
        << "                               Default: auto search cameras 0.."
        << kMaxCameraAutoSearchIndex << "\n"
        << "                               Automatically retries every 2 seconds; Enter retries now\n"
        << "  --camera-width <int>         Default: auto-select max supported resolution\n"
        << "  --camera-height <int>        Default: auto-select max supported resolution\n"
        << "  --camera-fps <float>         Default: 30.0\n"
        << "  --camera-focus <int>         Disable autofocus and set a manual focus value\n"
        << "                               (non-negative integer; camera-specific scale,\n"
        << "                               typically 0=infinity, larger=closer). Default:\n"
        << "                               unset (use autofocus).\n"
        << "  --camera-backend <name>      auto | dshow | v4l2 | any.\n"
        << "                               Default: auto (dshow on Windows,\n"
        << "                               v4l2 on Linux, else any).\n"
        << "  --window-name <string>       Default: Waybill Demo KR Live\n"
        << "  --fullscreen <0|1>           Default: 1\n"
        << "  --display-scale <float>      Default: 1.0\n"
        << "  --live-perf-log <0|1>        Print live latency/FPS logs. Default: 1\n"
        << "  --live-perf-log-interval <float>\n"
        << "                               Live performance log interval seconds. Default: 1.0\n"
        << "  --live-crop-smoothing-alpha <float>\n"
        << "                               EMA weight of the newest YOLO detection when smoothing the\n"
        << "                               waybill crop rect across frames (0 disables). Default: 0.3\n"
        << "  --live-ocr-stabilizer <0|1>  Per-slot temporal voting over recent OCR frames.\n"
        << "                               Default: 1\n"
        << "  --live-ocr-stabilizer-window <int>\n"
        << "                               Ring buffer size (frames) per slot. Default: 5\n"
        << "  --live-ocr-stabilizer-min-agreement <int>\n"
        << "                               Winning text in a slot's window must appear at least\n"
        << "                               this many times before the slot is displayed.\n"
        << "                               Default: 3\n"
        << "  --live-ocr-stabilizer-freeze-frames <int>\n"
        << "                               Consecutive frames of matching majority per slot before\n"
        << "                               the whole reading is frozen as a stable snapshot.\n"
        << "                               Default: 5. Use 0 to disable freeze.\n"
        << "  --live-ocr-stabilizer-freeze-iou <float>\n"
        << "                               Waybill-rect IoU below which a frozen snapshot is\n"
        << "                               invalidated. Default: 0.7\n"
        << "  --live-ocr-stabilizer-freeze-fraction <float>\n"
        << "                               Fraction of visible slots that must have reached\n"
        << "                               freeze_frames of same-majority to enter snapshot.\n"
        << "                               Default: 0.8\n"
        << "  --live-ocr-stabilizer-freeze-min-slots <int>\n"
        << "                               Minimum number of visible stable slots required\n"
        << "                               before entering snapshot. Default: 3\n"
        << "  --live-ocr-stabilizer-freeze-max-frames <int>\n"
        << "                               Max OCR-processed frames a snapshot may stay active\n"
        << "                               before automatic thaw (0 disables TTL). Only counts\n"
        << "                               frames where OCR actually ran, not wall time.\n"
        << "                               Default: 300\n"
        << "  --live-ocr-stabilizer-freeze-divergence <float>\n"
        << "                               Fraction of snapshot slots whose live majority may\n"
        << "                               disagree before snapshot is thawed. Default: 0.3\n"
        << "  --live-ocr-stabilizer-detection-miss-grace <int>\n"
        << "                               Consecutive no-detection OCR frames tolerated\n"
        << "                               before stabilizer state (including snapshot) is\n"
        << "                               reset. Counted in OCR-processed frames, not wall\n"
        << "                               time. Default: 15\n"
        << "  --output-video-path <path>   Optional live recording path\n"
        << "  --max-frames <int>           Optional frame limit\n\n"
        << "Benchmark mode:\n"
        << "  --image-dir <path>           Default: demo_images\n"
        << "  --output-json-path <path>    Default: tmp/benchmark_result.json\n"
        << "  --shuffle-seed <int>         Default: 26\n"
        << "  --max-images <int>           Optional limit\n\n"
        << "Defaults:\n"
        << "  yolo: " << DefaultYoloModelPath(repo_root, "yolo26m") << "\n"
        << "  det:  " << DefaultOcrDetModelPath(repo_root) << "\n"
        << "  rec:  " << DefaultOcrRecModelPath(repo_root) << "\n"
        << "  dict: " << DefaultOcrDictPath(repo_root) << "\n"
        << "  font: " << DefaultFontPath(repo_root) << "\n";
}

bool ParseArgs(int argc, char** argv, const fs::path& repo_root, Config* config,
               std::string* error) {
    if (argc < 2) {
        PrintUsage(argv[0], repo_root);
        return false;
    }
    const std::string command = argv[1];
    if (command == "-h" || command == "--help") {
        PrintUsage(argv[0], repo_root);
        return false;
    }
    if (command != "image" && command != "batch" && command != "video" && command != "live" &&
        command != "benchmark") {
        *error = "first argument must be 'image', 'batch', 'video', 'live', or 'benchmark'";
        return false;
    }
    config->video_mode = command == "video";
    config->live_mode = command == "live";
    config->benchmark_mode = command == "benchmark";
    config->batch_mode = command == "batch";
    config->image_path = DefaultImagePath(repo_root);
    config->image_dir = (repo_root / "demo_images").string();
    config->output_image_path = (repo_root / "tmp" / "single_image_result.jpg").string();
    config->output_video_path =
        config->live_mode ? std::string() : (repo_root / "tmp" / "demo_video_kr_cpp.mp4").string();
    config->output_dir = (repo_root / "tmp" / "batch_inference").string();
    config->output_json_path =
        (repo_root / "tmp" /
          (config->benchmark_mode
              ? "benchmark_result.json"
              : (config->video_mode ? "demo_video_kr_cpp.json"
                                    : (config->batch_mode ? "batch_inference_summary.json"
                                                          : "single_image_result.json"))))
            .string();
    config->ocr_det_model_path = DefaultOcrDetModelPath(repo_root);
    config->ocr_rec_model_path = DefaultOcrRecModelPath(repo_root);
    config->ocr_dict_path = DefaultOcrDictPath(repo_root);
    config->font_path = DefaultFontPath(repo_root);

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                *error = std::string("missing value for ") + name;
                return {};
            }
            return argv[++i];
        };
        if (arg == "--image-path") {
            config->image_path = require_value("--image-path");
        } else if (arg == "--image-dir") {
            config->image_dir = require_value("--image-dir");
        } else if (arg == "--input-video-path") {
            (void)require_value("--input-video-path");
            *error = "live mode is camera-only; --input-video-path is no longer supported";
            return false;
        } else if (arg == "--output-image-path") {
            config->output_image_path = require_value("--output-image-path");
        } else if (arg == "--output-video-path") {
            config->output_video_path = require_value("--output-video-path");
        } else if (arg == "--output-json-path") {
            config->output_json_path = require_value("--output-json-path");
        } else if (arg == "--output-dir") {
            config->output_dir = require_value("--output-dir");
        } else if (arg == "--yolo-model") {
            config->yolo_model_name = require_value("--yolo-model");
        } else if (arg == "--yolo-model-path") {
            config->yolo_model_path = require_value("--yolo-model-path");
        } else if (arg == "--ocr-det-model-path") {
            config->ocr_det_model_path = require_value("--ocr-det-model-path");
        } else if (arg == "--ocr-rec-model-path") {
            config->ocr_rec_model_path = require_value("--ocr-rec-model-path");
        } else if (arg == "--ocr-dict-path") {
            config->ocr_dict_path = require_value("--ocr-dict-path");
        } else if (arg == "--font-path") {
            config->font_path = require_value("--font-path");
        } else if (arg == "--conf-thres") {
            if (!ParseFloat(require_value("--conf-thres"), &config->conf_thres)) {
                *error = "invalid --conf-thres";
                return false;
            }
        } else if (arg == "--iou-thres") {
            if (!ParseFloat(require_value("--iou-thres"), &config->iou_thres)) {
                *error = "invalid --iou-thres";
                return false;
            }
        } else if (arg == "--crop-pad-ratio") {
            if (!ParseFloat(require_value("--crop-pad-ratio"), &config->crop_pad_ratio)) {
                *error = "invalid --crop-pad-ratio";
                return false;
            }
        } else if (arg == "--ocr-drop-score") {
            if (!ParseFloat(require_value("--ocr-drop-score"), &config->ocr_drop_score)) {
                *error = "invalid --ocr-drop-score";
                return false;
            }
        } else if (arg == "--fps") {
            if (!ParseDouble(require_value("--fps"), &config->fps)) {
                *error = "invalid --fps";
                return false;
            }
        } else if (arg == "--camera-fps") {
            if (!ParseDouble(require_value("--camera-fps"), &config->camera_fps)) {
                *error = "invalid --camera-fps";
                return false;
            }
        } else if (arg == "--display-scale") {
            if (!ParseDouble(require_value("--display-scale"), &config->display_scale)) {
                *error = "invalid --display-scale";
                return false;
            }
        } else if (arg == "--live-perf-log") {
            if (!ParseBool(require_value("--live-perf-log"), &config->live_perf_log)) {
                *error = "invalid --live-perf-log";
                return false;
            }
        } else if (arg == "--live-perf-log-interval") {
            if (!ParseDouble(require_value("--live-perf-log-interval"),
                             &config->live_perf_log_interval)) {
                *error = "invalid --live-perf-log-interval";
                return false;
            }
        } else if (arg == "--live-crop-smoothing-alpha") {
            if (!ParseFloat(require_value("--live-crop-smoothing-alpha"),
                            &config->live_crop_smoothing_alpha)) {
                *error = "invalid --live-crop-smoothing-alpha";
                return false;
            }
            if (config->live_crop_smoothing_alpha < 0.0f ||
                config->live_crop_smoothing_alpha > 1.0f) {
                *error = "--live-crop-smoothing-alpha must be in [0, 1]";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer") {
            if (!ParseBool(require_value("--live-ocr-stabilizer"),
                           &config->live_ocr_stabilizer)) {
                *error = "invalid --live-ocr-stabilizer";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-window") {
            if (!ParseInt(require_value("--live-ocr-stabilizer-window"),
                          &config->live_ocr_stabilizer_window) ||
                config->live_ocr_stabilizer_window < 1) {
                *error = "invalid --live-ocr-stabilizer-window";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-min-agreement") {
            if (!ParseInt(require_value("--live-ocr-stabilizer-min-agreement"),
                          &config->live_ocr_stabilizer_min_agreement) ||
                config->live_ocr_stabilizer_min_agreement < 1) {
                *error = "invalid --live-ocr-stabilizer-min-agreement";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-freeze-frames") {
            if (!ParseInt(require_value("--live-ocr-stabilizer-freeze-frames"),
                          &config->live_ocr_stabilizer_freeze_frames) ||
                config->live_ocr_stabilizer_freeze_frames < 0) {
                *error = "invalid --live-ocr-stabilizer-freeze-frames";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-freeze-iou") {
            if (!ParseFloat(require_value("--live-ocr-stabilizer-freeze-iou"),
                            &config->live_ocr_stabilizer_freeze_iou) ||
                config->live_ocr_stabilizer_freeze_iou < 0.0f ||
                config->live_ocr_stabilizer_freeze_iou > 1.0f) {
                *error = "invalid --live-ocr-stabilizer-freeze-iou";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-freeze-fraction") {
            if (!ParseFloat(require_value("--live-ocr-stabilizer-freeze-fraction"),
                            &config->live_ocr_stabilizer_freeze_fraction) ||
                config->live_ocr_stabilizer_freeze_fraction < 0.0f ||
                config->live_ocr_stabilizer_freeze_fraction > 1.0f) {
                *error = "invalid --live-ocr-stabilizer-freeze-fraction";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-freeze-min-slots") {
            if (!ParseInt(require_value("--live-ocr-stabilizer-freeze-min-slots"),
                          &config->live_ocr_stabilizer_freeze_min_slots) ||
                config->live_ocr_stabilizer_freeze_min_slots < 1) {
                *error = "invalid --live-ocr-stabilizer-freeze-min-slots";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-freeze-max-frames") {
            if (!ParseInt(require_value("--live-ocr-stabilizer-freeze-max-frames"),
                          &config->live_ocr_stabilizer_freeze_max_frames) ||
                config->live_ocr_stabilizer_freeze_max_frames < 0) {
                *error = "invalid --live-ocr-stabilizer-freeze-max-frames";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-freeze-divergence") {
            if (!ParseFloat(require_value("--live-ocr-stabilizer-freeze-divergence"),
                            &config->live_ocr_stabilizer_freeze_divergence) ||
                config->live_ocr_stabilizer_freeze_divergence < 0.0f ||
                config->live_ocr_stabilizer_freeze_divergence > 1.0f) {
                *error = "invalid --live-ocr-stabilizer-freeze-divergence";
                return false;
            }
        } else if (arg == "--live-ocr-stabilizer-detection-miss-grace") {
            if (!ParseInt(require_value("--live-ocr-stabilizer-detection-miss-grace"),
                          &config->live_ocr_stabilizer_detection_miss_grace) ||
                config->live_ocr_stabilizer_detection_miss_grace < 0) {
                *error = "invalid --live-ocr-stabilizer-detection-miss-grace";
                return false;
            }
        } else if (arg == "--shuffle-seed") {
            if (!ParseInt(require_value("--shuffle-seed"), &config->shuffle_seed)) {
                *error = "invalid --shuffle-seed";
                return false;
            }
        } else if (arg == "--max-images") {
            if (!ParseInt(require_value("--max-images"), &config->max_images)) {
                *error = "invalid --max-images";
                return false;
            }
        } else if (arg == "--max-frames") {
            if (!ParseInt(require_value("--max-frames"), &config->max_frames)) {
                *error = "invalid --max-frames";
                return false;
            }
        } else if (arg == "--camera-index") {
            if (!ParseInt(require_value("--camera-index"), &config->camera_index)) {
                *error = "invalid --camera-index";
                return false;
            }
        } else if (arg == "--camera-width") {
            if (!ParseInt(require_value("--camera-width"), &config->camera_width)) {
                *error = "invalid --camera-width";
                return false;
            }
        } else if (arg == "--camera-height") {
            if (!ParseInt(require_value("--camera-height"), &config->camera_height)) {
                *error = "invalid --camera-height";
                return false;
            }
        } else if (arg == "--camera-focus") {
            if (!ParseInt(require_value("--camera-focus"), &config->camera_focus)) {
                *error = "invalid --camera-focus";
                return false;
            }
            if (config->camera_focus < 0) {
                *error = "--camera-focus must be non-negative (use no flag for autofocus)";
                return false;
            }
        } else if (arg == "--camera-backend") {
            const std::string value = require_value("--camera-backend");
            if (value != "auto" && value != "dshow" && value != "v4l2" && value != "any") {
                *error = "--camera-backend must be auto, dshow, v4l2, or any";
                return false;
            }
            config->camera_backend = value;
        } else if (arg == "--loop-video") {
            (void)require_value("--loop-video");
            *error = "live mode is camera-only; --loop-video is no longer supported";
            return false;
        } else if (arg == "--fullscreen") {
            if (!ParseBool(require_value("--fullscreen"), &config->fullscreen)) {
                *error = "invalid --fullscreen";
                return false;
            }
        } else if (arg == "--window-name") {
            config->window_name = require_value("--window-name");
        } else if (arg == "--yolo-mode") {
            if (!ParseMode(require_value("--yolo-mode"), &config->yolo_mode)) {
                *error = "invalid --yolo-mode";
                return false;
            }
        } else if (arg == "--yolo-clusters") {
            if (!ParseIntList(require_value("--yolo-clusters"), &config->yolo_clusters)) {
                *error = "invalid --yolo-clusters";
                return false;
            }
        } else if (arg == "--yolo-cores") {
            if (!ParseIntList(require_value("--yolo-cores"), &config->yolo_cores)) {
                *error = "invalid --yolo-cores";
                return false;
            }
        } else if (arg == "--ocr-det-mode") {
            if (!ParseMode(require_value("--ocr-det-mode"), &config->ocr_det_mode)) {
                *error = "invalid --ocr-det-mode";
                return false;
            }
        } else if (arg == "--ocr-det-clusters") {
            if (!ParseIntList(require_value("--ocr-det-clusters"), &config->ocr_det_clusters)) {
                *error = "invalid --ocr-det-clusters";
                return false;
            }
        } else if (arg == "--ocr-det-cores") {
            if (!ParseIntList(require_value("--ocr-det-cores"), &config->ocr_det_cores)) {
                *error = "invalid --ocr-det-cores";
                return false;
            }
        } else if (arg == "--ocr-rec-mode") {
            if (!ParseMode(require_value("--ocr-rec-mode"), &config->ocr_rec_mode)) {
                *error = "invalid --ocr-rec-mode";
                return false;
            }
        } else if (arg == "--ocr-rec-clusters") {
            if (!ParseIntList(require_value("--ocr-rec-clusters"), &config->ocr_rec_clusters)) {
                *error = "invalid --ocr-rec-clusters";
                return false;
            }
        } else if (arg == "--ocr-rec-cores") {
            if (!ParseIntList(require_value("--ocr-rec-cores"), &config->ocr_rec_cores)) {
                *error = "invalid --ocr-rec-cores";
                return false;
            }
        } else if (arg == "--ocr-rec-lanes") {
            if (!ParseInt(require_value("--ocr-rec-lanes"), &config->ocr_rec_lanes) ||
                config->ocr_rec_lanes < 1 || config->ocr_rec_lanes > 2) {
                *error = "invalid --ocr-rec-lanes";
                return false;
            }
        } else if (arg == "--ocr-rec-orientation-mode") {
            const std::string value = require_value("--ocr-rec-orientation-mode");
            if (!IsSupportedOcrRecognitionOrientationMode(value)) {
                *error = "--ocr-rec-orientation-mode only supports parallel-all";
                return false;
            }
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0], repo_root);
            return false;
        } else {
            *error = "unknown argument: " + arg;
            return false;
        }
    }

    if (config->yolo_model_path.empty()) {
        config->yolo_model_path = DefaultYoloModelPath(repo_root, config->yolo_model_name);
    }
    if (config->ocr_rec_lanes < 1 || config->ocr_rec_lanes > 2) {
        *error = "--ocr-rec-lanes must be 1 or 2";
        return false;
    }
    if (config->live_mode) {
        if (config->display_scale <= 0.0) {
            *error = "--display-scale must be positive";
            return false;
        }
        if (config->max_frames < 0) {
            *error = "--max-frames must be non-negative";
            return false;
        }
        if (config->live_perf_log_interval <= 0.0) {
            *error = "--live-perf-log-interval must be positive";
            return false;
        }
    }
    return true;
}

struct DetectionSmoother {
    bool has_prev = false;
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;

    void Reset() { has_prev = false; }

    Detection Apply(const Detection& raw, float alpha) {
        const float new_cx = 0.5f * (raw.x1 + raw.x2);
        const float new_cy = 0.5f * (raw.y1 + raw.y2);
        const float new_w = std::max(1.0f, raw.x2 - raw.x1);
        const float new_h = std::max(1.0f, raw.y2 - raw.y1);

        if (alpha <= 0.0f) {
            cx = new_cx;
            cy = new_cy;
            w = new_w;
            h = new_h;
            has_prev = true;
            return raw;
        }

        if (!has_prev) {
            cx = new_cx;
            cy = new_cy;
            w = new_w;
            h = new_h;
            has_prev = true;
        } else {
            constexpr float kResetCenterRatio = 0.35f;
            constexpr float kResetSizeRatio = 1.5f;
            const float ref = std::max(w, h);
            const float dx = new_cx - cx;
            const float dy = new_cy - cy;
            const float dist2 = dx * dx + dy * dy;
            const float thresh = kResetCenterRatio * ref;
            const bool center_jump = dist2 > thresh * thresh;
            const bool size_jump = (new_w > w * kResetSizeRatio) || (w > new_w * kResetSizeRatio) ||
                                   (new_h > h * kResetSizeRatio) || (h > new_h * kResetSizeRatio);
            if (center_jump || size_jump) {
                cx = new_cx;
                cy = new_cy;
                w = new_w;
                h = new_h;
            } else {
                cx = alpha * new_cx + (1.0f - alpha) * cx;
                cy = alpha * new_cy + (1.0f - alpha) * cy;
                w = alpha * new_w + (1.0f - alpha) * w;
                h = alpha * new_h + (1.0f - alpha) * h;
            }
        }

        Detection smoothed = raw;
        smoothed.x1 = cx - 0.5f * w;
        smoothed.y1 = cy - 0.5f * h;
        smoothed.x2 = cx + 0.5f * w;
        smoothed.y2 = cy + 0.5f * h;
        return smoothed;
    }
};

namespace stab_detail {
inline cv::Point2f BoxCentroid(const std::vector<cv::Point2f>& box) {
    if (box.empty()) {
        return {0.0f, 0.0f};
    }
    cv::Point2f c{0.0f, 0.0f};
    for (const auto& p : box) {
        c.x += p.x;
        c.y += p.y;
    }
    const float n = static_cast<float>(box.size());
    return {c.x / n, c.y / n};
}

inline float BoxMinDim(const std::vector<cv::Point2f>& box) {
    if (box.size() < 2) {
        return 0.0f;
    }
    float minx = box[0].x;
    float maxx = box[0].x;
    float miny = box[0].y;
    float maxy = box[0].y;
    for (const auto& p : box) {
        minx = std::min(minx, p.x);
        maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y);
        maxy = std::max(maxy, p.y);
    }
    return std::min(maxx - minx, maxy - miny);
}

inline std::vector<cv::Point2f> ShiftedBox(const std::vector<cv::Point2f>& box,
                                           cv::Point2f delta) {
    std::vector<cv::Point2f> out;
    out.reserve(box.size());
    for (const auto& p : box) {
        out.emplace_back(p.x + delta.x, p.y + delta.y);
    }
    return out;
}

inline float RectIoU(const cv::Rect& a, const cv::Rect& b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    const int iw = std::max(0, x2 - x1);
    const int ih = std::max(0, y2 - y1);
    const float inter = static_cast<float>(iw) * static_cast<float>(ih);
    const float area_a = static_cast<float>(a.width) * static_cast<float>(a.height);
    const float area_b = static_cast<float>(b.width) * static_cast<float>(b.height);
    const float ua = area_a + area_b - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}
} // namespace stab_detail

struct OcrStabilizer {
    struct TextVote {
        std::string text;
        float score = 0.0f;
        int64_t frame_idx = 0;
    };

    struct Slot {
        std::array<cv::Point2f, 4> box_orig{};
        int box_pts = 0;
        float ema_score = 0.0f;
        std::deque<TextVote> history;
        int64_t last_seen_frame = -1;
        std::string current_text;
        int current_agreement = 0;
        int stable_streak = 0;
    };

    bool enabled = true;
    int window_size = 5;
    int min_agreement = 3;
    int freeze_threshold = 5;
    // Slot-liveness ladder (units = OCR frames processed by Apply, not wall time):
    //   visibility_grace_frames < prune_after_missing < detection_miss_grace.
    // A slot stops emitting after visibility_grace_frames without a match, gets
    // erased after prune_after_missing, and the entire stabilizer resets after
    // detection_miss_grace consecutive no-detection frames.
    int prune_after_missing = 3;
    int visibility_grace_frames = 2;
    float match_center_ratio = 0.5f;
    float box_ema_alpha = 0.5f;
    float snapshot_iou_threshold = 0.7f;
    int snapshot_min_slots = 3;
    float snapshot_freeze_fraction = 0.8f;
    int snapshot_max_frames = 300;
    float snapshot_divergence_threshold = 0.3f;
    int detection_miss_grace = 15;

    std::vector<Slot> slots;
    int64_t frame_counter = 0;

    bool snapshot_active = false;
    cv::Rect snapshot_waybill_rect;
    std::vector<OcrResult> snapshot_results_orig;
    int snapshot_age_frames = 0;
    int consecutive_missed_frames = 0;

    void Reset() {
        slots.clear();
        ClearSnapshot();
        consecutive_missed_frames = 0;
        frame_counter = 0;
    }

    void ClearSnapshot() {
        snapshot_active = false;
        snapshot_results_orig.clear();
        snapshot_age_frames = 0;
    }

    void NoteMissedDetection() {
        ++consecutive_missed_frames;
        if (consecutive_missed_frames > detection_miss_grace) {
            Reset();
        }
    }

    static cv::Point2f SlotCentroid(const Slot& s) {
        if (s.box_pts <= 0) {
            return {0.0f, 0.0f};
        }
        std::vector<cv::Point2f> pts(s.box_orig.begin(),
                                      s.box_orig.begin() + s.box_pts);
        return stab_detail::BoxCentroid(pts);
    }

    void Apply(const std::vector<TextBox>& raw_boxes,
               const std::vector<OcrResult>& raw_results,
               const cv::Rect& crop_rect,
               std::vector<TextBox>* out_boxes,
               std::vector<OcrResult>* out_results) {
        *out_boxes = raw_boxes;
        if (!enabled) {
            *out_results = raw_results;
            consecutive_missed_frames = 0;
            ClearSnapshot();
            return;
        }

        consecutive_missed_frames = 0;
        ++frame_counter;
        const int64_t this_frame = frame_counter;
        const cv::Point2f crop_tl(static_cast<float>(crop_rect.x),
                                  static_cast<float>(crop_rect.y));

        struct Incoming {
            std::vector<cv::Point2f> box_orig;
            std::string text;
            float score = 0.0f;
            cv::Point2f centroid;
            float min_dim = 0.0f;
        };
        std::vector<Incoming> incoming;
        incoming.reserve(raw_results.size());
        for (const auto& r : raw_results) {
            if (r.box.points.empty()) {
                continue;
            }
            Incoming inc;
            inc.box_orig = stab_detail::ShiftedBox(r.box.points, crop_tl);
            inc.text = r.text;
            inc.score = r.score;
            inc.centroid = stab_detail::BoxCentroid(inc.box_orig);
            inc.min_dim = stab_detail::BoxMinDim(inc.box_orig);
            incoming.push_back(std::move(inc));
        }

        std::vector<int> incoming_slot(incoming.size(), -1);
        std::vector<char> slot_used(slots.size(), 0);
        for (size_t i = 0; i < incoming.size(); ++i) {
            float best_d2 = std::numeric_limits<float>::infinity();
            int best_slot = -1;
            const float thresh =
                std::max(1.0f, incoming[i].min_dim) * match_center_ratio;
            const float thresh2 = thresh * thresh;
            for (size_t s = 0; s < slots.size(); ++s) {
                if (slot_used[s]) {
                    continue;
                }
                if (slots[s].box_pts == 0) {
                    continue;
                }
                const cv::Point2f sc = SlotCentroid(slots[s]);
                const float dx = sc.x - incoming[i].centroid.x;
                const float dy = sc.y - incoming[i].centroid.y;
                const float d2 = dx * dx + dy * dy;
                if (d2 < best_d2 && d2 <= thresh2) {
                    best_d2 = d2;
                    best_slot = static_cast<int>(s);
                }
            }
            if (best_slot >= 0) {
                incoming_slot[i] = best_slot;
                slot_used[best_slot] = 1;
            }
        }

        for (size_t i = 0; i < incoming.size(); ++i) {
            Slot* slot = nullptr;
            const int s_idx = incoming_slot[i];
            const int n_pts =
                static_cast<int>(std::min<size_t>(4, incoming[i].box_orig.size()));
            if (s_idx >= 0) {
                slot = &slots[s_idx];
                if (slot->box_pts == n_pts && n_pts > 0) {
                    for (int k = 0; k < n_pts; ++k) {
                        slot->box_orig[k].x =
                            box_ema_alpha * incoming[i].box_orig[k].x +
                            (1.0f - box_ema_alpha) * slot->box_orig[k].x;
                        slot->box_orig[k].y =
                            box_ema_alpha * incoming[i].box_orig[k].y +
                            (1.0f - box_ema_alpha) * slot->box_orig[k].y;
                    }
                } else if (n_pts > 0) {
                    for (int k = 0; k < n_pts; ++k) {
                        slot->box_orig[k] = incoming[i].box_orig[k];
                    }
                    slot->box_pts = n_pts;
                }
            } else {
                Slot new_slot;
                for (int k = 0; k < n_pts; ++k) {
                    new_slot.box_orig[k] = incoming[i].box_orig[k];
                }
                new_slot.box_pts = n_pts;
                slots.push_back(std::move(new_slot));
                slot = &slots.back();
            }
            slot->last_seen_frame = this_frame;
            slot->ema_score = 0.5f * incoming[i].score + 0.5f * slot->ema_score;
            slot->history.push_back({incoming[i].text, incoming[i].score, this_frame});
            while (static_cast<int>(slot->history.size()) > window_size) {
                slot->history.pop_front();
            }
        }

        for (auto it = slots.begin(); it != slots.end();) {
            while (!it->history.empty() &&
                   it->history.front().frame_idx < this_frame - window_size) {
                it->history.pop_front();
            }
            if (this_frame - it->last_seen_frame > prune_after_missing ||
                it->history.empty()) {
                it = slots.erase(it);
                continue;
            }
            std::map<std::string, float> scores;
            std::map<std::string, int> counts;
            for (const auto& v : it->history) {
                scores[v.text] += v.score;
                ++counts[v.text];
            }
            std::string best_text;
            float best_score = -1.0f;
            int best_count = 0;
            for (const auto& kv : scores) {
                const int this_count = counts[kv.first];
                const bool better = kv.second > best_score ||
                                    (kv.second == best_score && this_count > best_count);
                if (better) {
                    best_score = kv.second;
                    best_count = this_count;
                    best_text = kv.first;
                }
            }
            if (!best_text.empty()) {
                it->current_agreement = best_count;
                const bool same_text = (best_text == it->current_text);
                it->current_text = best_text;
                if (best_count >= min_agreement) {
                    it->stable_streak = same_text ? it->stable_streak + 1 : 1;
                } else {
                    it->stable_streak = 0;
                }
            } else {
                it->current_agreement = 0;
                it->stable_streak = 0;
            }
            ++it;
        }

        if (snapshot_active) {
            ++snapshot_age_frames;
            const float iou = stab_detail::RectIoU(snapshot_waybill_rect, crop_rect);
            bool thaw = false;
            const char* thaw_reason = nullptr;
            if (iou < snapshot_iou_threshold) {
                thaw = true;
                thaw_reason = "iou";
            } else if (snapshot_max_frames > 0 && snapshot_age_frames > snapshot_max_frames) {
                thaw = true;
                thaw_reason = "ttl";
            } else if (!snapshot_results_orig.empty() &&
                       snapshot_divergence_threshold < 1.0f) {
                int matched = 0;
                int disagreed = 0;
                for (const auto& snap : snapshot_results_orig) {
                    if (snap.box.points.empty()) {
                        continue;
                    }
                    const cv::Point2f snap_centroid =
                        stab_detail::BoxCentroid(snap.box.points);
                    const float snap_min_dim =
                        stab_detail::BoxMinDim(snap.box.points);
                    const float thresh =
                        std::max(1.0f, snap_min_dim) * match_center_ratio;
                    const float thresh2 = thresh * thresh;
                    float best_d2 = std::numeric_limits<float>::infinity();
                    const Slot* best_slot = nullptr;
                    for (const auto& s : slots) {
                        if (this_frame - s.last_seen_frame > visibility_grace_frames) {
                            continue;
                        }
                        if (s.current_agreement < min_agreement) {
                            continue;
                        }
                        const cv::Point2f sc = SlotCentroid(s);
                        const float dx = sc.x - snap_centroid.x;
                        const float dy = sc.y - snap_centroid.y;
                        const float d2 = dx * dx + dy * dy;
                        if (d2 < best_d2 && d2 <= thresh2) {
                            best_d2 = d2;
                            best_slot = &s;
                        }
                    }
                    if (best_slot != nullptr) {
                        ++matched;
                        if (best_slot->current_text != snap.text) {
                            ++disagreed;
                        }
                    }
                }
                if (matched > 0) {
                    const float frac = static_cast<float>(disagreed) /
                                       static_cast<float>(matched);
                    if (frac > snapshot_divergence_threshold) {
                        thaw = true;
                        thaw_reason = "divergence";
                    }
                }
            }
            if (thaw) {
                std::cout << "OCR stabilizer snapshot thawed (" << thaw_reason << ")\n";
                ClearSnapshot();
            }
        }

        out_results->clear();
        const cv::Point2f neg_tl(-crop_tl.x, -crop_tl.y);
        if (snapshot_active) {
            out_boxes->clear();
            for (const auto& r : snapshot_results_orig) {
                OcrResult rr = r;
                rr.box.points = stab_detail::ShiftedBox(r.box.points, neg_tl);
                TextBox tb;
                tb.points = rr.box.points;
                tb.score = rr.box.score;
                out_boxes->push_back(std::move(tb));
                out_results->push_back(std::move(rr));
            }
            return;
        }

        std::vector<OcrResult> emitted_results_orig;
        std::vector<TextBox> all_slot_boxes_orig;
        int visible = 0;
        int stable = 0;
        for (const auto& s : slots) {
            if (this_frame - s.last_seen_frame > visibility_grace_frames) {
                continue;
            }
            if (s.box_pts <= 0) {
                continue;
            }
            TextBox slot_box;
            slot_box.points.assign(s.box_orig.begin(), s.box_orig.begin() + s.box_pts);
            slot_box.score = s.ema_score;
            all_slot_boxes_orig.push_back(slot_box);

            if (s.current_text.empty() || s.current_agreement < min_agreement) {
                continue;
            }
            ++visible;
            if (freeze_threshold > 0 && s.stable_streak >= freeze_threshold) {
                ++stable;
            }
            OcrResult rec;
            rec.box = std::move(slot_box);
            rec.text = s.current_text;
            rec.score = s.ema_score;
            emitted_results_orig.push_back(std::move(rec));
        }

        if (freeze_threshold > 0 && visible >= snapshot_min_slots &&
            static_cast<float>(stable) / static_cast<float>(visible) >=
                snapshot_freeze_fraction) {
            snapshot_active = true;
            snapshot_age_frames = 0;
            snapshot_results_orig = emitted_results_orig;
            snapshot_waybill_rect = crop_rect;
            std::cout << "OCR stabilizer snapshot activated (" << visible
                      << " slots, " << stable << " stable)\n";
        }

        out_boxes->clear();
        for (auto& tb : all_slot_boxes_orig) {
            tb.points = stab_detail::ShiftedBox(tb.points, neg_tl);
            out_boxes->push_back(std::move(tb));
        }
        for (auto& r : emitted_results_orig) {
            r.box.points = stab_detail::ShiftedBox(r.box.points, neg_tl);
            out_results->push_back(std::move(r));
        }
    }
};

cv::Rect ExpandAndClampRect(const Detection& det, const cv::Size& size, float pad_ratio) {
    const float width = std::max(1.0f, det.x2 - det.x1);
    const float height = std::max(1.0f, det.y2 - det.y1);
    const float pad_x = width * pad_ratio;
    const float pad_y = height * pad_ratio;
    const int x1 = std::max(0, static_cast<int>(std::floor(det.x1 - pad_x)));
    const int y1 = std::max(0, static_cast<int>(std::floor(det.y1 - pad_y)));
    const int x2 = std::min(size.width, static_cast<int>(std::ceil(det.x2 + pad_x)));
    const int y2 = std::min(size.height, static_cast<int>(std::ceil(det.y2 + pad_y)));
    return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

class FontRenderer {
  public:
    struct TextMetrics {
        int width = 0;
        int height = 0;
        int baseline = 0;
        int min_y = 0;
        int max_y = 0;
    };

    bool Initialize(const std::string& font_path, std::string* error) {
        std::ifstream file(font_path, std::ios::binary);
        if (!file.is_open()) {
            *error = "failed to open font file: " + font_path;
            return false;
        }
        font_buffer_ = std::vector<unsigned char>(std::istreambuf_iterator<char>(file),
                                                  std::istreambuf_iterator<char>());
        if (font_buffer_.empty()) {
            *error = "font file is empty: " + font_path;
            return false;
        }
        if (stbtt_InitFont(&font_, font_buffer_.data(), 0) == 0) {
            *error = "failed to initialize truetype font: " + font_path;
            return false;
        }
        initialized_ = true;
        return true;
    }

    void Draw(cv::Mat* image, const std::string& text, cv::Point origin, int pixel_height,
              cv::Scalar color, int thickness = -1) const {
        (void)thickness;
        if (!initialized_) {
            const double font_scale = std::max(0.4, pixel_height / 32.0);
            cv::putText(*image, text, {origin.x, origin.y + pixel_height}, cv::FONT_HERSHEY_SIMPLEX,
                        font_scale, color, 1, cv::LINE_AA);
            return;
        }
        RenderText(image, text, origin, pixel_height, color);
    }

    TextMetrics MeasureText(const std::string& text, int pixel_height) const {
        const TextCacheKey key{text, pixel_height};
        auto cached = metrics_cache_.find(key);
        if (cached != metrics_cache_.end()) {
            return cached->second;
        }

        TextMetrics metrics;
        if (!initialized_ || text.empty()) {
            return metrics;
        }
        const std::vector<uint32_t>& codepoints = GetCodepoints(text);
        if (codepoints.empty()) {
            return metrics;
        }
        const float scale = stbtt_ScaleForPixelHeight(&font_, static_cast<float>(pixel_height));
        int ascent = 0;
        int descent = 0;
        int line_gap = 0;
        stbtt_GetFontVMetrics(&font_, &ascent, &descent, &line_gap);
        metrics.baseline = static_cast<int>(std::round(ascent * scale));
        metrics.height =
            std::max(1, static_cast<int>(std::round((ascent - descent + line_gap) * scale)));

        int pen_x = 0;
        int min_x = std::numeric_limits<int>::max();
        int max_x = std::numeric_limits<int>::min();
        int min_y = std::numeric_limits<int>::max();
        int max_y = std::numeric_limits<int>::min();
        int prev = 0;
        bool has_bounds = false;
        for (uint32_t codepoint : codepoints) {
            if (prev != 0) {
                pen_x += static_cast<int>(
                    std::round(scale * stbtt_GetCodepointKernAdvance(&font_, prev, codepoint)));
            }
            int ax = 0;
            int lsb = 0;
            stbtt_GetCodepointHMetrics(&font_, codepoint, &ax, &lsb);
            int x0 = 0;
            int y0 = 0;
            int x1 = 0;
            int y1 = 0;
            stbtt_GetCodepointBitmapBox(&font_, codepoint, scale, scale, &x0, &y0, &x1, &y1);
            if (x1 > x0 && y1 > y0) {
                min_x = std::min(min_x, pen_x + x0);
                max_x = std::max(max_x, pen_x + x1);
                min_y = std::min(min_y, y0);
                max_y = std::max(max_y, y1);
                has_bounds = true;
            }
            pen_x += static_cast<int>(std::round(ax * scale));
            prev = static_cast<int>(codepoint);
        }
        if (has_bounds) {
            metrics.width = std::max(1, max_x - min_x);
            metrics.height = max_y - min_y;
            metrics.min_y = min_y;
            metrics.max_y = max_y;
        } else {
            metrics.width = std::max(1, pen_x);
            metrics.min_y = -metrics.baseline;
            metrics.max_y = 0;
        }
        if (metrics_cache_.size() >= kMaxFontMetricsCacheEntries) {
            metrics_cache_.clear();
        }
        return metrics_cache_.emplace(key, metrics).first->second;
    }

    int PickFontSizeForBounds(const std::string& text, int max_width, int max_height,
                              int min_size = 8, int max_size = 96) const {
        if (!initialized_ || text.empty()) {
            return std::max(min_size, std::min(max_height, max_size));
        }
        int lo = min_size;
        int hi = std::max(min_size, max_size);
        int best = min_size;
        while (lo <= hi) {
            const int mid = (lo + hi) / 2;
            const TextMetrics metrics = MeasureText(text, mid);
            if (metrics.width <= max_width && metrics.height <= max_height) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return best;
    }

  private:
    struct TextCacheKey {
        std::string text;
        int pixel_height;
        bool operator<(const TextCacheKey& other) const {
            return std::tie(text, pixel_height) < std::tie(other.text, other.pixel_height);
        }
    };

    struct GlyphCacheKey {
        uint32_t codepoint;
        int pixel_height;
        bool operator<(const GlyphCacheKey& other) const {
            return std::tie(codepoint, pixel_height) <
                   std::tie(other.codepoint, other.pixel_height);
        }
    };

    struct CachedGlyph {
        std::vector<unsigned char> bitmap;
        int width = 0;
        int height = 0;
        int xoff = 0;
        int yoff = 0;
    };

    static std::vector<uint32_t> DecodeUtf8(const std::string& text) {
        std::vector<uint32_t> codepoints;
        for (size_t i = 0; i < text.size();) {
            const unsigned char ch = static_cast<unsigned char>(text[i]);
            if (ch < 0x80) {
                codepoints.push_back(ch);
                ++i;
                continue;
            }
            uint32_t codepoint = 0;
            size_t advance = 0;
            if ((ch & 0xE0) == 0xC0 && i + 1 < text.size()) {
                codepoint = ((ch & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
                advance = 2;
            } else if ((ch & 0xF0) == 0xE0 && i + 2 < text.size()) {
                codepoint = ((ch & 0x0F) << 12) |
                            ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                            (static_cast<unsigned char>(text[i + 2]) & 0x3F);
                advance = 3;
            } else if ((ch & 0xF8) == 0xF0 && i + 3 < text.size()) {
                codepoint = ((ch & 0x07) << 18) |
                            ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                            ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                            (static_cast<unsigned char>(text[i + 3]) & 0x3F);
                advance = 4;
            } else {
                codepoints.push_back('?');
                ++i;
                continue;
            }
            codepoints.push_back(codepoint);
            i += advance;
        }
        return codepoints;
    }

    const std::vector<uint32_t>& GetCodepoints(const std::string& text) const {
        auto it = codepoint_cache_.find(text);
        if (it != codepoint_cache_.end()) {
            return it->second;
        }
        if (codepoint_cache_.size() >= kMaxFontCodepointCacheEntries) {
            codepoint_cache_.clear();
        }
        return codepoint_cache_.emplace(text, DecodeUtf8(text)).first->second;
    }

    const CachedGlyph& GetCachedGlyph(uint32_t codepoint, int pixel_height) const {
        const GlyphCacheKey key{codepoint, pixel_height};
        auto it = glyph_cache_.find(key);
        if (it != glyph_cache_.end()) {
            return it->second;
        }

        const float scale = stbtt_ScaleForPixelHeight(&font_, static_cast<float>(pixel_height));
        CachedGlyph glyph;
        unsigned char* bitmap_raw =
            stbtt_GetCodepointBitmap(&font_, scale, scale, static_cast<int>(codepoint),
                                     &glyph.width, &glyph.height, &glyph.xoff, &glyph.yoff);
        if (bitmap_raw != nullptr) {
            glyph.bitmap.assign(bitmap_raw, bitmap_raw + (glyph.width * glyph.height));
            stbtt_FreeBitmap(bitmap_raw, nullptr);
        }
        if (glyph_cache_.size() >= kMaxFontGlyphCacheEntries) {
            glyph_cache_.clear();
        }
        return glyph_cache_.emplace(key, std::move(glyph)).first->second;
    }

    void RenderText(cv::Mat* image, const std::string& text, cv::Point origin, int pixel_height,
                    cv::Scalar color) const {
        const std::vector<uint32_t>& codepoints = GetCodepoints(text);
        if (codepoints.empty()) {
            return;
        }
        const float scale = stbtt_ScaleForPixelHeight(&font_, static_cast<float>(pixel_height));
        int ascent = 0;
        int descent = 0;
        int line_gap = 0;
        stbtt_GetFontVMetrics(&font_, &ascent, &descent, &line_gap);
        int pen_x = origin.x;
        const int baseline_y = origin.y + static_cast<int>(std::round(ascent * scale));
        int prev = 0;
        for (uint32_t codepoint : codepoints) {
            if (prev != 0) {
                pen_x += static_cast<int>(
                    std::round(scale * stbtt_GetCodepointKernAdvance(&font_, prev, codepoint)));
            }
            const CachedGlyph& glyph = GetCachedGlyph(codepoint, pixel_height);
            if (!glyph.bitmap.empty()) {
                const int glyph_x = pen_x + glyph.xoff;
                const int glyph_y = baseline_y + glyph.yoff;
                const int dst_x0 = std::max(0, glyph_x);
                const int dst_y0 = std::max(0, glyph_y);
                const int dst_x1 = std::min(image->cols, glyph_x + glyph.width);
                const int dst_y1 = std::min(image->rows, glyph_y + glyph.height);
                for (int y = dst_y0; y < dst_y1; ++y) {
                    cv::Vec3b* row = image->ptr<cv::Vec3b>(y);
                    const int gy = y - glyph_y;
                    for (int x = dst_x0; x < dst_x1; ++x) {
                        const int gx = x - glyph_x;
                        const unsigned char alpha = glyph.bitmap[gy * glyph.width + gx];
                        if (alpha > 0) {
                            cv::Vec3b& dst = row[x];
                            const float a = alpha / 255.0f;
                            for (int c = 0; c < 3; ++c) {
                                dst[c] = static_cast<unsigned char>(
                                    std::round(dst[c] * (1.0f - a) + color[c] * a));
                            }
                        }
                    }
                }
            }
            int advance = 0;
            int lsb = 0;
            stbtt_GetCodepointHMetrics(&font_, static_cast<int>(codepoint), &advance, &lsb);
            pen_x += static_cast<int>(std::round(advance * scale));
            prev = static_cast<int>(codepoint);
        }
    }

    std::vector<unsigned char> font_buffer_;
    stbtt_fontinfo font_{};
    bool initialized_ = false;
    mutable std::map<std::string, std::vector<uint32_t>> codepoint_cache_;
    mutable std::map<TextCacheKey, TextMetrics> metrics_cache_;
    mutable std::map<GlyphCacheKey, CachedGlyph> glyph_cache_;
};

void DrawRoundedRect(cv::Mat* canvas, cv::Rect rect, int radius, cv::Scalar fill,
                     cv::Scalar border, int border_thickness) {
    radius = std::max(0, std::min({radius, rect.width / 2, rect.height / 2}));
    if (radius == 0) {
        cv::rectangle(*canvas, rect, fill, cv::FILLED, cv::LINE_AA);
        if (border_thickness > 0) {
            cv::rectangle(*canvas, rect, border, border_thickness, cv::LINE_AA);
        }
        return;
    }
    const cv::Rect center_h(rect.x + radius, rect.y, rect.width - 2 * radius, rect.height);
    const cv::Rect center_v(rect.x, rect.y + radius, rect.width, rect.height - 2 * radius);
    cv::rectangle(*canvas, center_h, fill, cv::FILLED, cv::LINE_AA);
    cv::rectangle(*canvas, center_v, fill, cv::FILLED, cv::LINE_AA);
    cv::circle(*canvas, {rect.x + radius, rect.y + radius}, radius, fill, cv::FILLED, cv::LINE_AA);
    cv::circle(*canvas, {rect.x + rect.width - radius - 1, rect.y + radius}, radius, fill,
               cv::FILLED, cv::LINE_AA);
    cv::circle(*canvas, {rect.x + radius, rect.y + rect.height - radius - 1}, radius, fill,
               cv::FILLED, cv::LINE_AA);
    cv::circle(*canvas, {rect.x + rect.width - radius - 1, rect.y + rect.height - radius - 1},
               radius, fill, cv::FILLED, cv::LINE_AA);
    if (border_thickness <= 0) {
        return;
    }
    const int x1 = rect.x;
    const int y1 = rect.y;
    const int x2 = rect.x + rect.width - 1;
    const int y2 = rect.y + rect.height - 1;
    cv::line(*canvas, {x1 + radius, y1}, {x2 - radius, y1}, border, border_thickness, cv::LINE_AA);
    cv::line(*canvas, {x1 + radius, y2}, {x2 - radius, y2}, border, border_thickness, cv::LINE_AA);
    cv::line(*canvas, {x1, y1 + radius}, {x1, y2 - radius}, border, border_thickness, cv::LINE_AA);
    cv::line(*canvas, {x2, y1 + radius}, {x2, y2 - radius}, border, border_thickness, cv::LINE_AA);
    cv::ellipse(*canvas, {x1 + radius, y1 + radius}, {radius, radius}, 180.0, 0.0, 90.0, border,
                border_thickness, cv::LINE_AA);
    cv::ellipse(*canvas, {x2 - radius, y1 + radius}, {radius, radius}, 270.0, 0.0, 90.0, border,
                border_thickness, cv::LINE_AA);
    cv::ellipse(*canvas, {x1 + radius, y2 - radius}, {radius, radius}, 90.0, 0.0, 90.0, border,
                border_thickness, cv::LINE_AA);
    cv::ellipse(*canvas, {x2 - radius, y2 - radius}, {radius, radius}, 0.0, 0.0, 90.0, border,
                border_thickness, cv::LINE_AA);
}

const cv::Mat& GetBrandLogo() {
    static const cv::Mat logo = [] {
        cv::Mat raw = cv::imread("aries-waybill-demo.png", cv::IMREAD_UNCHANGED);
        return raw;
    }();
    return logo;
}

void BlitRgbaOverBgr(cv::Mat* dst_bgr, const cv::Mat& src_rgba, cv::Point origin) {
    if (src_rgba.empty()) {
        return;
    }
    const int channels = src_rgba.channels();
    const int x0 = std::max(0, origin.x);
    const int y0 = std::max(0, origin.y);
    const int x1 = std::min(dst_bgr->cols, origin.x + src_rgba.cols);
    const int y1 = std::min(dst_bgr->rows, origin.y + src_rgba.rows);
    for (int y = y0; y < y1; ++y) {
        auto* dst_row = dst_bgr->ptr<cv::Vec3b>(y);
        const auto* src_row = src_rgba.ptr<uint8_t>(y - origin.y);
        for (int x = x0; x < x1; ++x) {
            const int src_x = x - origin.x;
            const uint8_t* src_px = src_row + src_x * channels;
            const uint8_t alpha = channels == 4 ? src_px[3] : 255;
            if (alpha == 0) {
                continue;
            }
            const float a = alpha / 255.0f;
            for (int c = 0; c < 3; ++c) {
                dst_row[x][c] = static_cast<uint8_t>(std::round(
                    src_px[c] * a + dst_row[x][c] * (1.0f - a)));
            }
        }
    }
}

void DrawHeader(cv::Mat* canvas, const FontRenderer& font_renderer) {
    constexpr int kLogoSize = 56;
    constexpr int kTitleFontSize = 52;
    constexpr int kSubtitleFontSize = 20;
    const int logo_x = kLayoutOuterMargin;
    const int logo_y = kLayoutOuterMargin;
    const cv::Mat& logo = GetBrandLogo();
    if (!logo.empty()) {
        cv::Mat logo_scaled;
        cv::resize(logo, logo_scaled, cv::Size(kLogoSize, kLogoSize), 0.0, 0.0, cv::INTER_AREA);
        BlitRgbaOverBgr(canvas, logo_scaled, {logo_x, logo_y});
    }

    const std::string title = "ARIES Waybill OCR";
    const FontRenderer::TextMetrics title_metrics =
        font_renderer.MeasureText(title, kTitleFontSize);
    const int title_x = logo_x + kLogoSize + 20;
    const int title_y = logo_y + (kLogoSize - title_metrics.height) / 2;
    font_renderer.Draw(canvas, title, {title_x, title_y}, kTitleFontSize, kColorTextPrimary);

    const std::string subtitle = "Mobilint ARIES NPU";
    const FontRenderer::TextMetrics subtitle_metrics =
        font_renderer.MeasureText(subtitle, kSubtitleFontSize);
    const int subtitle_x = canvas->cols - kLayoutOuterMargin - subtitle_metrics.width;
    const int subtitle_y = logo_y + (kLogoSize - subtitle_metrics.height) / 2;
    font_renderer.Draw(canvas, subtitle, {subtitle_x, subtitle_y}, kSubtitleFontSize,
                       kColorBrandPrimary);
}

struct PanelBadge {
    std::string text;
    cv::Scalar background;
    cv::Scalar foreground = cv::Scalar(255, 255, 255);
};

void DrawPanelBadge(cv::Mat* canvas, int right_x, int center_y, const PanelBadge& badge,
                    const FontRenderer& font_renderer) {
    constexpr int kBadgeFontSize = 16;
    constexpr int kBadgePadX = 14;
    constexpr int kBadgePadY = 7;
    const FontRenderer::TextMetrics metrics =
        font_renderer.MeasureText(badge.text, kBadgeFontSize);
    const int badge_w = metrics.width + kBadgePadX * 2;
    const int badge_h = metrics.height + kBadgePadY * 2;
    const cv::Rect badge_rect(right_x - badge_w, center_y - badge_h / 2, badge_w, badge_h);
    DrawRoundedRect(canvas, badge_rect, badge_h / 2, badge.background, badge.background, 0);
    font_renderer.Draw(canvas, badge.text,
                       {badge_rect.x + kBadgePadX, badge_rect.y + kBadgePadY - 2},
                       kBadgeFontSize, badge.foreground);
}

void DrawSectionLabel(cv::Mat* canvas, int x, int y, const std::string& label,
                      const FontRenderer& font_renderer, bool accent = false,
                      const std::optional<PanelBadge>& badge = std::nullopt, int panel_width = 0) {
    constexpr int kLabelFontSize = 22;
    const cv::Scalar color = accent ? kColorBrandPrimary : kColorTextMuted;
    font_renderer.Draw(canvas, label, {x, y}, kLabelFontSize, color);
    if (badge.has_value() && panel_width > 0) {
        const FontRenderer::TextMetrics label_metrics =
            font_renderer.MeasureText(label, kLabelFontSize);
        DrawPanelBadge(canvas, x + panel_width, y + label_metrics.height / 2, *badge,
                       font_renderer);
    }
}

void DrawPanel(cv::Mat* canvas, const cv::Mat& image, int x, int y, int width, int height,
               const FontRenderer& font_renderer, bool focused = false) {
    const cv::Rect panel_rect(x, y, width, height);
    const cv::Scalar border = focused ? kColorBrandPrimary : kColorPanelBorder;
    const int border_thickness = focused ? 2 : 1;
    DrawRoundedRect(canvas, panel_rect, kPanelCornerRadius, kColorPanelBg, border,
                    border_thickness);

    const int inner_margin = 22;
    const int target_w = width - inner_margin * 2;
    const int target_h = height - inner_margin * 2;
    const double scale = std::min(target_w / static_cast<double>(image.cols),
                                  target_h / static_cast<double>(image.rows));
    const int fitted_w = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int fitted_h = std::max(1, static_cast<int>(std::round(image.rows * scale)));
    const int paste_x = x + (width - fitted_w) / 2;
    const int paste_y = y + (height - fitted_h) / 2;
    cv::Mat dst = (*canvas)(cv::Rect(paste_x, paste_y, fitted_w, fitted_h));
    cv::resize(image, dst, cv::Size(fitted_w, fitted_h), 0.0, 0.0, cv::INTER_LINEAR);
}

cv::Scalar LatencyStatusColor(double ms, double warn_ms, double bad_ms) {
    if (ms >= bad_ms) return kColorStatusBad;
    if (ms >= warn_ms) return kColorStatusWarn;
    return kColorStatusPositive;
}

void DrawPerformanceDashboard(cv::Mat* canvas, cv::Rect rect, const PipelineResult& result,
                              const FontRenderer& font_renderer) {
    DrawRoundedRect(canvas, rect, kPanelCornerRadius, kColorPanelBgAlt, kColorPanelBorder, 1);

    const double yolo_ms = (result.yolo_timing.preprocess_sec + result.yolo_timing.infer_sec +
                            result.yolo_timing.decode_sec) *
                           1000.0;
    const double ocr_det_ms =
        (result.ocr_timing.det_preprocess_sec + result.ocr_timing.det_infer_sec +
         result.ocr_timing.det_postprocess_sec) *
        1000.0;
    const double ocr_rec_ms =
        (result.ocr_timing.rec_crop_sec + result.ocr_timing.rec_preprocess_sec +
         result.ocr_timing.rec_infer_sec + result.ocr_timing.rec_postprocess_sec) *
        1000.0;
    const double total_ms = result.overall_latency_sec * 1000.0;

    constexpr int kPadX = 32;
    constexpr int kPadY = 28;
    constexpr int kLeftBlockHeight = 30 + 34 + 32 * 2 + 22;
    const int split_x = rect.x + rect.width * 62 / 100;

    const int left_px = rect.x + kPadX;
    int left_py = rect.y + std::max(kPadY, (rect.height - kLeftBlockHeight) / 2);
    font_renderer.Draw(canvas, "Performance", {left_px, left_py}, 22, kColorTextPrimary);
    left_py += 30;
    font_renderer.Draw(canvas, "Latency per pipeline stage", {left_px, left_py}, 15,
                       kColorTextMuted);
    left_py += 34;

    constexpr int kLabelFont = 19;
    constexpr int kValueFont = 20;
    const int value_right = split_x - 32;
    auto draw_row = [&](const std::string& label, double ms, double warn_ms, double bad_ms) {
        const int dot_radius = 5;
        cv::circle(*canvas, {left_px + dot_radius, left_py + kLabelFont / 2 + 4}, dot_radius,
                   LatencyStatusColor(ms, warn_ms, bad_ms), cv::FILLED, cv::LINE_AA);
        font_renderer.Draw(canvas, label, {left_px + dot_radius * 2 + 14, left_py}, kLabelFont,
                           kColorTextSecondary);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << ms << " ms";
        const FontRenderer::TextMetrics metrics =
            font_renderer.MeasureText(oss.str(), kValueFont);
        font_renderer.Draw(canvas, oss.str(), {value_right - metrics.width, left_py - 1},
                           kValueFont, kColorTextPrimary);
        left_py += 32;
    };
    draw_row("Detection (YOLO)", yolo_ms, 20.0, 40.0);
    draw_row("OCR Detection", ocr_det_ms, 30.0, 60.0);
    draw_row("OCR Recognition", ocr_rec_ms, 40.0, 80.0);

    const int divider_x = split_x;
    const int divider_top = rect.y + kPadY + 6;
    const int divider_bottom = rect.y + rect.height - kPadY - 6;
    cv::line(*canvas, {divider_x, divider_top}, {divider_x, divider_bottom}, kColorPanelBorder, 1,
             cv::LINE_AA);

    const int right_px = split_x + 32;
    constexpr int kTotalFont = 56;
    constexpr int kRightBlockHeight = 18 + 14 + kTotalFont + 12 + 15;
    const int right_top = rect.y + (rect.height - kRightBlockHeight) / 2;
    font_renderer.Draw(canvas, "Total latency", {right_px, right_top}, 18, kColorTextMuted);
    std::ostringstream total_oss;
    total_oss << std::fixed << std::setprecision(1) << total_ms << " ms";
    font_renderer.Draw(canvas, total_oss.str(), {right_px, right_top + 18 + 14}, kTotalFont,
                       kColorBrandPrimary);
    font_renderer.Draw(canvas, "End-to-end per frame",
                       {right_px, right_top + 18 + 14 + kTotalFont + 12}, 15, kColorTextMuted);
}

cv::Mat MakeDetectionPlot(const cv::Mat& image, const Detection& detection) {
    cv::Mat annotated = image.clone();
    const int tl =
        std::max(2, static_cast<int>(std::round(0.0035 * (image.rows + image.cols) / 2.0)));
    cv::Rect rect(static_cast<int>(detection.x1), static_cast<int>(detection.y1),
                  static_cast<int>(detection.x2 - detection.x1),
                  static_cast<int>(detection.y2 - detection.y1));
    cv::rectangle(annotated, rect, {0, 180, 255}, tl, cv::LINE_AA);
    std::ostringstream label;
    label << "waybill: " << std::fixed << std::setprecision(1) << detection.score * 100.0f << "%";
    cv::putText(annotated, label.str(), {rect.x, std::max(20, rect.y - 10)},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
    return annotated;
}

bool RectsOverlap(const cv::Rect& a, const cv::Rect& b) {
    return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height &&
           a.y + a.height > b.y;
}

cv::Rect ClampRect(int x1, int y1, int width, int height, int max_w, int max_h) {
    x1 = std::min(std::max(0, x1), std::max(0, max_w - width));
    y1 = std::min(std::max(0, y1), std::max(0, max_h - height));
    return {x1, y1, width, height};
}

cv::Rect PlaceNonOverlappingRect(const cv::Rect& preferred,
                                 const std::vector<cv::Rect>& placed_rects, int max_w, int max_h) {
    cv::Rect best =
        ClampRect(preferred.x, preferred.y, preferred.width, preferred.height, max_w, max_h);
    bool overlaps = false;
    for (const auto& existing : placed_rects) {
        if (RectsOverlap(best, existing)) {
            overlaps = true;
            break;
        }
    }
    if (!overlaps) {
        return best;
    }
    const int search_steps = std::max(max_w, max_h);
    for (int radius = 4; radius < search_steps; radius += 4) {
        const std::vector<cv::Rect> candidates = {
            ClampRect(preferred.x, preferred.y + radius, preferred.width, preferred.height, max_w,
                      max_h),
            ClampRect(preferred.x, preferred.y - radius, preferred.width, preferred.height, max_w,
                      max_h),
            ClampRect(preferred.x + radius, preferred.y, preferred.width, preferred.height, max_w,
                      max_h),
            ClampRect(preferred.x - radius, preferred.y, preferred.width, preferred.height, max_w,
                      max_h),
            ClampRect(preferred.x + radius, preferred.y + radius, preferred.width, preferred.height,
                      max_w, max_h),
            ClampRect(preferred.x - radius, preferred.y + radius, preferred.width, preferred.height,
                      max_w, max_h),
            ClampRect(preferred.x + radius, preferred.y - radius, preferred.width, preferred.height,
                      max_w, max_h),
            ClampRect(preferred.x - radius, preferred.y - radius, preferred.width, preferred.height,
                      max_w, max_h)};
        for (const auto& candidate : candidates) {
            bool candidate_overlaps = false;
            for (const auto& existing : placed_rects) {
                if (RectsOverlap(candidate, existing)) {
                    candidate_overlaps = true;
                    break;
                }
            }
            if (!candidate_overlaps) {
                return candidate;
            }
        }
    }
    return best;
}

cv::Mat DrawTextPatch(const std::string& text, int width, int height,
                      const FontRenderer& font_renderer) {
    cv::Mat patch(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    if (!text.empty()) {
        const int max_text_w = std::max(1, width - 4);
        const int max_text_h = std::max(1, height - 4);
        const int font_size = font_renderer.PickFontSizeForBounds(text, max_text_w, max_text_h, 8,
                                                                  std::min(96, height));
        const FontRenderer::TextMetrics metrics = font_renderer.MeasureText(text, font_size);
        const int x = std::max(1, (width - metrics.width) / 2);
        const int y = (height - metrics.height) / 2 - metrics.min_y - metrics.baseline;
        font_renderer.Draw(&patch, text, {x, y}, font_size, {0, 0, 0});
    }
    return patch;
}

std::string CanonicalTextKey(const std::string& text) {
    std::string key;
    key.reserve(text.size());
    for (unsigned char ch : text) {
        if (!std::isspace(ch) && ch != '-') {
            key.push_back(static_cast<char>(std::toupper(ch)));
        }
    }
    return key;
}

bool IsRouteCodeAnchor(const std::string& text) {
    const std::string key = CanonicalTextKey(text);
    if (key.size() < 2 || key.size() > 4 || key[0] < 'A' || key[0] > 'Z') {
        return false;
    }
    for (size_t i = 1; i < key.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(key[i]))) {
            return false;
        }
    }
    return true;
}

cv::Point2f TransformNormalizedPoint(const cv::Point2f& point,
                                     NormalizedLayoutTransform transform) {
    switch (transform) {
    case NormalizedLayoutTransform::kRotateClockwise90:
        return {std::clamp(1.0f - point.y, 0.0f, 1.0f), std::clamp(point.x, 0.0f, 1.0f)};
    case NormalizedLayoutTransform::kRotateCounterClockwise90:
        return {std::clamp(point.y, 0.0f, 1.0f), std::clamp(1.0f - point.x, 0.0f, 1.0f)};
    case NormalizedLayoutTransform::kRotate180:
        return {std::clamp(1.0f - point.x, 0.0f, 1.0f), std::clamp(1.0f - point.y, 0.0f, 1.0f)};
    case NormalizedLayoutTransform::kIdentity:
    default:
        return {std::clamp(point.x, 0.0f, 1.0f), std::clamp(point.y, 0.0f, 1.0f)};
    }
}

cv::Rect2f BoundingRectFromPoints(const std::vector<cv::Point2f>& points);

void RecomputeCanonicalGeometry(DrawTextItem* item) {
    item->canonical_center =
        std::accumulate(item->canonical_box.begin(), item->canonical_box.end(),
                        cv::Point2f(0.0f, 0.0f),
                        [](const cv::Point2f& acc, const cv::Point2f& point) {
                            return cv::Point2f(acc.x + point.x, acc.y + point.y);
                        }) *
        (1.0f / static_cast<float>(item->canonical_box.size()));
    item->canonical_bounds = BoundingRectFromPoints(item->canonical_box);
}

std::optional<cv::Point2f> BoxCenter(const std::vector<cv::Point2f>& points) {
    cv::Point2f sum(0.0f, 0.0f);
    int count = 0;
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }
        sum += point;
        ++count;
    }
    if (count == 0) {
        return std::nullopt;
    }
    return sum * (1.0f / static_cast<float>(count));
}

std::vector<DrawTextItem> KeepDominantOcrLayoutCluster(const std::vector<DrawTextItem>& items) {
    if (items.size() < kOcrLayoutOutlierMinItems) {
        return items;
    }

    std::vector<cv::Point2f> centers(items.size());
    std::vector<bool> has_center(items.size(), false);
    int valid_count = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto center = BoxCenter(items[i].box);
        if (!center.has_value()) {
            continue;
        }
        centers[i] = *center;
        has_center[i] = true;
        ++valid_count;
    }
    if (valid_count < static_cast<int>(kOcrLayoutOutlierMinItems)) {
        return items;
    }

    std::vector<float> nearest_distances;
    nearest_distances.reserve(valid_count);
    for (size_t i = 0; i < items.size(); ++i) {
        if (!has_center[i]) {
            continue;
        }
        float nearest = std::numeric_limits<float>::max();
        for (size_t j = 0; j < items.size(); ++j) {
            if (i == j || !has_center[j]) {
                continue;
            }
            nearest = std::min(nearest, static_cast<float>(cv::norm(centers[i] - centers[j])));
        }
        if (std::isfinite(nearest)) {
            nearest_distances.push_back(nearest);
        }
    }
    if (nearest_distances.empty()) {
        return items;
    }
    std::sort(nearest_distances.begin(), nearest_distances.end());
    const float median_nearest = nearest_distances[nearest_distances.size() / 2];
    const float neighbor_threshold =
        std::max(kOcrLayoutOutlierMinNeighborDistance,
                 median_nearest * kOcrLayoutOutlierNeighborScale);

    std::vector<int> component(items.size(), -1);
    std::vector<int> component_sizes;
    int component_id = 0;
    for (size_t start = 0; start < items.size(); ++start) {
        if (!has_center[start] || component[start] >= 0) {
            continue;
        }
        std::vector<size_t> stack = {start};
        component[start] = component_id;
        int size = 0;
        while (!stack.empty()) {
            const size_t current = stack.back();
            stack.pop_back();
            ++size;
            for (size_t next = 0; next < items.size(); ++next) {
                if (!has_center[next] || component[next] >= 0) {
                    continue;
                }
                if (cv::norm(centers[current] - centers[next]) <= neighbor_threshold) {
                    component[next] = component_id;
                    stack.push_back(next);
                }
            }
        }
        component_sizes.push_back(size);
        ++component_id;
    }
    if (component_sizes.size() <= 1) {
        return items;
    }

    const auto largest_it = std::max_element(component_sizes.begin(), component_sizes.end());
    const int largest_component = static_cast<int>(largest_it - component_sizes.begin());
    const int largest_size = *largest_it;
    const int dropped_count = valid_count - largest_size;
    if (largest_size < static_cast<int>(std::ceil(valid_count * kOcrLayoutOutlierMinKeepRatio)) ||
        dropped_count <= 0 ||
        dropped_count > static_cast<int>(std::floor(valid_count * kOcrLayoutOutlierMaxDropRatio))) {
        return items;
    }

    std::vector<DrawTextItem> filtered;
    filtered.reserve(static_cast<size_t>(largest_size));
    for (size_t i = 0; i < items.size(); ++i) {
        if (component[i] == largest_component) {
            filtered.push_back(items[i]);
        }
    }
    return filtered.empty() ? items : filtered;
}

const DrawTextItem* PickRouteGeometryAnchor(const std::vector<DrawTextItem>& normalized_items) {
    std::vector<float> areas;
    areas.reserve(normalized_items.size());
    for (const auto& item : normalized_items) {
        const float area = item.canonical_bounds.area();
        if (std::isfinite(area) && area > 0.0f) {
            areas.push_back(area);
        }
    }
    if (areas.empty()) {
        return nullptr;
    }
    std::sort(areas.begin(), areas.end());
    const float median_area = std::max(areas[areas.size() / 2], 1e-5f);

    const DrawTextItem* best = nullptr;
    float best_score = 0.0f;
    for (const auto& item : normalized_items) {
        const std::string key = CanonicalTextKey(item.text);
        const bool has_short_text = !key.empty() && key.size() <= 6;
        if (!key.empty() && !has_short_text) {
            continue;
        }
        const float width = item.canonical_bounds.width;
        const float height = item.canonical_bounds.height;
        const float area = width * height;
        if (!std::isfinite(area)) {
            continue;
        }
        const float short_side = std::max(std::min(width, height), 1e-5f);
        const float aspect = std::max(width, height) / short_side;
        const float max_aspect = has_short_text ? 8.0f : 2.7f;
        const float min_area_factor = has_short_text ? 1.35f : 2.2f;
        if (area < median_area * min_area_factor || aspect > max_aspect) {
            continue;
        }

        float score = area;
        if (has_short_text && key.size() <= 4) {
            score *= 1.25f;
        }
        const bool has_ascii_letter =
            std::any_of(key.begin(), key.end(), [](char ch) { return ch >= 'A' && ch <= 'Z'; });
        const bool has_digit =
            std::any_of(key.begin(), key.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
        if (has_ascii_letter && has_digit) {
            score *= 1.5f;
        }
        if (!has_short_text) {
            score *= 0.95f;
        }
        if (score > best_score) {
            best = &item;
            best_score = score;
        }
    }
    return best;
}

NormalizedLayoutTransform
ChooseCanonicalLayoutTransform(const std::vector<DrawTextItem>& normalized_items,
                               const LayoutFrame& layout_frame) {
    const DrawTextItem* anchor = nullptr;
    for (const auto& item : normalized_items) {
        if (IsRouteCodeAnchor(item.text)) {
            anchor = &item;
            break;
        }
    }
    if (anchor == nullptr) {
        anchor = PickRouteGeometryAnchor(normalized_items);
    }

    if (anchor != nullptr) {
        const std::vector<NormalizedLayoutTransform> candidates = {
            NormalizedLayoutTransform::kIdentity, NormalizedLayoutTransform::kRotateClockwise90,
            NormalizedLayoutTransform::kRotateCounterClockwise90,
            NormalizedLayoutTransform::kRotate180};
        NormalizedLayoutTransform best = NormalizedLayoutTransform::kIdentity;
        float best_score = -std::numeric_limits<float>::max();
        for (const auto candidate : candidates) {
            const cv::Point2f center =
                TransformNormalizedPoint(anchor->canonical_center, candidate);
            const float upper_right_score = center.x + (1.0f - center.y);
            if (upper_right_score > best_score) {
                best = candidate;
                best_score = upper_right_score;
            }
        }
        return best;
    }

    if (layout_frame.valid && layout_frame.bounds.width < layout_frame.bounds.height) {
        return NormalizedLayoutTransform::kRotateCounterClockwise90;
    }
    return NormalizedLayoutTransform::kIdentity;
}

void ApplyCanonicalLayoutTransform(std::vector<DrawTextItem>* normalized_items,
                                   NormalizedLayoutTransform transform) {
    if (transform == NormalizedLayoutTransform::kIdentity) {
        return;
    }
    for (auto& item : *normalized_items) {
        for (auto& point : item.canonical_box) {
            point = TransformNormalizedPoint(point, transform);
        }
        RecomputeCanonicalGeometry(&item);
    }
}

cv::Rect2f BoundingRectFromPoints(const std::vector<cv::Point2f>& points) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    bool has_point = false;
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
        has_point = true;
    }
    if (!has_point) {
        return {};
    }
    return {min_x, min_y, std::max(0.0f, max_x - min_x), std::max(0.0f, max_y - min_y)};
}

bool IsFinitePoint(const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

float DotPoint(const cv::Point2f& lhs, const cv::Point2f& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

std::optional<cv::Point2f> NormalizeVector(const cv::Point2f& vector) {
    const float length = cv::norm(vector);
    if (!std::isfinite(length) || length < 1e-4f) {
        return std::nullopt;
    }
    return vector * (1.0f / length);
}

cv::Point2f PerpendicularClockwise(const cv::Point2f& axis) { return {-axis.y, axis.x}; }

std::vector<cv::Point2f> CollectOcrBoxPoints(const std::vector<DrawTextItem>& items) {
    std::vector<cv::Point2f> points;
    for (const auto& item : items) {
        for (const auto& point : item.box) {
            if (IsFinitePoint(point)) {
                points.push_back(point);
            }
        }
    }
    return points;
}

struct BoxAxisEstimate {
    cv::Point2f axis = {1.0f, 0.0f};
    float long_side = 0.0f;
    float short_side = 0.0f;
};

std::optional<BoxAxisEstimate> EstimateBoxAxis(const std::vector<cv::Point2f>& box) {
    std::vector<cv::Point2f> finite_points;
    finite_points.reserve(box.size());
    for (const auto& point : box) {
        if (IsFinitePoint(point)) {
            finite_points.push_back(point);
        }
    }
    if (finite_points.size() < 3) {
        return std::nullopt;
    }

    const cv::RotatedRect rect = cv::minAreaRect(finite_points);
    cv::Point2f vertices[4];
    rect.points(vertices);
    const cv::Point2f edge0 = vertices[1] - vertices[0];
    const cv::Point2f edge1 = vertices[2] - vertices[1];
    const float len0 = cv::norm(edge0);
    const float len1 = cv::norm(edge1);
    if (!std::isfinite(len0) || !std::isfinite(len1)) {
        return std::nullopt;
    }

    const cv::Point2f long_edge = len0 >= len1 ? edge0 : edge1;
    const auto axis = NormalizeVector(long_edge);
    if (!axis.has_value()) {
        return std::nullopt;
    }

    BoxAxisEstimate estimate;
    estimate.axis = *axis;
    estimate.long_side = std::max(len0, len1);
    estimate.short_side = std::min(len0, len1);
    return estimate;
}

std::optional<cv::Point2f> EstimateDominantLayoutAxis(const std::vector<DrawTextItem>& items) {
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    double weight_sum = 0.0;

    for (const auto& item : items) {
        const auto estimate = EstimateBoxAxis(item.box);
        if (!estimate.has_value() || estimate->long_side < 4.0f || estimate->short_side < 1.0f) {
            continue;
        }
        const float aspect = estimate->long_side / std::max(estimate->short_side, 1.0f);
        if (aspect < 1.15f) {
            continue;
        }

        const double theta = std::atan2(estimate->axis.y, estimate->axis.x);
        const double weight = std::clamp(static_cast<double>(estimate->long_side), 4.0, 96.0);
        sum_cos += weight * std::cos(theta * 2.0);
        sum_sin += weight * std::sin(theta * 2.0);
        weight_sum += weight;
    }

    if (weight_sum <= 0.0 || std::hypot(sum_cos, sum_sin) / weight_sum < 0.08) {
        return std::nullopt;
    }

    const double theta = 0.5 * std::atan2(sum_sin, sum_cos);
    return NormalizeVector(
        {static_cast<float>(std::cos(theta)), static_cast<float>(std::sin(theta))});
}

std::optional<cv::Point2f>
EstimateLayoutAxisFromPointCloud(const std::vector<cv::Point2f>& points) {
    if (points.size() < 3) {
        return std::nullopt;
    }
    const auto estimate = EstimateBoxAxis(points);
    if (!estimate.has_value()) {
        return std::nullopt;
    }
    return estimate->axis;
}

std::pair<float, float> RobustProjectionRange(std::vector<float> values) {
    if (values.empty()) {
        return {0.0f, 1.0f};
    }
    std::sort(values.begin(), values.end());
    size_t trim = 0;
    if (values.size() >= 80) {
        trim = static_cast<size_t>(
            std::floor(static_cast<float>(values.size()) * kOcrLayoutProjectionTrimRatio));
        trim = std::min(trim, (values.size() - 1) / 2);
    }

    float low = values[trim];
    float high = values[values.size() - 1 - trim];
    if (!std::isfinite(low) || !std::isfinite(high) || high - low < 1.0f) {
        low = values.front();
        high = values.back();
    }
    if (high - low < 1.0f) {
        high = low + 1.0f;
    }
    return {low, high};
}

cv::Rect2f EstimateProjectedBounds(const std::vector<cv::Point2f>& points,
                                   const cv::Point2f& x_axis, const cv::Point2f& y_axis) {
    std::vector<float> projected_x;
    std::vector<float> projected_y;
    projected_x.reserve(points.size());
    projected_y.reserve(points.size());
    for (const auto& point : points) {
        if (!IsFinitePoint(point)) {
            continue;
        }
        projected_x.push_back(DotPoint(point, x_axis));
        projected_y.push_back(DotPoint(point, y_axis));
    }
    if (projected_x.empty() || projected_y.empty()) {
        return {};
    }

    const auto [min_x, max_x] = RobustProjectionRange(std::move(projected_x));
    const auto [min_y, max_y] = RobustProjectionRange(std::move(projected_y));
    const float layout_width = std::max(1.0f, max_x - min_x);
    const float layout_height = std::max(1.0f, max_y - min_y);
    const float padding_x = layout_width * kOcrLayoutPaddingXRatio;
    const float padding_y = layout_height * kOcrLayoutPaddingYRatio;
    return {min_x - padding_x, min_y - padding_y, layout_width + padding_x * 2.0f,
            layout_height + padding_y * 2.0f};
}

LayoutFrame EstimateLayoutFrameFromOcrBoxes(const std::vector<DrawTextItem>& items) {
    const std::vector<cv::Point2f> points = CollectOcrBoxPoints(items);
    if (points.empty()) {
        return {};
    }

    auto x_axis = EstimateDominantLayoutAxis(items);
    if (!x_axis.has_value()) {
        x_axis = EstimateLayoutAxisFromPointCloud(points);
    }
    if (!x_axis.has_value()) {
        x_axis = cv::Point2f(1.0f, 0.0f);
    }

    cv::Point2f frame_x_axis = *x_axis;
    cv::Point2f frame_y_axis = PerpendicularClockwise(frame_x_axis);
    cv::Rect2f bounds = EstimateProjectedBounds(points, frame_x_axis, frame_y_axis);

    if (bounds.width < bounds.height) {
        frame_x_axis = frame_y_axis;
        frame_y_axis = PerpendicularClockwise(frame_x_axis);
        bounds = EstimateProjectedBounds(points, frame_x_axis, frame_y_axis);
    }

    LayoutFrame frame;
    frame.bounds = bounds;
    frame.x_axis = frame_x_axis;
    frame.y_axis = frame_y_axis;
    frame.valid = frame.bounds.width > 0.0f && frame.bounds.height > 0.0f;
    frame.oriented = true;
    return frame;
}

std::vector<DrawTextItem> NormalizeOcrResultsToLayoutFrame(const std::vector<DrawTextItem>& items,
                                                           const LayoutFrame& layout_frame) {
    std::vector<DrawTextItem> normalized;
    normalized.reserve(items.size());
    if (!layout_frame.valid) {
        return normalized;
    }

    for (const auto& item : items) {
        DrawTextItem normalized_item = item;
        normalized_item.canonical_box.clear();
        for (const auto& point : item.box) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                continue;
            }
            const float projected_x = DotPoint(point, layout_frame.x_axis);
            const float projected_y = DotPoint(point, layout_frame.y_axis);
            const float x = (projected_x - layout_frame.bounds.x) / layout_frame.bounds.width;
            const float y = (projected_y - layout_frame.bounds.y) / layout_frame.bounds.height;
            normalized_item.canonical_box.emplace_back(std::clamp(x, 0.0f, 1.0f),
                                                       std::clamp(y, 0.0f, 1.0f));
        }
        if (normalized_item.canonical_box.empty()) {
            continue;
        }
        RecomputeCanonicalGeometry(&normalized_item);
        normalized.push_back(std::move(normalized_item));
    }
    return normalized;
}

cv::Size GetCanonicalCanvasSize(const LayoutFrame& layout_frame) {
    (void)layout_frame;
    const int target_panel_w = (kLayoutWidth / 2) - 36;
    const int target_panel_h = (kLayoutHeight * 6 / 10) - 50 - 36;
    int canvas_w = target_panel_w;
    int canvas_h =
        std::max(1, static_cast<int>(std::round(canvas_w / kCanonicalWaybillAspectRatio)));
    if (canvas_h > target_panel_h) {
        canvas_h = target_panel_h;
        canvas_w =
            std::max(1, static_cast<int>(std::round(canvas_h * kCanonicalWaybillAspectRatio)));
    }
    return {std::max(1, canvas_w), std::max(1, canvas_h)};
}

cv::Mat MakeOcrTextImage(const std::vector<OcrResult>& results,
                         const std::vector<TextBox>& detected_boxes, float drop_score,
                         const FontRenderer& font_renderer) {
    std::vector<DrawTextItem> layout_items;
    std::vector<DrawTextItem> result_items;
    std::vector<DrawTextItem> orientation_items;
    std::vector<DrawTextItem> draw_items;
    layout_items.reserve(detected_boxes.size());
    result_items.reserve(results.size());
    orientation_items.reserve(detected_boxes.size() + results.size());
    draw_items.reserve(results.size());
    for (const auto& box : detected_boxes) {
        if (box.points.empty()) {
            continue;
        }
        DrawTextItem item;
        item.box = box.points;
        layout_items.push_back(item);
    }
    for (const auto& result : results) {
        if (result.box.points.empty()) {
            continue;
        }
        DrawTextItem item;
        item.box = result.box.points;
        item.text = result.text;
        result_items.push_back(item);
        if (result.score >= drop_score) {
            draw_items.push_back(item);
        }
    }
    if (layout_items.empty()) {
        layout_items = result_items;
    }
    layout_items = KeepDominantOcrLayoutCluster(layout_items);
    result_items = KeepDominantOcrLayoutCluster(result_items);
    draw_items = KeepDominantOcrLayoutCluster(draw_items);
    orientation_items = layout_items;
    orientation_items.insert(orientation_items.end(), result_items.begin(), result_items.end());

    const LayoutFrame layout_frame = EstimateLayoutFrameFromOcrBoxes(layout_items);
    const cv::Size canvas_size = GetCanonicalCanvasSize(layout_frame);
    const int canvas_w = canvas_size.width;
    const int canvas_h = canvas_size.height;
    cv::Mat ocr_text_image(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));
    if (!layout_frame.valid || draw_items.empty()) {
        return ocr_text_image;
    }

    std::vector<DrawTextItem> normalized =
        NormalizeOcrResultsToLayoutFrame(draw_items, layout_frame);
    if (normalized.empty()) {
        return ocr_text_image;
    }
    std::vector<DrawTextItem> normalized_for_orientation =
        NormalizeOcrResultsToLayoutFrame(orientation_items, layout_frame);
    const NormalizedLayoutTransform layout_transform = ChooseCanonicalLayoutTransform(
        normalized_for_orientation.empty() ? normalized : normalized_for_orientation, layout_frame);
    ApplyCanonicalLayoutTransform(&normalized, layout_transform);
    std::sort(normalized.begin(), normalized.end(),
              [](const DrawTextItem& lhs, const DrawTextItem& rhs) {
                  return std::tie(lhs.canonical_center.y, lhs.canonical_center.x) <
                         std::tie(rhs.canonical_center.y, rhs.canonical_center.x);
              });

    std::vector<float> short_sides;
    short_sides.reserve(normalized.size());
    for (const auto& item : normalized) {
        const float box_w = std::max(1.0f, item.canonical_bounds.width * canvas_w);
        const float box_h = std::max(1.0f, item.canonical_bounds.height * canvas_h);
        short_sides.push_back(std::min(box_w, box_h));
    }
    float display_scale = 1.0f;
    if (!short_sides.empty()) {
        std::sort(short_sides.begin(), short_sides.end());
        const float median_short_side = short_sides[short_sides.size() / 2];
        const float target_median_height =
            static_cast<float>(std::clamp(static_cast<int>(std::round(canvas_h * 0.075)), 32, 64));
        display_scale = std::clamp(target_median_height / median_short_side, 1.2f, 2.5f);
    }
    display_scale *= 0.75f;
    constexpr int kBaseMinDisplayHeight = 8;
    constexpr int kBaseMinDisplayWidth = 12;
    const int min_display_h = std::max(1, kBaseMinDisplayHeight);
    const int min_display_w = std::max(1, kBaseMinDisplayWidth);
    std::vector<cv::Rect> placed_rects;
    auto place_with_minimal_shift = [&](const cv::Rect& preferred) {
        auto has_overlap = [&](const cv::Rect& candidate) {
            constexpr int kGap = 2;
            cv::Rect padded_candidate(candidate.x - kGap, candidate.y - kGap,
                                      candidate.width + kGap * 2, candidate.height + kGap * 2);
            for (const auto& existing : placed_rects) {
                if (RectsOverlap(padded_candidate, existing)) {
                    return true;
                }
            }
            return false;
        };

        auto search_for_size = [&](int width, int height, int max_search_radius) {
            const int center_x = preferred.x + preferred.width / 2;
            const int center_y = preferred.y + preferred.height / 2;
            const cv::Rect resized_preferred(center_x - width / 2, center_y - height / 2, width,
                                             height);
            const cv::Rect clamped =
                ClampRect(resized_preferred.x, resized_preferred.y, resized_preferred.width,
                          resized_preferred.height, canvas_w, canvas_h);
            if (!has_overlap(clamped)) {
                return clamped;
            }

            const int step = std::max(2, std::min(width, height) / 6);
            const int max_radius = std::min(std::max(canvas_w, canvas_h), max_search_radius);
            cv::Rect best = clamped;
            int best_distance = std::numeric_limits<int>::max();

            for (int radius = step; radius <= max_radius; radius += step) {
                auto try_candidate = [&](int dx, int dy) {
                    const cv::Rect candidate = ClampRect(
                        resized_preferred.x + dx, resized_preferred.y + dy, resized_preferred.width,
                        resized_preferred.height, canvas_w, canvas_h);
                    if (!has_overlap(candidate)) {
                        const int candidate_center_x = candidate.x + candidate.width / 2;
                        const int candidate_center_y = candidate.y + candidate.height / 2;
                        const int distance = std::abs(candidate_center_x - center_x) +
                                             std::abs(candidate_center_y - center_y);
                        if (distance < best_distance) {
                            best = candidate;
                            best_distance = distance;
                        }
                    }
                };

                for (int dx = -radius; dx <= radius; dx += step) {
                    try_candidate(dx, -radius);
                    try_candidate(dx, radius);
                }
                for (int dy = -radius + step; dy <= radius - step; dy += step) {
                    try_candidate(-radius, dy);
                    try_candidate(radius, dy);
                }
                if (best_distance != std::numeric_limits<int>::max()) {
                    return best;
                }
            }
            return cv::Rect();
        };

        auto scan_for_size = [&](int width, int height) {
            const int center_x = preferred.x + preferred.width / 2;
            const int center_y = preferred.y + preferred.height / 2;
            cv::Rect best;
            int best_distance = std::numeric_limits<int>::max();
            constexpr int kStep = 2;
            for (int y = 0; y <= canvas_h - height; y += kStep) {
                for (int x = 0; x <= canvas_w - width; x += kStep) {
                    const cv::Rect candidate(x, y, width, height);
                    if (has_overlap(candidate)) {
                        continue;
                    }
                    const int candidate_center_x = candidate.x + candidate.width / 2;
                    const int candidate_center_y = candidate.y + candidate.height / 2;
                    const int distance = std::abs(candidate_center_x - center_x) +
                                         std::abs(candidate_center_y - center_y);
                    if (distance < best_distance) {
                        best = candidate;
                        best_distance = distance;
                    }
                }
            }
            return best;
        };

        const int base_search_radius = std::max(
            12, static_cast<int>(std::round(std::max(preferred.width, preferred.height) * 1.4)));
        cv::Rect placed = search_for_size(preferred.width, preferred.height, base_search_radius);
        if (!placed.empty()) {
            return placed;
        }

        const int min_w = std::max(1, min_display_w);
        const int min_h = std::max(1, min_display_h);
        for (double ratio = 0.92; ratio >= 0.42; ratio -= 0.05) {
            const int new_w =
                std::max(min_w, static_cast<int>(std::round(preferred.width * ratio)));
            const int new_h =
                std::max(min_h, static_cast<int>(std::round(preferred.height * ratio)));
            const int local_radius =
                std::max(12, static_cast<int>(std::round(std::max(new_w, new_h) * 1.8)));
            placed = search_for_size(new_w, new_h, local_radius);
            if (!placed.empty()) {
                return placed;
            }
        }
        placed = scan_for_size(min_w, min_h);
        if (!placed.empty()) {
            return placed;
        }
        return ClampRect(preferred.x, preferred.y, min_w, min_h, canvas_w, canvas_h);
    };
    for (size_t index = 0; index < normalized.size(); ++index) {
        const auto& item = normalized[index];
        const cv::Scalar color((37 * (static_cast<int>(index) + 3)) % 255,
                               (83 * (static_cast<int>(index) + 5)) % 255,
                               (151 * (static_cast<int>(index) + 7)) % 255);
        const float box_w = std::max(1.0f, item.canonical_bounds.width * canvas_w);
        const float box_h = std::max(1.0f, item.canonical_bounds.height * canvas_h);
        const float long_side = std::max(box_w, box_h);
        const float short_side = std::min(box_w, box_h);
        const int base_upright_h = std::max(
            kBaseMinDisplayHeight, static_cast<int>(std::round(short_side * display_scale)));
        const int base_upright_w =
            std::max(kBaseMinDisplayWidth, static_cast<int>(std::round(long_side * display_scale)));
        const int upright_h = std::min(canvas_h, std::max(min_display_h, base_upright_h));
        const int upright_w = std::min(canvas_w, std::max(min_display_w, base_upright_w));
        const int base_inner_pad = std::clamp(base_upright_h / 8, 2, 5);
        const int inner_pad = std::max(1, base_inner_pad);
        const float center_x =
            std::clamp(item.canonical_center.x, 0.0f, 1.0f) * static_cast<float>(canvas_w);
        const float center_y =
            std::clamp(item.canonical_center.y, 0.0f, 1.0f) * static_cast<float>(canvas_h);
        cv::Rect preferred(static_cast<int>(std::round(center_x - upright_w / 2.0f)),
                           static_cast<int>(std::round(center_y - upright_h / 2.0f)), upright_w,
                           upright_h);
        cv::Rect placed = place_with_minimal_shift(preferred);
        cv::rectangle(ocr_text_image, placed, cv::Scalar(255, 255, 255), -1);
        const int draw_pad =
            std::min(inner_pad, std::max(1, (std::min(placed.width, placed.height) - 2) / 4));
        const int text_w = std::max(1, placed.width - draw_pad * 2);
        const int text_h = std::max(1, placed.height - draw_pad * 2);
        cv::Mat patch = DrawTextPatch(item.text, text_w, text_h, font_renderer);
        patch.copyTo(ocr_text_image(
            cv::Rect(placed.x + draw_pad, placed.y + draw_pad, patch.cols, patch.rows)));
        cv::rectangle(ocr_text_image, placed, color, 1);
        placed_rects.push_back(placed);
    }
    return ocr_text_image;
}

cv::Mat MakeCropWithBoxes(const cv::Mat& crop_bgr, const std::vector<TextBox>& boxes) {
    cv::Mat crop_with_boxes = crop_bgr.clone();
    cv::Mat overlay = crop_bgr.clone();
    const int tl =
        std::max(2, static_cast<int>(std::round(0.004 * (crop_bgr.rows + crop_bgr.cols) / 2.0)));

    for (size_t i = 0; i < boxes.size(); ++i) {
        std::vector<cv::Point> poly;
        for (const auto& point : boxes[i].points) {
            poly.emplace_back(static_cast<int>(std::round(point.x)),
                              static_cast<int>(std::round(point.y)));
        }
        const cv::Scalar color((37 * (static_cast<int>(i) + 3)) % 255,
                               (83 * (static_cast<int>(i) + 5)) % 255,
                               (151 * (static_cast<int>(i) + 7)) % 255);
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{poly}, color);
        cv::polylines(crop_with_boxes, std::vector<std::vector<cv::Point>>{poly}, true, color, tl,
                      cv::LINE_AA);
    }
    cv::addWeighted(overlay, 0.4, crop_with_boxes, 0.6, 0.0, crop_with_boxes);
    return crop_with_boxes;
}

cv::Mat ComposeFinalLayout(const cv::Mat& original_plot, const cv::Mat& crop_with_boxes,
                           const cv::Mat& ocr_text_image, const PipelineResult& result,
                           const FontRenderer& font_renderer) {
    cv::Mat canvas(kLayoutHeight, kLayoutWidth, CV_8UC3, kColorCanvasBg);
    DrawHeader(&canvas, font_renderer);

    const int content_top = kHeaderHeight + kLayoutOuterMargin;
    const int content_bottom = kLayoutHeight - kLayoutOuterMargin;
    const int content_left = kLayoutOuterMargin;
    const int content_right = kLayoutWidth - kLayoutOuterMargin;
    const int content_w = content_right - content_left;
    const int content_h = content_bottom - content_top;
    const int left_w = (content_w - kLayoutGutter) / 2;
    const int right_w = content_w - kLayoutGutter - left_w;
    const int right_x = content_left + left_w + kLayoutGutter;

    const int row_content_h = content_h - kSectionLabelHeight * 2 - kLayoutGutter;
    const int left_top_h = row_content_h * 65 / 100;
    const int left_bottom_h = row_content_h - left_top_h;
    const int right_top_h = row_content_h * 40 / 100;
    const int right_bottom_h = row_content_h - right_top_h;

    const int left_top_panel_y = content_top + kSectionLabelHeight;
    const int left_bottom_label_y = left_top_panel_y + left_top_h + kLayoutGutter;
    const int left_bottom_panel_y = left_bottom_label_y + kSectionLabelHeight;

    const int right_top_panel_y = content_top + kSectionLabelHeight;
    const int right_bottom_label_y = right_top_panel_y + right_top_h + kLayoutGutter;
    const int right_bottom_panel_y = right_bottom_label_y + kSectionLabelHeight;

    const PanelBadge detection_badge =
        result.detected
            ? PanelBadge{"DETECTED", kColorBrandPrimary, cv::Scalar(255, 255, 255)}
            : PanelBadge{"SCANNING", cv::Scalar(70, 66, 62), kColorTextSecondary};
    DrawSectionLabel(&canvas, content_left, content_top, "Live capture + detection", font_renderer,
                     true, detection_badge, left_w);
    DrawPanel(&canvas, original_plot, content_left, left_top_panel_y, left_w, left_top_h,
              font_renderer, result.detected);

    DrawSectionLabel(&canvas, content_left, left_bottom_label_y, "Pipeline performance",
                     font_renderer);
    DrawPerformanceDashboard(&canvas,
                             cv::Rect(content_left, left_bottom_panel_y, left_w, left_bottom_h),
                             result, font_renderer);

    DrawSectionLabel(&canvas, right_x, content_top, "Cropped waybill + OCR boxes", font_renderer);
    DrawPanel(&canvas, crop_with_boxes, right_x, right_top_panel_y, right_w, right_top_h,
              font_renderer);

    DrawSectionLabel(&canvas, right_x, right_bottom_label_y, "Recognized text", font_renderer);
    DrawPanel(&canvas, ocr_text_image, right_x, right_bottom_panel_y, right_w, right_bottom_h,
              font_renderer);

    return canvas;
}

cv::Mat MakeMissingDetectionLayout(const cv::Mat& original_bgr, const PipelineResult& result,
                                   const FontRenderer& font_renderer) {
    cv::Mat blank_panel(720, 720, CV_8UC3, kColorPanelBg);
    const std::string message = "Waiting for a waybill...";
    constexpr int kFontSize = 30;
    const FontRenderer::TextMetrics metrics = font_renderer.MeasureText(message, kFontSize);
    const int text_x = std::max(0, (blank_panel.cols - metrics.width) / 2);
    const int text_y = (blank_panel.rows - metrics.height) / 2;
    font_renderer.Draw(&blank_panel, message, {text_x, text_y}, kFontSize, kColorTextMuted);
    return ComposeFinalLayout(original_bgr, blank_panel, blank_panel.clone(), result,
                              font_renderer);
}

std::vector<std::string> CollectDemoImages(const std::string& image_dir) {
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

bool RunPipeline(const Config& config, YoloRuntime* yolo, OcrRuntime* ocr, const cv::Mat& image,
                 PipelineResult* result, std::string* error) {
    *result = {};
    const auto begin = std::chrono::steady_clock::now();
    Detection best_detection;
    bool found_detection = false;
    if (!yolo->InferBest(image, &best_detection, &found_detection, error)) {
        return false;
    }
    result->yolo_timing = yolo->last_timing();
    if (!found_detection) {
        result->overall_latency_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        return true;
    }
    result->detected = true;
    result->detection = best_detection;
    result->crop_rect = ExpandAndClampRect(best_detection, image.size(), config.crop_pad_ratio);
    if (result->crop_rect.width <= 0 || result->crop_rect.height <= 0) {
        result->detected = false;
        result->overall_latency_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        return true;
    }
    result->crop = image(result->crop_rect).clone();
    if (!ocr->Detect(result->crop, &result->boxes, error)) {
        return false;
    }
    if (!ocr->Recognize(result->crop, result->boxes, &result->results, error)) {
        return false;
    }
    result->ocr_timing = ocr->last_timing();
    result->overall_latency_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return true;
}

cv::Mat RenderPipelineResult(const cv::Mat& image, const PipelineResult& result, float drop_score,
                             const FontRenderer& font_renderer) {
    if (!result.detected) {
        return MakeMissingDetectionLayout(image, result, font_renderer);
    }
    const cv::Mat detection_plot = MakeDetectionPlot(image, result.detection);
    const cv::Mat crop_with_boxes = MakeCropWithBoxes(result.crop, result.boxes);
    const cv::Mat ocr_text_image =
        MakeOcrTextImage(result.results, result.boxes, drop_score, font_renderer);
    return ComposeFinalLayout(detection_plot, crop_with_boxes, ocr_text_image, result,
                              font_renderer);
}

void SetLiveError(LiveSharedState* state, const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->error = true;
        state->stop = true;
        state->error_message = message;
    }
    state->capture_cv.notify_all();
    state->frame_cv.notify_all();
    state->detection_cv.notify_all();
    state->render_cv.notify_all();
}

void LogLiveException(const char* worker, const std::string& message) {
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::ofstream out(fs::current_path() / "live_errors.log", std::ios::app);
    if (out.is_open()) {
        out << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S")
            << " [" << worker << "] " << message << "\n";
    }
    std::cerr << "[live-error] [" << worker << "] " << message << "\n";
}

void SetLiveException(LiveSharedState* state, const char* worker, const char* fallback) {
    try {
        throw;
    } catch (const std::exception& e) {
        const std::string message = std::string(fallback) + ": " + e.what();
        LogLiveException(worker, message);
        SetLiveError(state, message);
    } catch (...) {
        LogLiveException(worker, fallback);
        SetLiveError(state, fallback);
    }
}

void StopLiveWorkers(LiveSharedState* state) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->stop = true;
    }
    state->capture_cv.notify_all();
    state->frame_cv.notify_all();
    state->detection_cv.notify_all();
    state->render_cv.notify_all();
}

std::optional<LiveFramePacket> TakeLatestFrameForDetection(LiveSharedState* state) {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->frame_cv.wait(lock,
                         [&] { return state->stop || state->error || state->latest_frame.has_value(); });
    if (state->stop || state->error) {
        return std::nullopt;
    }
    std::optional<LiveFramePacket> packet = std::move(state->latest_frame);
    state->latest_frame.reset();
    return packet;
}

std::optional<LiveDetectionPacket> TakeLatestDetectionForOcr(LiveSharedState* state) {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->detection_cv.wait(
        lock, [&] { return state->stop || state->error || state->latest_detection.has_value(); });
    if (state->stop || state->error) {
        return std::nullopt;
    }
    std::optional<LiveDetectionPacket> packet = std::move(state->latest_detection);
    state->latest_detection.reset();
    return packet;
}

bool RunYoloStageForLive(const Config& config, YoloRuntime* yolo,
                         const LiveFramePacket& frame_packet,
                         DetectionSmoother* smoother,
                         LiveDetectionPacket* detection_packet, std::string* error) {
    *detection_packet = {};
    detection_packet->frame_id = frame_packet.frame_id;
    detection_packet->frame = frame_packet.frame.clone();
    detection_packet->captured_at = frame_packet.captured_at;
    detection_packet->yolo_started_at = std::chrono::steady_clock::now();

    Detection best_detection;
    bool found_detection = false;
    if (!yolo->InferBest(frame_packet.frame, &best_detection, &found_detection, error)) {
        return false;
    }
    detection_packet->yolo_timing = yolo->last_timing();
    detection_packet->detected_at = std::chrono::steady_clock::now();
    if (!found_detection) {
        if (smoother != nullptr) {
            smoother->Reset();
        }
        return true;
    }

    if (smoother != nullptr) {
        best_detection = smoother->Apply(best_detection, config.live_crop_smoothing_alpha);
    }

    detection_packet->detected = true;
    detection_packet->detection = best_detection;
    detection_packet->crop_rect =
        ExpandAndClampRect(best_detection, frame_packet.frame.size(), config.crop_pad_ratio);
    if (detection_packet->crop_rect.width <= 0 || detection_packet->crop_rect.height <= 0) {
        detection_packet->detected = false;
        detection_packet->crop_rect = {};
        return true;
    }
    detection_packet->crop = frame_packet.frame(detection_packet->crop_rect).clone();
    return true;
}

PipelineResult MakeDetectionOnlyPipelineResult(const LiveDetectionPacket& detection_packet) {
    PipelineResult result;
    result.detected = detection_packet.detected;
    result.detection = detection_packet.detection;
    result.crop_rect = detection_packet.crop_rect;
    result.crop = detection_packet.crop.clone();
    result.yolo_timing = detection_packet.yolo_timing;
    result.overall_latency_sec =
        std::chrono::duration<double>(detection_packet.detected_at - detection_packet.captured_at)
            .count();
    return result;
}

bool RunOcrStageForLive(const Config& config, OcrRuntime* ocr,
                        const LiveDetectionPacket& detection_packet, PipelineResult* result,
                        std::string* error) {
    *result = MakeDetectionOnlyPipelineResult(detection_packet);
    if (!detection_packet.detected) {
        return true;
    }
    if (!ocr->Detect(result->crop, &result->boxes, error)) {
        return false;
    }
    if (!ocr->Recognize(result->crop, result->boxes, &result->results, error)) {
        return false;
    }
    result->ocr_timing = ocr->last_timing();
    result->overall_latency_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - detection_packet.captured_at)
            .count();
    return true;
}

void WriteSingleImageJson(const std::string& path, const std::string& image_path,
                          const PipelineResult& result) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"image_path\": \"" << JsonEscape(image_path) << "\",\n";
    out << "  \"detected\": " << (result.detected ? "true" : "false") << ",\n";
    out << "  \"overall_latency_ms\": " << result.overall_latency_sec * 1000.0 << ",\n";
    out << "  \"yolo_latency_ms\": "
        << (result.yolo_timing.preprocess_sec + result.yolo_timing.infer_sec +
            result.yolo_timing.decode_sec) *
               1000.0
        << ",\n";
    out << "  \"ocr_det_latency_ms\": "
        << (result.ocr_timing.det_preprocess_sec + result.ocr_timing.det_infer_sec +
            result.ocr_timing.det_postprocess_sec) *
               1000.0
        << ",\n";
    out << "  \"ocr_rec_latency_ms\": "
        << (result.ocr_timing.rec_crop_sec + result.ocr_timing.rec_preprocess_sec +
            result.ocr_timing.rec_infer_sec + result.ocr_timing.rec_postprocess_sec) *
               1000.0
        << ",\n";
    out << "  \"recognized_texts\": [\n";
    for (size_t i = 0; i < result.results.size(); ++i) {
        out << "    {\"text\": \"" << JsonEscape(result.results[i].text)
            << "\", \"score\": " << result.results[i].score << "}";
        out << (i + 1 == result.results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

double YoloTotalMs(const PipelineResult& result) {
    return (result.yolo_timing.preprocess_sec + result.yolo_timing.infer_sec +
            result.yolo_timing.decode_sec) *
           1000.0;
}

double OcrDetTotalMs(const PipelineResult& result) {
    return (result.ocr_timing.det_preprocess_sec + result.ocr_timing.det_infer_sec +
            result.ocr_timing.det_postprocess_sec) *
           1000.0;
}

double OcrRecTotalMs(const PipelineResult& result) {
    return (result.ocr_timing.rec_crop_sec + result.ocr_timing.rec_preprocess_sec +
            result.ocr_timing.rec_infer_sec + result.ocr_timing.rec_postprocess_sec) *
           1000.0;
}

struct BenchmarkTotals {
    int count = 0;
    double yolo_preprocess_ms = 0.0;
    double yolo_infer_ms = 0.0;
    double yolo_decode_ms = 0.0;
    double ocr_det_preprocess_ms = 0.0;
    double ocr_det_infer_ms = 0.0;
    double ocr_det_postprocess_ms = 0.0;
    double ocr_rec_crop_ms = 0.0;
    double ocr_rec_preprocess_ms = 0.0;
    double ocr_rec_infer_ms = 0.0;
    double ocr_rec_postprocess_ms = 0.0;
    double total_latency_ms = 0.0;
};

void AccumulateBenchmarkTotals(const PipelineResult& result, BenchmarkTotals* totals) {
    ++totals->count;
    totals->yolo_preprocess_ms += result.yolo_timing.preprocess_sec * 1000.0;
    totals->yolo_infer_ms += result.yolo_timing.infer_sec * 1000.0;
    totals->yolo_decode_ms += result.yolo_timing.decode_sec * 1000.0;
    totals->ocr_det_preprocess_ms += result.ocr_timing.det_preprocess_sec * 1000.0;
    totals->ocr_det_infer_ms += result.ocr_timing.det_infer_sec * 1000.0;
    totals->ocr_det_postprocess_ms += result.ocr_timing.det_postprocess_sec * 1000.0;
    totals->ocr_rec_crop_ms += result.ocr_timing.rec_crop_sec * 1000.0;
    totals->ocr_rec_preprocess_ms += result.ocr_timing.rec_preprocess_sec * 1000.0;
    totals->ocr_rec_infer_ms += result.ocr_timing.rec_infer_sec * 1000.0;
    totals->ocr_rec_postprocess_ms += result.ocr_timing.rec_postprocess_sec * 1000.0;
    totals->total_latency_ms += result.overall_latency_sec * 1000.0;
}

void WritePointArrayJson(std::ostream& out, const std::vector<cv::Point2f>& points) {
    out << "[";
    for (size_t i = 0; i < points.size(); ++i) {
        out << "[" << points[i].x << ", " << points[i].y << "]";
        if (i + 1 != points.size()) {
            out << ", ";
        }
    }
    out << "]";
}

void WriteBenchmarkRecordJson(std::ostream& out, const std::string& image_path, size_t image_index,
                              const PipelineResult& result) {
    out << "    {\n";
    out << "      \"image_path\": \"" << JsonEscape(image_path) << "\",\n";
    out << "      \"image_index\": " << image_index << ",\n";
    out << "      \"detected\": " << (result.detected ? "true" : "false") << ",\n";
    out << "      \"waybill_detection\": ";
    if (result.detected) {
        out << "{\"x1\": " << result.detection.x1 << ", \"y1\": " << result.detection.y1
            << ", \"x2\": " << result.detection.x2 << ", \"y2\": " << result.detection.y2
            << ", \"score\": " << result.detection.score
            << ", \"class_id\": " << result.detection.cls << "}";
    } else {
        out << "null";
    }
    out << ",\n";
    out << "      \"crop_rect\": {\"x\": " << result.crop_rect.x
        << ", \"y\": " << result.crop_rect.y << ", \"width\": " << result.crop_rect.width
        << ", \"height\": " << result.crop_rect.height << "},\n";
    out << "      \"ocr_box_count\": " << result.boxes.size() << ",\n";
    out << "      \"ocr_text_count\": " << result.results.size() << ",\n";
    out << "      \"timings_ms\": {\n";
    out << "        \"yolo_preprocess\": " << result.yolo_timing.preprocess_sec * 1000.0 << ",\n";
    out << "        \"yolo_infer\": " << result.yolo_timing.infer_sec * 1000.0 << ",\n";
    out << "        \"yolo_decode\": " << result.yolo_timing.decode_sec * 1000.0 << ",\n";
    out << "        \"yolo_total\": " << YoloTotalMs(result) << ",\n";
    out << "        \"ocr_det_preprocess\": " << result.ocr_timing.det_preprocess_sec * 1000.0
        << ",\n";
    out << "        \"ocr_det_infer\": " << result.ocr_timing.det_infer_sec * 1000.0 << ",\n";
    out << "        \"ocr_det_postprocess\": " << result.ocr_timing.det_postprocess_sec * 1000.0
        << ",\n";
    out << "        \"ocr_det_total\": " << OcrDetTotalMs(result) << ",\n";
    out << "        \"ocr_rec_crop\": " << result.ocr_timing.rec_crop_sec * 1000.0 << ",\n";
    out << "        \"ocr_rec_preprocess\": " << result.ocr_timing.rec_preprocess_sec * 1000.0
        << ",\n";
    out << "        \"ocr_rec_infer\": " << result.ocr_timing.rec_infer_sec * 1000.0 << ",\n";
    out << "        \"ocr_rec_postprocess\": " << result.ocr_timing.rec_postprocess_sec * 1000.0
        << ",\n";
    out << "        \"ocr_rec_total\": " << OcrRecTotalMs(result) << ",\n";
    out << "        \"total_latency\": " << result.overall_latency_sec * 1000.0 << "\n";
    out << "      },\n";
    out << "      \"ocr_boxes\": [\n";
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        out << "        {\"index\": " << i << ", \"score\": " << result.boxes[i].score
            << ", \"points\": ";
        WritePointArrayJson(out, result.boxes[i].points);
        out << "}";
        out << (i + 1 == result.boxes.size() ? "\n" : ",\n");
    }
    out << "      ],\n";
    out << "      \"recognized_texts\": [\n";
    for (size_t i = 0; i < result.results.size(); ++i) {
        out << "        {\"index\": " << i << ", \"text\": \"" << JsonEscape(result.results[i].text)
            << "\", \"score\": " << result.results[i].score
            << ", \"box_score\": " << result.results[i].box.score << ", \"points\": ";
        WritePointArrayJson(out, result.results[i].box.points);
        out << "}";
        out << (i + 1 == result.results.size() ? "\n" : ",\n");
    }
    out << "      ]\n";
    out << "    }";
}

void WriteBenchmarkAveragesJson(std::ostream& out, const BenchmarkTotals& totals) {
    const double count = std::max(1, totals.count);
    const double avg_yolo_preprocess = totals.yolo_preprocess_ms / count;
    const double avg_yolo_infer = totals.yolo_infer_ms / count;
    const double avg_yolo_decode = totals.yolo_decode_ms / count;
    const double avg_ocr_det_preprocess = totals.ocr_det_preprocess_ms / count;
    const double avg_ocr_det_infer = totals.ocr_det_infer_ms / count;
    const double avg_ocr_det_postprocess = totals.ocr_det_postprocess_ms / count;
    const double avg_ocr_rec_crop = totals.ocr_rec_crop_ms / count;
    const double avg_ocr_rec_preprocess = totals.ocr_rec_preprocess_ms / count;
    const double avg_ocr_rec_infer = totals.ocr_rec_infer_ms / count;
    const double avg_ocr_rec_postprocess = totals.ocr_rec_postprocess_ms / count;
    out << "  \"averages_ms\": {\n";
    out << "    \"yolo_preprocess\": " << avg_yolo_preprocess << ",\n";
    out << "    \"yolo_infer\": " << avg_yolo_infer << ",\n";
    out << "    \"yolo_decode\": " << avg_yolo_decode << ",\n";
    out << "    \"yolo_total\": " << (avg_yolo_preprocess + avg_yolo_infer + avg_yolo_decode)
        << ",\n";
    out << "    \"ocr_det_preprocess\": " << avg_ocr_det_preprocess << ",\n";
    out << "    \"ocr_det_infer\": " << avg_ocr_det_infer << ",\n";
    out << "    \"ocr_det_postprocess\": " << avg_ocr_det_postprocess << ",\n";
    out << "    \"ocr_det_total\": "
        << (avg_ocr_det_preprocess + avg_ocr_det_infer + avg_ocr_det_postprocess) << ",\n";
    out << "    \"ocr_rec_crop\": " << avg_ocr_rec_crop << ",\n";
    out << "    \"ocr_rec_preprocess\": " << avg_ocr_rec_preprocess << ",\n";
    out << "    \"ocr_rec_infer\": " << avg_ocr_rec_infer << ",\n";
    out << "    \"ocr_rec_postprocess\": " << avg_ocr_rec_postprocess << ",\n";
    out << "    \"ocr_rec_total\": "
        << (avg_ocr_rec_crop + avg_ocr_rec_preprocess + avg_ocr_rec_infer + avg_ocr_rec_postprocess)
        << ",\n";
    out << "    \"total_latency\": " << totals.total_latency_ms / count << "\n";
    out << "  }\n";
}

bool EnsureParentDir(const std::string& path, std::string* error) {
    const fs::path parent = fs::path(path).parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
        *error = "failed to create directory: " + parent.string();
        return false;
    }
    return true;
}

bool EnsureDir(const std::string& path, std::string* error) {
    std::error_code ec;
    fs::create_directories(path, ec);
    if (ec) {
        *error = "failed to create directory: " + path;
        return false;
    }
    return true;
}

bool ReadProbeFrame(cv::VideoCapture* cap, cv::Mat* frame) {
    frame->release();
    for (int attempt = 0; attempt < 5; ++attempt) {
        *cap >> *frame;
        if (!frame->empty()) {
            return true;
        }
    }
    return false;
}

void ApplyCameraFocus(const Config& config, cv::VideoCapture* cap) {
    if (config.camera_focus >= 0) {
        const bool af_off_ok = cap->set(cv::CAP_PROP_AUTOFOCUS, 0.0);
        const bool focus_ok =
            cap->set(cv::CAP_PROP_FOCUS, static_cast<double>(config.camera_focus));
        const double read_af = cap->get(cv::CAP_PROP_AUTOFOCUS);
        const double read_focus = cap->get(cv::CAP_PROP_FOCUS);
        std::cout << "Camera focus manual requested=" << config.camera_focus
                  << " af_off_set_ok=" << (af_off_ok ? "yes" : "no")
                  << " focus_set_ok=" << (focus_ok ? "yes" : "no")
                  << " read AUTOFOCUS=" << read_af << " FOCUS=" << read_focus << "\n";
        return;
    }
    const bool af_on_ok = cap->set(cv::CAP_PROP_AUTOFOCUS, 1.0);
    const double read_af = cap->get(cv::CAP_PROP_AUTOFOCUS);
    std::cout << "Camera autofocus requested"
              << " set_ok=" << (af_on_ok ? "yes" : "no")
              << " read AUTOFOCUS=" << read_af << "\n";
}

void ConfigureCameraPresenceProbe(const Config& config, cv::VideoCapture* cap) {
    cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap->set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (config.camera_fps > 0.0) {
        cap->set(cv::CAP_PROP_FPS, config.camera_fps);
    }
    cap->set(cv::CAP_PROP_FRAME_WIDTH, kCameraPresenceProbeWidth);
    cap->set(cv::CAP_PROP_FRAME_HEIGHT, kCameraPresenceProbeHeight);
}

void ConfigureCameraCapture(const Config& config, cv::VideoCapture* cap) {
    cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap->set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (config.camera_fps > 0.0) {
        cap->set(cv::CAP_PROP_FPS, config.camera_fps);
    }
    if (config.camera_width > 0 && config.camera_height > 0) {
        cap->set(cv::CAP_PROP_FRAME_WIDTH, config.camera_width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, config.camera_height);
        ApplyCameraFocus(config, cap);
        return;
    }

    const std::vector<cv::Size> candidate_resolutions = {
        {3840, 2160}, {2560, 1440}, {1920, 1080}, {1600, 1200}, {1280, 1024},
        {1280, 720},  {1024, 768},  {800, 600},   {640, 480},   {640, 360}};
    cv::Size best_size;
    double best_area = 0.0;
    for (const auto& candidate : candidate_resolutions) {
        cap->set(cv::CAP_PROP_FRAME_WIDTH, candidate.width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, candidate.height);
        cv::Mat probe_frame;
        if (!ReadProbeFrame(cap, &probe_frame)) {
            continue;
        }
        const cv::Size actual_size = probe_frame.size();
        const double actual_area = static_cast<double>(actual_size.width) * actual_size.height;
        if (actual_area > best_area) {
            best_area = actual_area;
            best_size = actual_size;
        }
    }
    if (best_size.width > 0 && best_size.height > 0) {
        cap->set(cv::CAP_PROP_FRAME_WIDTH, best_size.width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, best_size.height);
        std::cout << "Using camera resolution: " << best_size.width << "x" << best_size.height
                  << "\n";
    }
    ApplyCameraFocus(config, cap);
}

struct CameraBackendCandidate {
    int api;
    const char* name;
};

std::vector<CameraBackendCandidate> CameraBackendOrder(const std::string& requested) {
    if (requested == "dshow") {
        return {{cv::CAP_DSHOW, "DSHOW"}};
    }
    if (requested == "v4l2") {
        return {{cv::CAP_V4L2, "V4L2"}};
    }
    if (requested == "any") {
        return {{cv::CAP_ANY, "ANY"}};
    }
#ifdef _WIN32
    return {{cv::CAP_DSHOW, "DSHOW"}, {cv::CAP_ANY, "ANY"}};
#elif defined(__linux__)
    return {{cv::CAP_V4L2, "V4L2"}, {cv::CAP_ANY, "ANY"}};
#else
    return {{cv::CAP_ANY, "ANY"}};
#endif
}

bool TryOpenCameraIndex(const Config& config, int camera_index, cv::VideoCapture* cap,
                        bool log_details) {
    const auto backends = CameraBackendOrder(config.camera_backend);
    for (const auto& backend : backends) {
        cap->release();
        bool opened = false;
        try {
            opened = cap->open(camera_index, backend.api);
        } catch (const cv::Exception& e) {
            if (log_details) {
                std::cout << "Backend " << backend.name << " open threw: " << e.what() << "\n";
            }
            opened = false;
        }
        if (!opened || !cap->isOpened()) {
            continue;
        }

        try {
            ConfigureCameraPresenceProbe(config, cap);
        } catch (const cv::Exception& e) {
            if (log_details) {
                std::cout << "Backend " << backend.name << " presence-probe config threw: "
                          << e.what() << "\n";
            }
            cap->release();
            continue;
        }
        cv::Mat presence_probe_frame;
        bool presence_ok = false;
        try {
            presence_ok = ReadProbeFrame(cap, &presence_probe_frame);
        } catch (const cv::Exception& e) {
            if (log_details) {
                std::cout << "Backend " << backend.name << " presence-probe read threw: "
                          << e.what() << "\n";
            }
            presence_ok = false;
        }
        if (!presence_ok) {
            cap->release();
            continue;
        }

        try {
            ConfigureCameraCapture(config, cap);
        } catch (const cv::Exception& e) {
            if (log_details) {
                std::cout << "Backend " << backend.name << " capture config threw: " << e.what()
                          << "\n";
            }
            cap->release();
            continue;
        }

        cv::Mat probe_frame;
        bool probe_ok = false;
        try {
            probe_ok = ReadProbeFrame(cap, &probe_frame);
        } catch (const cv::Exception& e) {
            if (log_details) {
                std::cout << "Backend " << backend.name << " capture probe read threw: "
                          << e.what() << "\n";
            }
            probe_ok = false;
        }
        if (!probe_ok) {
            cap->release();
            continue;
        }

        if (log_details) {
            std::cout << "Opened camera index " << camera_index << " via " << backend.name << " at "
                      << probe_frame.cols << "x" << probe_frame.rows << "\n";
        }
        return true;
    }
    return false;
}

bool TryOpenLiveCaptureOnce(const Config& config, cv::VideoCapture* cap,
                            int* opened_camera_index, bool log_details) {
    if (config.camera_index >= 0) {
        if (log_details) {
            std::cout << "Trying camera index: " << config.camera_index << "\n";
        }
        if (TryOpenCameraIndex(config, config.camera_index, cap, log_details)) {
            *opened_camera_index = config.camera_index;
            return true;
        }
        return false;
    }
    for (int camera_index = 0; camera_index <= kMaxCameraAutoSearchIndex; ++camera_index) {
        if (log_details) {
            std::cout << "Trying camera index: " << camera_index << "\n";
        }
        if (TryOpenCameraIndex(config, camera_index, cap, log_details)) {
            *opened_camera_index = camera_index;
            return true;
        }
    }
    return false;
}

cv::Mat MakeLiveDisplayFrame(const cv::Mat& layout, double display_scale);

cv::Mat MakeCameraWaitingLayout(const std::string& primary, const std::string& secondary,
                                const FontRenderer& font_renderer) {
    cv::Mat canvas(kLayoutHeight, kLayoutWidth, CV_8UC3, kColorCanvasBg);
    DrawHeader(&canvas, font_renderer);
    constexpr int kPrimaryFontSize = 64;
    constexpr int kSecondaryFontSize = 32;
    const FontRenderer::TextMetrics primary_metrics =
        font_renderer.MeasureText(primary, kPrimaryFontSize);
    const FontRenderer::TextMetrics secondary_metrics =
        font_renderer.MeasureText(secondary, kSecondaryFontSize);
    const int primary_x = std::max(0, (kLayoutWidth - primary_metrics.width) / 2);
    const int primary_y = kLayoutHeight / 2 - primary_metrics.height;
    font_renderer.Draw(&canvas, primary, {primary_x, primary_y}, kPrimaryFontSize,
                       kColorTextPrimary);
    if (!secondary.empty()) {
        const int secondary_x = std::max(0, (kLayoutWidth - secondary_metrics.width) / 2);
        const int secondary_y = primary_y + primary_metrics.height + 32;
        font_renderer.Draw(&canvas, secondary, {secondary_x, secondary_y}, kSecondaryFontSize,
                           kColorTextMuted);
    }
    return canvas;
}

bool WaitForCameraReady(const Config& config, cv::VideoCapture* cap, int* opened_camera_index,
                        const std::string& window_name, double display_scale,
                        const FontRenderer& font_renderer, const std::string& primary_message,
                        const std::string& secondary_message,
                        const std::string& configuring_primary_message,
                        const std::string& configuring_secondary_message) {
    const cv::Mat waiting_layout =
        MakeCameraWaitingLayout(primary_message, secondary_message, font_renderer);
    const cv::Mat waiting_display = MakeLiveDisplayFrame(waiting_layout, display_scale);
    const cv::Mat configuring_layout = MakeCameraWaitingLayout(
        configuring_primary_message, configuring_secondary_message, font_renderer);
    const cv::Mat configuring_display = MakeLiveDisplayFrame(configuring_layout, display_scale);
    constexpr auto kCameraRetryInterval = std::chrono::seconds(2);
    std::cout << "Waiting for USB camera... (automatic retry every 2 seconds, ESC/q to quit)\n";
    int attempt = 0;
    auto next_attempt = std::chrono::steady_clock::now();
    while (true) {
        cv::imshow(window_name, waiting_display);
        const int key = cv::waitKey(50);
        if (key == 27 || key == 'q' || key == 'Q') {
            return false;
        }
        const auto now = std::chrono::steady_clock::now();
        const bool retry_requested = key == 10 || key == 13;
        if (!retry_requested && now < next_attempt) {
            continue;
        }
        next_attempt = now + kCameraRetryInterval;
        ++attempt;
        std::cout << "Camera detection attempt #" << attempt << "\n";
        cv::imshow(window_name, configuring_display);
        cv::waitKey(1);
        if (TryOpenLiveCaptureOnce(config, cap, opened_camera_index, true)) {
            std::cout << "Using camera index: " << *opened_camera_index << "\n";
            return true;
        }
    }
}

cv::Mat MakeLiveDisplayFrame(const cv::Mat& layout, double display_scale) {
    if (std::abs(display_scale - 1.0) < 1e-6) {
        return layout;
    }
    const int width = std::max(1, static_cast<int>(std::round(layout.cols * display_scale)));
    const int height = std::max(1, static_cast<int>(std::round(layout.rows * display_scale)));
    cv::Mat display;
    cv::resize(layout, display, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    return display;
}

double MsBetween(std::chrono::steady_clock::time_point begin,
                 std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void AccumulateLiveYoloPerf(const LiveDetectionPacket& packet, LivePerfStats* perf) {
    ++perf->yolo_window;
    perf->yolo_wait_ms += MsBetween(packet.captured_at, packet.yolo_started_at);
    perf->yolo_preprocess_ms += packet.yolo_timing.preprocess_sec * 1000.0;
    perf->yolo_infer_ms += packet.yolo_timing.infer_sec * 1000.0;
    perf->yolo_decode_ms += packet.yolo_timing.decode_sec * 1000.0;
    perf->yolo_e2e_ms += MsBetween(packet.captured_at, packet.detected_at);
}

void AccumulateLiveOcrPerf(const LiveDetectionPacket& packet,
                           std::chrono::steady_clock::time_point ocr_started_at,
                           std::chrono::steady_clock::time_point ocr_done_at,
                           const OcrTiming& timing, LivePerfStats* perf) {
    ++perf->ocr_window;
    perf->ocr_wait_ms += MsBetween(packet.detected_at, ocr_started_at);
    perf->ocr_det_preprocess_ms += timing.det_preprocess_sec * 1000.0;
    perf->ocr_det_infer_ms += timing.det_infer_sec * 1000.0;
    perf->ocr_det_postprocess_ms += timing.det_postprocess_sec * 1000.0;
    perf->ocr_rec_crop_ms += timing.rec_crop_sec * 1000.0;
    perf->ocr_rec_preprocess_ms += timing.rec_preprocess_sec * 1000.0;
    perf->ocr_rec_infer_ms += timing.rec_infer_sec * 1000.0;
    perf->ocr_rec_postprocess_ms += timing.rec_postprocess_sec * 1000.0;
    perf->ocr_e2e_ms += MsBetween(packet.captured_at, ocr_done_at);
    perf->ocr_rec_box_count_sum += timing.rec_box_count;
    perf->ocr_rec_crop_count_sum += timing.rec_crop_count;
    perf->ocr_rec_batch_size_sum += timing.rec_max_batch_size;
    perf->ocr_rec_bucket_width_sum += timing.rec_max_bucket_width;
    perf->ocr_rec_box_count_max = std::max(perf->ocr_rec_box_count_max, timing.rec_box_count);
    perf->ocr_rec_crop_count_max = std::max(perf->ocr_rec_crop_count_max, timing.rec_crop_count);
    perf->ocr_rec_batch_size_max =
        std::max(perf->ocr_rec_batch_size_max, timing.rec_max_batch_size);
    perf->ocr_rec_bucket_width_max =
        std::max(perf->ocr_rec_bucket_width_max, timing.rec_max_bucket_width);

    auto lane_int = [](const std::vector<int>& values, size_t lane) {
        return lane < values.size() ? values[lane] : 0;
    };
    auto lane_sec = [](const std::vector<double>& values, size_t lane) {
        return lane < values.size() ? values[lane] : 0.0;
    };
    const double lane0_infer_ms = lane_sec(timing.rec_lane_infer_secs, 0) * 1000.0;
    const double lane1_infer_ms = lane_sec(timing.rec_lane_infer_secs, 1) * 1000.0;
    const int lane0_batch_size = lane_int(timing.rec_lane_batch_sizes, 0);
    const int lane1_batch_size = lane_int(timing.rec_lane_batch_sizes, 1);
    const int lane0_bucket_width = lane_int(timing.rec_lane_max_bucket_widths, 0);
    const int lane1_bucket_width = lane_int(timing.rec_lane_max_bucket_widths, 1);
    perf->ocr_rec_lane0_infer_ms += lane0_infer_ms;
    perf->ocr_rec_lane1_infer_ms += lane1_infer_ms;
    perf->ocr_rec_lane0_infer_max_ms =
        std::max(perf->ocr_rec_lane0_infer_max_ms, lane0_infer_ms);
    perf->ocr_rec_lane1_infer_max_ms =
        std::max(perf->ocr_rec_lane1_infer_max_ms, lane1_infer_ms);
    perf->ocr_rec_lane0_batch_size_sum += lane0_batch_size;
    perf->ocr_rec_lane1_batch_size_sum += lane1_batch_size;
    perf->ocr_rec_lane0_bucket_width_sum += lane0_bucket_width;
    perf->ocr_rec_lane1_bucket_width_sum += lane1_bucket_width;
    perf->ocr_rec_lane0_batch_size_max =
        std::max(perf->ocr_rec_lane0_batch_size_max, lane0_batch_size);
    perf->ocr_rec_lane1_batch_size_max =
        std::max(perf->ocr_rec_lane1_batch_size_max, lane1_batch_size);
    perf->ocr_rec_lane0_bucket_width_max =
        std::max(perf->ocr_rec_lane0_bucket_width_max, lane0_bucket_width);
    perf->ocr_rec_lane1_bucket_width_max =
        std::max(perf->ocr_rec_lane1_bucket_width_max, lane1_bucket_width);
}

void MaybePrintLivePerfLog(const Config& config, LiveSharedState* state,
                           std::chrono::steady_clock::time_point now) {
    if (!config.live_perf_log) {
        return;
    }
    LivePerfStats snapshot;
    int64_t captured_frames = 0;
    int64_t yolo_frames = 0;
    int64_t ocr_frames = 0;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->perf.window_start.time_since_epoch().count() == 0) {
            state->perf.window_start = now;
            return;
        }
        const double elapsed = std::chrono::duration<double>(now - state->perf.window_start).count();
        if (elapsed < config.live_perf_log_interval) {
            return;
        }
        snapshot = state->perf;
        captured_frames = state->captured_frames;
        yolo_frames = state->yolo_frames;
        ocr_frames = state->ocr_frames;
        state->perf = {};
        state->perf.window_start = now;
    }

    const double elapsed = std::max(1e-9, std::chrono::duration<double>(now - snapshot.window_start).count());
    auto avg = [](double sum, int64_t count) { return count > 0 ? sum / count : 0.0; };
    const double yolo_total_ms = avg(snapshot.yolo_preprocess_ms + snapshot.yolo_infer_ms +
                                         snapshot.yolo_decode_ms,
                                     snapshot.yolo_window);
    const double ocr_det_ms = avg(snapshot.ocr_det_preprocess_ms + snapshot.ocr_det_infer_ms +
                                      snapshot.ocr_det_postprocess_ms,
                                  snapshot.ocr_window);
    const double ocr_rec_ms = avg(snapshot.ocr_rec_crop_ms + snapshot.ocr_rec_preprocess_ms +
                                      snapshot.ocr_rec_infer_ms + snapshot.ocr_rec_postprocess_ms,
                                  snapshot.ocr_window);
    const double rec_box_count_avg = avg(snapshot.ocr_rec_box_count_sum, snapshot.ocr_window);
    const double rec_crop_count_avg = avg(snapshot.ocr_rec_crop_count_sum, snapshot.ocr_window);
    const double rec_batch_size_avg = avg(snapshot.ocr_rec_batch_size_sum, snapshot.ocr_window);
    const double rec_bucket_width_avg = avg(snapshot.ocr_rec_bucket_width_sum, snapshot.ocr_window);
    const double lane0_infer_avg_ms = avg(snapshot.ocr_rec_lane0_infer_ms, snapshot.ocr_window);
    const double lane1_infer_avg_ms = avg(snapshot.ocr_rec_lane1_infer_ms, snapshot.ocr_window);
    const double lane0_batch_avg = avg(snapshot.ocr_rec_lane0_batch_size_sum, snapshot.ocr_window);
    const double lane1_batch_avg = avg(snapshot.ocr_rec_lane1_batch_size_sum, snapshot.ocr_window);
    const double lane0_bucket_avg = avg(snapshot.ocr_rec_lane0_bucket_width_sum, snapshot.ocr_window);
    const double lane1_bucket_avg = avg(snapshot.ocr_rec_lane1_bucket_width_sum, snapshot.ocr_window);

    std::cout << std::fixed << std::setprecision(1)
              << "[live-perf] fps cap=" << (snapshot.captured_window / elapsed)
              << " yolo=" << (snapshot.yolo_window / elapsed)
              << " ocr=" << (snapshot.ocr_window / elapsed)
              << " render=" << (snapshot.render_window / elapsed)
              << " | avg_ms cap_read=" << avg(snapshot.capture_read_ms, snapshot.captured_window)
              << " wait_yolo=" << avg(snapshot.yolo_wait_ms, snapshot.yolo_window)
              << " yolo=" << yolo_total_ms
              << " wait_ocr=" << avg(snapshot.ocr_wait_ms, snapshot.ocr_window)
              << " ocr_det=" << ocr_det_ms << " ocr_rec=" << ocr_rec_ms
              << " e2e_ocr=" << avg(snapshot.ocr_e2e_ms, snapshot.ocr_window)
              << " render=" << avg(snapshot.render_ms, snapshot.render_window)
              << " display=" << avg(snapshot.display_ms, snapshot.render_window)
              << " | rec_meta boxes=" << rec_box_count_avg << "/"
              << snapshot.ocr_rec_box_count_max << " crops=" << rec_crop_count_avg << "/"
              << snapshot.ocr_rec_crop_count_max << " batch=" << rec_batch_size_avg << "/"
              << snapshot.ocr_rec_batch_size_max << " bucket=" << rec_bucket_width_avg << "/"
              << snapshot.ocr_rec_bucket_width_max
              << " | rec_lane l0_infer=" << lane0_infer_avg_ms << "/"
              << snapshot.ocr_rec_lane0_infer_max_ms << " l0_batch=" << lane0_batch_avg << "/"
              << snapshot.ocr_rec_lane0_batch_size_max << " l0_bucket=" << lane0_bucket_avg
              << "/" << snapshot.ocr_rec_lane0_bucket_width_max << " l1_infer="
              << lane1_infer_avg_ms << "/" << snapshot.ocr_rec_lane1_infer_max_ms
              << " l1_batch=" << lane1_batch_avg << "/"
              << snapshot.ocr_rec_lane1_batch_size_max << " l1_bucket=" << lane1_bucket_avg
              << "/" << snapshot.ocr_rec_lane1_bucket_width_max
              << " | frames cap=" << captured_frames << " yolo=" << yolo_frames
              << " ocr=" << ocr_frames << "\n";
}

bool RunLiveMode(const Config& config, YoloRuntime* yolo, OcrRuntime* ocr,
                 const FontRenderer& font_renderer, std::string* error) {
    cv::VideoWriter writer;
    if (!config.output_video_path.empty()) {
        writer.open(config.output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    config.fps, cv::Size(kLayoutWidth, kLayoutHeight));
        if (!writer.isOpened()) {
            *error = "failed to open live output video: " + config.output_video_path;
            return false;
        }
    }

    cv::namedWindow(config.window_name, cv::WINDOW_GUI_NORMAL);
    cv::moveWindow(config.window_name, 0, 0);
    if (config.fullscreen) {
        cv::setWindowProperty(config.window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    } else {
        cv::resizeWindow(config.window_name,
                         cv::Size(std::max(1, static_cast<int>(std::round(kLayoutWidth * config.display_scale))),
                                  std::max(1, static_cast<int>(std::round(kLayoutHeight * config.display_scale)))));
    }

    cv::VideoCapture cap;
    int opened_camera_index = -1;
    if (!WaitForCameraReady(config, &cap, &opened_camera_index, config.window_name,
                            config.display_scale, font_renderer,
                            "USB 카메라 탐지중...",
                            "2초마다 자동으로 재연결을 시도합니다",
                            "USB 카메라 설정중...",
                            "잠시만 기다려주세요")) {
        cv::destroyWindow(config.window_name);
        return true;
    }

    LiveSharedState live_state;
    std::thread capture_worker([&] {
        try {
        constexpr int kEmptyFrameThreshold = 10;
        constexpr double kSlowReadLogThresholdMs = 500.0;
        constexpr auto kCameraReadLogInterval = std::chrono::seconds(30);
        auto last_camera_read_log = std::chrono::steady_clock::now() - kCameraReadLogInterval;
        auto last_slow_read_log = std::chrono::steady_clock::now() - kCameraReadLogInterval;
        int consecutive_empty = 0;
        while (true) {
            {
                std::lock_guard<std::mutex> lock(live_state.mutex);
                if (live_state.stop || live_state.error) {
                    break;
                }
            }

            cv::Mat frame;
            const auto capture_read_begin = std::chrono::steady_clock::now();
            try {
                cap >> frame;
            } catch (const cv::Exception& e) {
                frame.release();
                const auto read_error_now = std::chrono::steady_clock::now();
                if (read_error_now - last_camera_read_log >= kCameraReadLogInterval) {
                    last_camera_read_log = read_error_now;
                    std::cout << "Camera read threw: " << e.what() << "\n";
                }
            }
            const auto capture_read_end = std::chrono::steady_clock::now();
            const double capture_read_ms = MsBetween(capture_read_begin, capture_read_end);

            if (frame.empty()) {
                ++consecutive_empty;
            } else {
                consecutive_empty = 0;
            }
            if (capture_read_ms > kSlowReadLogThresholdMs &&
                capture_read_end - last_slow_read_log >= kCameraReadLogInterval) {
                last_slow_read_log = capture_read_end;
                std::cout << "Camera read is slow (" << capture_read_ms
                          << " ms); keeping capture open.\n";
            }

            // Low-light auto-exposure can legitimately make a USB camera read take longer
            // than one nominal frame period. Do not turn slow reads into a disconnect: only
            // an unopened capture or repeated empty frames indicates a camera loss here.
            const bool disconnected = !cap.isOpened() || consecutive_empty >= kEmptyFrameThreshold;
            if (disconnected) {
                {
                    std::lock_guard<std::mutex> lock(live_state.mutex);
                    if (live_state.stop || live_state.error) {
                        break;
                    }
                    live_state.camera_disconnected = true;
                }
                live_state.capture_cv.notify_all();
                constexpr auto kCameraRetryInterval = std::chrono::seconds(2);
                std::cout << "Camera disconnected; retrying automatically every 2 seconds.\n";
                cap.release();
                int reopened_index = -1;
                int reconnect_attempt = 0;
                while (true) {
                    std::unique_lock<std::mutex> lock(live_state.mutex);
                    live_state.capture_cv.wait_for(lock, kCameraRetryInterval, [&] {
                        return live_state.stop || live_state.error ||
                               live_state.camera_retry_requested;
                    });
                    if (live_state.stop || live_state.error) {
                        break;
                    }
                    live_state.camera_retry_requested = false;
                    lock.unlock();
                    ++reconnect_attempt;
                    std::cout << "Camera reconnect attempt #" << reconnect_attempt << "\n";
                    if (TryOpenLiveCaptureOnce(config, &cap, &reopened_index, true)) {
                        {
                            std::lock_guard<std::mutex> state_lock(live_state.mutex);
                            live_state.camera_disconnected = false;
                        }
                        std::cout << "Camera reconnected on index: " << reopened_index << "\n";
                        live_state.capture_cv.notify_all();
                        consecutive_empty = 0;
                        break;
                    }
                    std::cout << "Camera retry failed; will retry automatically.\n";
                }
                continue;
            }

            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            const auto captured_at = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(live_state.mutex);
                if (live_state.stop || live_state.error) {
                    break;
                }
                ++live_state.captured_frames;
                ++live_state.perf.captured_window;
                live_state.perf.capture_read_ms += MsBetween(capture_read_begin, capture_read_end);
                LiveFramePacket packet{live_state.captured_frames, frame.clone(), captured_at};
                live_state.latest_capture_frame = LiveFramePacket{packet.frame_id, packet.frame.clone(),
                                                                  packet.captured_at};
                live_state.latest_frame = std::move(packet);
            }
            live_state.capture_cv.notify_one();
            live_state.frame_cv.notify_one();
        }
        } catch (...) {
            SetLiveException(&live_state, "capture", "uncaught exception in capture worker");
        }
    });

    DetectionSmoother detection_smoother;
    OcrStabilizer ocr_stabilizer;
    ocr_stabilizer.enabled = config.live_ocr_stabilizer;
    ocr_stabilizer.window_size = config.live_ocr_stabilizer_window;
    ocr_stabilizer.min_agreement =
        std::min(config.live_ocr_stabilizer_min_agreement, config.live_ocr_stabilizer_window);
    ocr_stabilizer.freeze_threshold = config.live_ocr_stabilizer_freeze_frames;
    ocr_stabilizer.snapshot_iou_threshold = config.live_ocr_stabilizer_freeze_iou;
    ocr_stabilizer.snapshot_freeze_fraction = config.live_ocr_stabilizer_freeze_fraction;
    ocr_stabilizer.snapshot_min_slots = config.live_ocr_stabilizer_freeze_min_slots;
    ocr_stabilizer.snapshot_max_frames = config.live_ocr_stabilizer_freeze_max_frames;
    ocr_stabilizer.snapshot_divergence_threshold = config.live_ocr_stabilizer_freeze_divergence;
    ocr_stabilizer.detection_miss_grace = config.live_ocr_stabilizer_detection_miss_grace;

    std::thread detection_worker([&] {
        try {
        while (true) {
            std::optional<LiveFramePacket> frame_packet = TakeLatestFrameForDetection(&live_state);
            if (!frame_packet.has_value()) {
                break;
            }
            LiveDetectionPacket detection_packet;
            std::string worker_error;
            if (!RunYoloStageForLive(config, yolo, *frame_packet, &detection_smoother,
                                     &detection_packet, &worker_error)) {
                SetLiveError(&live_state, worker_error);
                break;
            }

            PipelineResult detection_result = MakeDetectionOnlyPipelineResult(detection_packet);
            {
                std::lock_guard<std::mutex> lock(live_state.mutex);
                if (live_state.stop || live_state.error) {
                    break;
                }
                ++live_state.yolo_frames;
                AccumulateLiveYoloPerf(detection_packet, &live_state.perf);
                live_state.latest_detection_render.frame_id = detection_packet.frame_id;
                live_state.latest_detection_render.frame = detection_packet.frame.clone();
                live_state.latest_detection_render.result = std::move(detection_result);
                live_state.latest_detection_render.has_result = true;
                live_state.latest_detection = std::move(detection_packet);
            }
            live_state.detection_cv.notify_one();
        }
        } catch (...) {
            SetLiveException(&live_state, "yolo", "uncaught exception in YOLO worker");
        }
    });

    std::thread ocr_worker([&] {
        try {
        while (true) {
            std::optional<LiveDetectionPacket> detection_packet = TakeLatestDetectionForOcr(&live_state);
            if (!detection_packet.has_value()) {
                break;
            }
            PipelineResult result;
            std::string worker_error;
            const auto ocr_started_at = std::chrono::steady_clock::now();
            if (!RunOcrStageForLive(config, ocr, *detection_packet, &result, &worker_error)) {
                SetLiveError(&live_state, worker_error);
                break;
            }
            const auto ocr_done_at = std::chrono::steady_clock::now();
            if (result.detected) {
                std::vector<TextBox> stabilized_boxes;
                std::vector<OcrResult> stabilized_results;
                ocr_stabilizer.Apply(result.boxes, result.results, result.crop_rect,
                                     &stabilized_boxes, &stabilized_results);
                result.boxes = std::move(stabilized_boxes);
                result.results = std::move(stabilized_results);
            } else {
                ocr_stabilizer.NoteMissedDetection();
            }
            {
                std::lock_guard<std::mutex> lock(live_state.mutex);
                if (live_state.stop || live_state.error) {
                    break;
                }
                ++live_state.ocr_frames;
                AccumulateLiveOcrPerf(*detection_packet, ocr_started_at, ocr_done_at,
                                      result.ocr_timing, &live_state.perf);
                live_state.latest_ocr_render.frame_id = detection_packet->frame_id;
                live_state.latest_ocr_render.frame = detection_packet->frame.clone();
                live_state.latest_ocr_render.result = std::move(result);
                live_state.latest_ocr_render.has_result = true;
            }
        }
        } catch (...) {
            SetLiveException(&live_state, "ocr", "uncaught exception in OCR worker");
        }
    });

    auto join_workers = [&] {
        StopLiveWorkers(&live_state);
        if (capture_worker.joinable()) {
            capture_worker.join();
        }
        if (detection_worker.joinable()) {
            detection_worker.join();
        }
        if (ocr_worker.joinable()) {
            ocr_worker.join();
        }
    };

    bool fullscreen = config.fullscreen;
    int frame_count = 0;
    bool ok = true;
    int64_t last_rendered_capture_id = 0;
    while (true) {
        bool disconnected = false;
        std::optional<LiveFramePacket> capture_packet;
        {
            std::unique_lock<std::mutex> lock(live_state.mutex);
            live_state.capture_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return live_state.stop || live_state.error ||
                       live_state.camera_disconnected ||
                       (live_state.latest_capture_frame.has_value() &&
                        live_state.latest_capture_frame->frame_id > last_rendered_capture_id);
            });
            if (live_state.error) {
                *error = live_state.error_message;
                ok = false;
                break;
            }
            if (live_state.stop) {
                break;
            }
            disconnected = live_state.camera_disconnected;
            if (!disconnected && live_state.latest_capture_frame.has_value() &&
                live_state.latest_capture_frame->frame_id > last_rendered_capture_id) {
                capture_packet = LiveFramePacket{live_state.latest_capture_frame->frame_id,
                                                  live_state.latest_capture_frame->frame.clone(),
                                                  live_state.latest_capture_frame->captured_at};
                last_rendered_capture_id = capture_packet->frame_id;
            }
        }

        cv::Mat layout;
        const auto render_begin = std::chrono::steady_clock::now();
        if (disconnected) {
            layout = MakeCameraWaitingLayout("USB 카메라 연결이 끊어졌습니다",
                                             "2초마다 자동으로 재연결을 시도합니다", font_renderer);
        } else if (!capture_packet.has_value()) {
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
            continue;
        } else {
            LiveRenderState render_state;
            {
                std::lock_guard<std::mutex> lock(live_state.mutex);
                if (live_state.latest_ocr_render.has_result) {
                    render_state = live_state.latest_ocr_render;
                } else if (live_state.latest_detection_render.has_result) {
                    render_state = live_state.latest_detection_render;
                }
            }
            if (render_state.has_result && !render_state.frame.empty()) {
                layout = RenderPipelineResult(render_state.frame, render_state.result,
                                              config.ocr_drop_score, font_renderer);
            } else {
                PipelineResult empty_result;
                layout = MakeMissingDetectionLayout(capture_packet->frame, empty_result,
                                                    font_renderer);
            }
        }
        const auto render_end = std::chrono::steady_clock::now();
        if (!disconnected && writer.isOpened()) {
            writer.write(layout);
        }

        const cv::Mat display = MakeLiveDisplayFrame(layout, config.display_scale);
        cv::imshow(config.window_name, display);
        const auto display_end = std::chrono::steady_clock::now();
        if (!disconnected) {
            std::lock_guard<std::mutex> lock(live_state.mutex);
            ++live_state.perf.render_window;
            live_state.perf.render_ms += MsBetween(render_begin, render_end);
            live_state.perf.display_ms += MsBetween(render_end, display_end);
        }

        if (!disconnected) {
            ++frame_count;
            MaybePrintLivePerfLog(config, &live_state, display_end);
        }
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
        if ((key == 10 || key == 13) && disconnected) {
            {
                std::lock_guard<std::mutex> lock(live_state.mutex);
                live_state.camera_retry_requested = true;
            }
            live_state.capture_cv.notify_one();
        }
        if (key == 'm' || key == 'M') {
            fullscreen = !fullscreen;
            cv::setWindowProperty(config.window_name, cv::WND_PROP_FULLSCREEN,
                                  fullscreen ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
            if (!fullscreen) {
                cv::resizeWindow(
                    config.window_name,
                    cv::Size(std::max(1, static_cast<int>(std::round(kLayoutWidth * config.display_scale))),
                             std::max(1, static_cast<int>(std::round(kLayoutHeight * config.display_scale)))));
            }
        }
        if (config.max_frames > 0 && frame_count >= config.max_frames) {
            break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(live_state.mutex);
        if (live_state.error) {
            *error = live_state.error_message;
            ok = false;
        }
    }

    join_workers();

    writer.release();
    cap.release();
    cv::destroyWindow(config.window_name);
    if (!ok) {
        return false;
    }
    std::cout << "{\n"
              << "  \"camera_index\": " << opened_camera_index << ",\n"
              << "  \"frames\": " << frame_count;
    if (!config.output_video_path.empty()) {
        std::cout << ",\n  \"output_video_path\": \"" << JsonEscape(config.output_video_path) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";
    return true;
}

} // namespace
} // namespace waybill

int main(int argc, char** argv) {
    namespace wb = waybill;
    const fs::path repo_root = fs::current_path();
    wb::Config config;
    std::string error;
    if (!wb::ParseArgs(argc, argv, repo_root, &config, &error)) {
        if (!error.empty()) {
            std::cerr << error << "\n";
            return 1;
        }
        return 0;
    }

    if (!wb::EnsureParentDir(config.output_json_path, &error) ||
        (!config.video_mode && !config.benchmark_mode && !config.batch_mode &&
         !wb::EnsureParentDir(config.output_image_path, &error)) ||
        (config.batch_mode && !wb::EnsureDir(config.output_dir, &error)) ||
        (config.video_mode && !wb::EnsureParentDir(config.output_video_path, &error)) ||
        (config.live_mode && !config.output_video_path.empty() &&
         !wb::EnsureParentDir(config.output_video_path, &error))) {
        std::cerr << error << "\n";
        return 1;
    }

    wb::FontRenderer font_renderer;
    if (!config.benchmark_mode && !font_renderer.Initialize(config.font_path, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    mobilint::StatusCode accelerator_status = mobilint::StatusCode::OK;
    std::unique_ptr<mobilint::Accelerator> accelerator =
        mobilint::Accelerator::create(accelerator_status);
    if (accelerator_status != mobilint::StatusCode::OK || accelerator == nullptr) {
        std::cerr << "failed to create accelerator status=" << static_cast<int>(accelerator_status)
                  << "\n";
        return 1;
    }
    wb::YoloRuntime yolo(config.yolo_model_path,
                         wb::InferYoloPostprocessType(config.yolo_model_name), config.yolo_mode,
                         config.yolo_clusters, config.yolo_cores, config.conf_thres,
                         config.iou_thres);
    wb::OcrRuntime ocr(config.ocr_det_model_path, config.ocr_rec_model_path, config.ocr_dict_path,
                       config.ocr_det_mode, config.ocr_det_clusters, config.ocr_det_cores,
                       config.ocr_rec_mode, config.ocr_rec_clusters, config.ocr_rec_cores,
                       config.ocr_drop_score, config.ocr_rec_lanes);
    if (!yolo.Initialize(accelerator.get(), &error) || !ocr.Initialize(accelerator.get(), &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    if (config.live_mode) {
        if (!wb::RunLiveMode(config, &yolo, &ocr, font_renderer, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        return 0;
    }

    if (config.benchmark_mode) {
        std::vector<std::string> image_paths = wb::CollectDemoImages(config.image_dir);
        if (image_paths.empty()) {
            std::cerr << "no demo images found in: " << config.image_dir << "\n";
            return 1;
        }
        std::mt19937 rng(config.shuffle_seed);
        std::shuffle(image_paths.begin(), image_paths.end(), rng);
        if (config.max_images > 0 && static_cast<int>(image_paths.size()) > config.max_images) {
            image_paths.resize(config.max_images);
        }

        std::ofstream out(config.output_json_path);
        if (!out.is_open()) {
            std::cerr << "failed to write output json: " << config.output_json_path << "\n";
            return 1;
        }
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"image_dir\": \"" << wb::JsonEscape(config.image_dir) << "\",\n";
        out << "  \"shuffle_seed\": " << config.shuffle_seed << ",\n";
        out << "  \"max_images\": " << config.max_images << ",\n";
        out << "  \"records\": [\n";

        wb::BenchmarkTotals totals;
        bool first_record = true;
        for (size_t i = 0; i < image_paths.size(); ++i) {
            const cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_COLOR);
            if (image.empty()) {
                continue;
            }
            wb::PipelineResult result;
            if (!wb::RunPipeline(config, &yolo, &ocr, image, &result, &error)) {
                std::cerr << error << "\n";
                return 1;
            }
            if (!first_record) {
                out << ",\n";
            }
            wb::WriteBenchmarkRecordJson(out, image_paths[i], i + 1, result);
            first_record = false;
            wb::AccumulateBenchmarkTotals(result, &totals);
        }
        out << "\n";
        out << "  ],\n";
        out << "  \"image_count\": " << totals.count << ",\n";
        wb::WriteBenchmarkAveragesJson(out, totals);
        out << "}\n";

        std::cout << std::fixed << std::setprecision(3) << "{\n"
                  << "  \"output_json_path\": \"" << wb::JsonEscape(config.output_json_path)
                  << "\",\n"
                  << "  \"image_count\": " << totals.count << ",\n"
                  << "  \"avg_overall_latency_ms\": "
                  << (totals.total_latency_ms / std::max(1, totals.count)) << "\n"
                  << "}\n";
        return 0;
    }

    if (config.batch_mode) {
        std::vector<std::string> image_paths = wb::CollectDemoImages(config.image_dir);
        if (image_paths.empty()) {
            std::cerr << "no demo images found in: " << config.image_dir << "\n";
            return 1;
        }
        std::mt19937 rng(config.shuffle_seed);
        std::shuffle(image_paths.begin(), image_paths.end(), rng);
        if (config.max_images > 0 && static_cast<int>(image_paths.size()) > config.max_images) {
            image_paths.resize(config.max_images);
        }

        std::ofstream out(config.output_json_path);
        if (!out.is_open()) {
            std::cerr << "failed to write output json: " << config.output_json_path << "\n";
            return 1;
        }
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"image_dir\": \"" << wb::JsonEscape(config.image_dir) << "\",\n";
        out << "  \"output_dir\": \"" << wb::JsonEscape(config.output_dir) << "\",\n";
        out << "  \"shuffle_seed\": " << config.shuffle_seed << ",\n";
        out << "  \"max_images\": " << config.max_images << ",\n";
        out << "  \"records\": [\n";

        wb::BenchmarkTotals totals;
        bool first_record = true;
        for (size_t i = 0; i < image_paths.size(); ++i) {
            const cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_COLOR);
            if (image.empty()) {
                continue;
            }
            wb::PipelineResult result;
            if (!wb::RunPipeline(config, &yolo, &ocr, image, &result, &error)) {
                std::cerr << error << "\n";
                return 1;
            }
            const cv::Mat layout =
                wb::RenderPipelineResult(image, result, config.ocr_drop_score, font_renderer);
            const std::string stem = wb::StemWithoutExtension(image_paths[i]);
            const fs::path image_out = fs::path(config.output_dir) / (stem + "_result.jpg");
            const fs::path json_out = fs::path(config.output_dir) / (stem + "_result.json");
            if (!cv::imwrite(image_out.string(), layout)) {
                std::cerr << "failed to write output image: " << image_out.string() << "\n";
                return 1;
            }
            wb::WriteSingleImageJson(json_out.string(), image_paths[i], result);

            if (!first_record) {
                out << ",\n";
            }
            out << "    {\n";
            out << "      \"image_path\": \"" << wb::JsonEscape(image_paths[i]) << "\",\n";
            out << "      \"output_image_path\": \"" << wb::JsonEscape(image_out.string())
                << "\",\n";
            out << "      \"output_json_path\": \"" << wb::JsonEscape(json_out.string()) << "\",\n";
            out << "      \"detected\": " << (result.detected ? "true" : "false") << ",\n";
            out << "      \"ocr_text_count\": " << result.results.size() << ",\n";
            out << "      \"overall_latency_ms\": " << result.overall_latency_sec * 1000.0 << "\n";
            out << "    }";
            first_record = false;
            wb::AccumulateBenchmarkTotals(result, &totals);
        }

        out << "\n";
        out << "  ],\n";
        out << "  \"image_count\": " << totals.count << ",\n";
        wb::WriteBenchmarkAveragesJson(out, totals);
        out << "}\n";

        std::cout << std::fixed << std::setprecision(3) << "{\n"
                  << "  \"output_dir\": \"" << wb::JsonEscape(config.output_dir) << "\",\n"
                  << "  \"output_json_path\": \"" << wb::JsonEscape(config.output_json_path)
                  << "\",\n"
                  << "  \"image_count\": " << totals.count << ",\n"
                  << "  \"avg_overall_latency_ms\": "
                  << (totals.total_latency_ms / std::max(1, totals.count)) << "\n"
                  << "}\n";
        return 0;
    }

    if (!config.video_mode) {
        const cv::Mat image = cv::imread(config.image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read input image: " << config.image_path << "\n";
            return 1;
        }
        wb::PipelineResult result;
        if (!wb::RunPipeline(config, &yolo, &ocr, image, &result, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        const cv::Mat layout =
            wb::RenderPipelineResult(image, result, config.ocr_drop_score, font_renderer);
        if (!cv::imwrite(config.output_image_path, layout)) {
            std::cerr << "failed to write output image: " << config.output_image_path << "\n";
            return 1;
        }
        wb::WriteSingleImageJson(config.output_json_path, config.image_path, result);
        std::cout << "{\n"
                  << "  \"output_image_path\": \"" << wb::JsonEscape(config.output_image_path)
                  << "\",\n"
                  << "  \"output_json_path\": \"" << wb::JsonEscape(config.output_json_path)
                  << "\"\n"
                  << "}\n";
        return 0;
    }

    std::vector<std::string> image_paths = wb::CollectDemoImages(config.image_dir);
    if (image_paths.empty()) {
        std::cerr << "no demo images found in: " << config.image_dir << "\n";
        return 1;
    }
    std::mt19937 rng(config.shuffle_seed);
    std::shuffle(image_paths.begin(), image_paths.end(), rng);
    if (config.max_images > 0 && static_cast<int>(image_paths.size()) > config.max_images) {
        image_paths.resize(config.max_images);
    }

    cv::VideoWriter writer(config.output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           config.fps, cv::Size(wb::kLayoutWidth, wb::kLayoutHeight));
    if (!writer.isOpened()) {
        std::cerr << "failed to open output video: " << config.output_video_path << "\n";
        return 1;
    }

    std::ofstream out(config.output_json_path);
    out << "{\n";
    out << "  \"image_dir\": \"" << wb::JsonEscape(config.image_dir) << "\",\n";
    out << "  \"output_video_path\": \"" << wb::JsonEscape(config.output_video_path) << "\",\n";
    out << "  \"fps\": " << config.fps << ",\n";
    out << "  \"records\": [\n";

    double total_yolo_ms = 0.0;
    double total_ocr_det_ms = 0.0;
    double total_ocr_rec_ms = 0.0;
    double total_overall_ms = 0.0;
    for (size_t i = 0; i < image_paths.size(); ++i) {
        const cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            continue;
        }
        wb::PipelineResult result;
        if (!wb::RunPipeline(config, &yolo, &ocr, image, &result, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        cv::Mat layout =
            wb::RenderPipelineResult(image, result, config.ocr_drop_score, font_renderer);
        writer.write(layout);

        const double yolo_ms = (result.yolo_timing.preprocess_sec + result.yolo_timing.infer_sec +
                                result.yolo_timing.decode_sec) *
                               1000.0;
        const double ocr_det_ms =
            (result.ocr_timing.det_preprocess_sec + result.ocr_timing.det_infer_sec +
             result.ocr_timing.det_postprocess_sec) *
            1000.0;
        const double ocr_rec_ms =
            (result.ocr_timing.rec_crop_sec + result.ocr_timing.rec_preprocess_sec +
             result.ocr_timing.rec_infer_sec + result.ocr_timing.rec_postprocess_sec) *
            1000.0;
        const double overall_ms = result.overall_latency_sec * 1000.0;
        total_yolo_ms += yolo_ms;
        total_ocr_det_ms += ocr_det_ms;
        total_ocr_rec_ms += ocr_rec_ms;
        total_overall_ms += overall_ms;

        out << "    {\n";
        out << "      \"image_path\": \"" << wb::JsonEscape(image_paths[i]) << "\",\n";
        out << "      \"frame_index\": " << (i + 1) << ",\n";
        out << "      \"detected\": " << (result.detected ? "true" : "false") << ",\n";
        out << "      \"ocr_text_count\": " << result.results.size() << ",\n";
        out << "      \"overall_latency_ms\": " << overall_ms << "\n";
        out << "    }" << (i + 1 == image_paths.size() ? "\n" : ",\n");
    }
    writer.release();
    out << "  ],\n";
    const double count = std::max<size_t>(1, image_paths.size());
    out << "  \"avg_yolo_latency_ms\": " << (total_yolo_ms / count) << ",\n";
    out << "  \"avg_ocr_det_latency_ms\": " << (total_ocr_det_ms / count) << ",\n";
    out << "  \"avg_ocr_rec_latency_ms\": " << (total_ocr_rec_ms / count) << ",\n";
    out << "  \"avg_overall_latency_ms\": " << (total_overall_ms / count) << "\n";
    out << "}\n";

    std::cout << "{\n"
              << "  \"output_video_path\": \"" << wb::JsonEscape(config.output_video_path)
              << "\",\n"
              << "  \"output_json_path\": \"" << wb::JsonEscape(config.output_json_path) << "\"\n"
              << "}\n";
    return 0;
}

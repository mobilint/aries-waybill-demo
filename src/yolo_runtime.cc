#include "yolo_runtime.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <omp.h>

namespace waybill {
namespace {

constexpr int kInputSize = 640;
constexpr int kNumClasses = 1;
constexpr int kRegMax = 16;
constexpr int kMaxDetections = 300;
constexpr std::array<int, 3> kStrides = {8, 16, 32};

float SoftmaxIntegral(const float* values) {
    float max_value = values[0];
    for (int i = 1; i < kRegMax; ++i) {
        max_value = std::max(max_value, values[i]);
    }
    float sum = 0.0f;
    float weighted = 0.0f;
    for (int i = 0; i < kRegMax; ++i) {
        const float exp_value = std::exp(values[i] - max_value);
        sum += exp_value;
        weighted += exp_value * static_cast<float>(i);
    }
    return sum > 0.0f ? weighted / sum : 0.0f;
}

} // namespace

YoloPostprocessType InferYoloPostprocessType(const std::string& model_name) {
    const std::string lower = [&model_name]() {
        std::string value = model_name;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return value;
    }();
    return lower.find("yolo11") != std::string::npos ? YoloPostprocessType::kYolo11
                                                     : YoloPostprocessType::kYolo26;
}

YoloRuntime::YoloRuntime(std::string model_path, YoloPostprocessType postprocess_type,
                         ExecMode mode, std::vector<int> cluster_ids, std::vector<int> core_ids,
                         float conf_thres, float iou_thres)
    : model_(std::move(model_path), mode, std::move(cluster_ids), std::move(core_ids)),
      postprocess_type_(postprocess_type), conf_thres_(conf_thres), iou_thres_(iou_thres) {}

YoloRuntime::~YoloRuntime() { Dispose(); }

bool YoloRuntime::Initialize(mobilint::Accelerator* accelerator, std::string* error) {
    if (!model_.Load(error)) {
        return false;
    }
    if (!model_.Launch(accelerator, error)) {
        return false;
    }
    return model_.PrepareBuffers({{1, 3, kInputSize, kInputSize}}, error);
}

float YoloRuntime::Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float YoloRuntime::IoU(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-9f);
}

std::vector<Detection> YoloRuntime::ApplyNms(std::vector<Detection> detections) const {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    kept.reserve(std::min<int>(kMaxDetections, detections.size()));
    for (const Detection& det : detections) {
        bool suppressed = false;
        for (const Detection& prior : kept) {
            if (IoU(det, prior) > iou_thres_) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(det);
            if (static_cast<int>(kept.size()) >= kMaxDetections) {
                break;
            }
        }
    }
    return kept;
}

void YoloRuntime::Preprocess(const cv::Mat& image, LetterboxInfo* info, float* dst) {
    info->original_h = image.rows;
    info->original_w = image.cols;
    info->gain = std::min(static_cast<float>(kInputSize) / image.rows,
                          static_cast<float>(kInputSize) / image.cols);
    const int resized_w = static_cast<int>(std::round(image.cols * info->gain));
    const int resized_h = static_cast<int>(std::round(image.rows * info->gain));
    info->pad_x = (kInputSize - resized_w) / 2.0f;
    info->pad_y = (kInputSize - resized_h) / 2.0f;

    cv::Mat resized;
    if (resized_w != image.cols || resized_h != image.rows) {
        cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }
    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed, static_cast<int>(std::round(info->pad_y - 0.1f)),
                       static_cast<int>(std::round(info->pad_y + 0.1f)),
                       static_cast<int>(std::round(info->pad_x - 0.1f)),
                       static_cast<int>(std::round(info->pad_x + 0.1f)), cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    cv::cvtColor(letterboxed, letterboxed, cv::COLOR_BGR2RGB);
    letterboxed.convertTo(letterboxed, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(kInputSize, kInputSize, CV_32FC1, dst + c * kInputSize * kInputSize);
    }
    cv::split(letterboxed, channels);
}

void YoloRuntime::ScaleToOriginal(Detection* detection, const LetterboxInfo& info) {
    detection->x1 = std::clamp((detection->x1 - info.pad_x) / info.gain, 0.0f,
                               static_cast<float>(info.original_w));
    detection->x2 = std::clamp((detection->x2 - info.pad_x) / info.gain, 0.0f,
                               static_cast<float>(info.original_w));
    detection->y1 = std::clamp((detection->y1 - info.pad_y) / info.gain, 0.0f,
                               static_cast<float>(info.original_h));
    detection->y2 = std::clamp((detection->y2 - info.pad_y) / info.gain, 0.0f,
                               static_cast<float>(info.original_h));
}

std::vector<Detection> YoloRuntime::Decode(const std::vector<mobilint::NDArray<float>>& outputs,
                                           const LetterboxInfo& info) const {
    return postprocess_type_ == YoloPostprocessType::kYolo11 ? DecodeYolo11(outputs, info)
                                                             : DecodeYolo26(outputs, info);
}

bool YoloRuntime::DecodeBest(const std::vector<mobilint::NDArray<float>>& outputs,
                             const LetterboxInfo& info, Detection* detection) const {
    return postprocess_type_ == YoloPostprocessType::kYolo11
               ? DecodeBestYolo11(outputs, info, detection)
               : DecodeBestYolo26(outputs, info, detection);
}

std::vector<Detection>
YoloRuntime::DecodeYolo26(const std::vector<mobilint::NDArray<float>>& outputs,
                          const LetterboxInfo& info) const {
    std::vector<const mobilint::NDArray<float>*> box_outputs;
    std::vector<const mobilint::NDArray<float>*> cls_outputs;
    for (const auto& output : outputs) {
        const auto& shape = output.shape();
        if (shape.size() != 4) {
            continue;
        }
        if (shape[1] == 4) {
            box_outputs.push_back(&output);
        } else if (shape[1] == kNumClasses) {
            cls_outputs.push_back(&output);
        }
    }
    auto area_desc = [](const mobilint::NDArray<float>* a, const mobilint::NDArray<float>* b) {
        return a->shape()[2] * a->shape()[3] > b->shape()[2] * b->shape()[3];
    };
    std::sort(box_outputs.begin(), box_outputs.end(), area_desc);
    std::sort(cls_outputs.begin(), cls_outputs.end(), area_desc);

    const float logit_thres = -std::log(1.0f / conf_thres_ - 1.0f);
    std::vector<Detection> detections;
    for (size_t level = 0; level < std::min(box_outputs.size(), cls_outputs.size()); ++level) {
        const mobilint::NDArray<float>& box = *box_outputs[level];
        const mobilint::NDArray<float>& cls = *cls_outputs[level];
        const int grid_h = static_cast<int>(box.shape()[2]);
        const int grid_w = static_cast<int>(box.shape()[3]);
        const int stride = kStrides[level];
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                const int64_t cls_idx = y * grid_w + x;
                const float logit = cls[cls_idx];
                if (logit < logit_thres) {
                    continue;
                }
                const float score = Sigmoid(logit);
                auto box_at = [&](int channel) {
                    const int64_t idx = (static_cast<int64_t>(channel) * grid_h + y) * grid_w + x;
                    return box[idx];
                };
                const float ax = (static_cast<float>(x) + 0.5f) * stride;
                const float ay = (static_cast<float>(y) + 0.5f) * stride;
                Detection det;
                det.x1 = ax - box_at(0) * stride;
                det.y1 = ay - box_at(1) * stride;
                det.x2 = ax + box_at(2) * stride;
                det.y2 = ay + box_at(3) * stride;
                det.score = score;
                detections.push_back(det);
            }
        }
    }
    detections = ApplyNms(std::move(detections));
    for (Detection& det : detections) {
        ScaleToOriginal(&det, info);
    }
    return detections;
}

bool YoloRuntime::DecodeBestYolo26(const std::vector<mobilint::NDArray<float>>& outputs,
                                   const LetterboxInfo& info, Detection* detection) const {
    std::vector<const mobilint::NDArray<float>*> box_outputs;
    std::vector<const mobilint::NDArray<float>*> cls_outputs;
    for (const auto& output : outputs) {
        const auto& shape = output.shape();
        if (shape.size() != 4) {
            continue;
        }
        if (shape[1] == 4) {
            box_outputs.push_back(&output);
        } else if (shape[1] == kNumClasses) {
            cls_outputs.push_back(&output);
        }
    }
    auto area_desc = [](const mobilint::NDArray<float>* a, const mobilint::NDArray<float>* b) {
        return a->shape()[2] * a->shape()[3] > b->shape()[2] * b->shape()[3];
    };
    std::sort(box_outputs.begin(), box_outputs.end(), area_desc);
    std::sort(cls_outputs.begin(), cls_outputs.end(), area_desc);

    bool found = false;
    float best_score = conf_thres_;
    float best_logit = -std::log(1.0f / conf_thres_ - 1.0f);
    size_t best_level = 0;
    int best_x = 0;
    int best_y = 0;
    for (size_t level = 0; level < std::min(box_outputs.size(), cls_outputs.size()); ++level) {
        const mobilint::NDArray<float>& box = *box_outputs[level];
        const mobilint::NDArray<float>& cls = *cls_outputs[level];
        const int grid_h = static_cast<int>(box.shape()[2]);
        const int grid_w = static_cast<int>(box.shape()[3]);
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                const int64_t cls_idx = y * grid_w + x;
                const float logit = cls[cls_idx];
                if (logit <= best_logit) {
                    continue;
                }
                found = true;
                best_logit = logit;
                best_score = Sigmoid(logit);
                best_level = level;
                best_x = x;
                best_y = y;
            }
        }
    }
    if (!found) {
        return false;
    }

    const mobilint::NDArray<float>& box = *box_outputs[best_level];
    const int grid_h = static_cast<int>(box.shape()[2]);
    const int grid_w = static_cast<int>(box.shape()[3]);
    const int stride = kStrides[best_level];
    auto box_at = [&](int channel) {
        const int64_t idx = (static_cast<int64_t>(channel) * grid_h + best_y) * grid_w + best_x;
        return box[idx];
    };
    const float ax = (static_cast<float>(best_x) + 0.5f) * stride;
    const float ay = (static_cast<float>(best_y) + 0.5f) * stride;
    Detection det;
    det.x1 = ax - box_at(0) * stride;
    det.y1 = ay - box_at(1) * stride;
    det.x2 = ax + box_at(2) * stride;
    det.y2 = ay + box_at(3) * stride;
    det.score = best_score;
    ScaleToOriginal(&det, info);
    *detection = det;
    return true;
}

std::vector<Detection>
YoloRuntime::DecodeYolo11(const std::vector<mobilint::NDArray<float>>& outputs,
                          const LetterboxInfo& info) const {
    std::vector<const mobilint::NDArray<float>*> box_outputs;
    std::vector<const mobilint::NDArray<float>*> cls_outputs;
    for (const auto& output : outputs) {
        const auto& shape = output.shape();
        if (shape.size() != 4) {
            continue;
        }
        if (shape[1] == kRegMax * 4) {
            box_outputs.push_back(&output);
        } else if (shape[1] == kNumClasses) {
            cls_outputs.push_back(&output);
        }
    }
    auto area_desc = [](const mobilint::NDArray<float>* a, const mobilint::NDArray<float>* b) {
        return a->shape()[2] * a->shape()[3] > b->shape()[2] * b->shape()[3];
    };
    std::sort(box_outputs.begin(), box_outputs.end(), area_desc);
    std::sort(cls_outputs.begin(), cls_outputs.end(), area_desc);

    const float logit_thres = -std::log(1.0f / conf_thres_ - 1.0f);
    std::vector<Detection> detections;
    for (size_t level = 0; level < std::min(box_outputs.size(), cls_outputs.size()); ++level) {
        const mobilint::NDArray<float>& box = *box_outputs[level];
        const mobilint::NDArray<float>& cls = *cls_outputs[level];
        const int grid_h = static_cast<int>(box.shape()[2]);
        const int grid_w = static_cast<int>(box.shape()[3]);
        const int stride = kStrides[level];
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                const int64_t cls_idx = y * grid_w + x;
                const float logit = cls[cls_idx];
                if (logit < logit_thres) {
                    continue;
                }
                const float score = Sigmoid(logit);
                float distances[4];
                for (int side = 0; side < 4; ++side) {
                    float logits[kRegMax];
                    for (int bin = 0; bin < kRegMax; ++bin) {
                        const int channel = side * kRegMax + bin;
                        const int64_t idx =
                            (static_cast<int64_t>(channel) * grid_h + y) * grid_w + x;
                        logits[bin] = box[idx];
                    }
                    distances[side] = SoftmaxIntegral(logits);
                }
                const float ax = (static_cast<float>(x) + 0.5f) * stride;
                const float ay = (static_cast<float>(y) + 0.5f) * stride;
                Detection det;
                det.x1 = ax - distances[0] * stride;
                det.y1 = ay - distances[1] * stride;
                det.x2 = ax + distances[2] * stride;
                det.y2 = ay + distances[3] * stride;
                det.score = score;
                detections.push_back(det);
            }
        }
    }
    detections = ApplyNms(std::move(detections));
    for (Detection& det : detections) {
        ScaleToOriginal(&det, info);
    }
    return detections;
}

bool YoloRuntime::DecodeBestYolo11(const std::vector<mobilint::NDArray<float>>& outputs,
                                   const LetterboxInfo& info, Detection* detection) const {
    std::vector<const mobilint::NDArray<float>*> box_outputs;
    std::vector<const mobilint::NDArray<float>*> cls_outputs;
    for (const auto& output : outputs) {
        const auto& shape = output.shape();
        if (shape.size() != 4) {
            continue;
        }
        if (shape[1] == kRegMax * 4) {
            box_outputs.push_back(&output);
        } else if (shape[1] == kNumClasses) {
            cls_outputs.push_back(&output);
        }
    }
    auto area_desc = [](const mobilint::NDArray<float>* a, const mobilint::NDArray<float>* b) {
        return a->shape()[2] * a->shape()[3] > b->shape()[2] * b->shape()[3];
    };
    std::sort(box_outputs.begin(), box_outputs.end(), area_desc);
    std::sort(cls_outputs.begin(), cls_outputs.end(), area_desc);

    bool found = false;
    float best_score = conf_thres_;
    float best_logit = -std::log(1.0f / conf_thres_ - 1.0f);
    size_t best_level = 0;
    int best_x = 0;
    int best_y = 0;
    for (size_t level = 0; level < std::min(box_outputs.size(), cls_outputs.size()); ++level) {
        const mobilint::NDArray<float>& box = *box_outputs[level];
        const mobilint::NDArray<float>& cls = *cls_outputs[level];
        const int grid_h = static_cast<int>(box.shape()[2]);
        const int grid_w = static_cast<int>(box.shape()[3]);
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                const int64_t cls_idx = y * grid_w + x;
                const float logit = cls[cls_idx];
                if (logit <= best_logit) {
                    continue;
                }
                found = true;
                best_logit = logit;
                best_score = Sigmoid(logit);
                best_level = level;
                best_x = x;
                best_y = y;
            }
        }
    }
    if (!found) {
        return false;
    }

    const mobilint::NDArray<float>& box = *box_outputs[best_level];
    const int grid_h = static_cast<int>(box.shape()[2]);
    const int grid_w = static_cast<int>(box.shape()[3]);
    const int stride = kStrides[best_level];
    float distances[4];
    for (int side = 0; side < 4; ++side) {
        float logits[kRegMax];
        for (int bin = 0; bin < kRegMax; ++bin) {
            const int channel = side * kRegMax + bin;
            const int64_t idx = (static_cast<int64_t>(channel) * grid_h + best_y) * grid_w + best_x;
            logits[bin] = box[idx];
        }
        distances[side] = SoftmaxIntegral(logits);
    }
    const float ax = (static_cast<float>(best_x) + 0.5f) * stride;
    const float ay = (static_cast<float>(best_y) + 0.5f) * stride;
    Detection det;
    det.x1 = ax - distances[0] * stride;
    det.y1 = ay - distances[1] * stride;
    det.x2 = ax + distances[2] * stride;
    det.y2 = ay + distances[3] * stride;
    det.score = best_score;
    ScaleToOriginal(&det, info);
    *detection = det;
    return true;
}

bool YoloRuntime::Infer(const cv::Mat& image, std::vector<Detection>* detections,
                        std::string* error) {
    detections->clear();
    last_timing_ = {};
    if (image.empty()) {
        return true;
    }
    const auto preprocess_begin = std::chrono::steady_clock::now();
    LetterboxInfo info;
    Preprocess(image, &info, model_.input_buffers()[0].data());
    const auto infer_begin = std::chrono::steady_clock::now();
    if (!model_.InferIntoCHW(error)) {
        return false;
    }
    const auto decode_begin = std::chrono::steady_clock::now();
    *detections = Decode(model_.output_buffers(), info);
    const auto end = std::chrono::steady_clock::now();
    last_timing_.preprocess_sec =
        std::chrono::duration<double>(infer_begin - preprocess_begin).count();
    last_timing_.infer_sec = std::chrono::duration<double>(decode_begin - infer_begin).count();
    last_timing_.decode_sec = std::chrono::duration<double>(end - decode_begin).count();
    model_.AdvanceSlot();
    return true;
}

bool YoloRuntime::InferBest(const cv::Mat& image, Detection* detection, bool* found,
                            std::string* error) {
    *detection = {};
    *found = false;
    last_timing_ = {};
    if (image.empty()) {
        return true;
    }
    const auto preprocess_begin = std::chrono::steady_clock::now();
    LetterboxInfo info;
    Preprocess(image, &info, model_.input_buffers()[0].data());
    const auto infer_begin = std::chrono::steady_clock::now();
    if (!model_.InferIntoCHW(error)) {
        return false;
    }
    const auto decode_begin = std::chrono::steady_clock::now();
    *found = DecodeBest(model_.output_buffers(), info, detection);
    const auto end = std::chrono::steady_clock::now();
    last_timing_.preprocess_sec =
        std::chrono::duration<double>(infer_begin - preprocess_begin).count();
    last_timing_.infer_sec = std::chrono::duration<double>(decode_begin - infer_begin).count();
    last_timing_.decode_sec = std::chrono::duration<double>(end - decode_begin).count();
    model_.AdvanceSlot();
    return true;
}

void YoloRuntime::Dispose() { model_.Dispose(); }

} // namespace waybill

#include "ocr_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <memory>
#include <tuple>

#include <opencv2/imgproc.hpp>

namespace waybill {
namespace {

constexpr int kDetInputHeight = 960;
constexpr int kDetInputWidth = 960;
constexpr int kRecHeight = 48;
constexpr int kRecChannels = 3;
constexpr int kDefaultRecognitionLaneCount = 2;
constexpr size_t kMaxRecognitionBufferCacheEntries = 8;

std::vector<int> RecognitionLaneClusters(const std::vector<int>& cluster_ids, int lane_index,
                                         int recognition_lane_count) {
    if (recognition_lane_count == 2 && cluster_ids.size() == 2 && lane_index >= 0 &&
        lane_index < static_cast<int>(cluster_ids.size())) {
        return {cluster_ids[static_cast<size_t>(lane_index)]};
    }
    return cluster_ids;
}

std::pair<std::vector<cv::Point2f>, float> GetMiniBox(const std::vector<cv::Point2f>& contour) {
    const cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f vertices[4];
    rect.points(vertices);
    std::vector<cv::Point2f> points(vertices, vertices + 4);
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.x < b.x; });
    const int index_1 = points[1].y > points[0].y ? 0 : 1;
    const int index_4 = points[1].y > points[0].y ? 1 : 0;
    const int index_2 = points[3].y > points[2].y ? 2 : 3;
    const int index_3 = points[3].y > points[2].y ? 3 : 2;
    return {{points[index_1], points[index_2], points[index_3], points[index_4]},
            std::min(rect.size.width, rect.size.height)};
}

float BoxScoreFast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box) {
    int xmin = bitmap.cols - 1;
    int xmax = 0;
    int ymin = bitmap.rows - 1;
    int ymax = 0;
    for (const auto& point : box) {
        xmin = std::max(0, std::min(xmin, static_cast<int>(std::floor(point.x))));
        xmax = std::min(bitmap.cols - 1, std::max(xmax, static_cast<int>(std::ceil(point.x))));
        ymin = std::max(0, std::min(ymin, static_cast<int>(std::floor(point.y))));
        ymax = std::min(bitmap.rows - 1, std::max(ymax, static_cast<int>(std::ceil(point.y))));
    }
    if (xmin > xmax || ymin > ymax) {
        return 0.0f;
    }
    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
    std::vector<cv::Point> local;
    for (const auto& point : box) {
        local.emplace_back(static_cast<int>(std::round(point.x)) - xmin,
                           static_cast<int>(std::round(point.y)) - ymin);
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{local}, cv::Scalar(1));
    return static_cast<float>(
        cv::mean(bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask)[0]);
}

std::vector<cv::Point2f> Unclip(const std::vector<cv::Point2f>& box, float unclip_ratio) {
    if (box.size() < 3) {
        return {};
    }
    double area_twice = 0.0;
    double perimeter = 0.0;
    for (size_t i = 0; i < box.size(); ++i) {
        const cv::Point2f& a = box[i];
        const cv::Point2f& b = box[(i + 1) % box.size()];
        area_twice += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
        perimeter += cv::norm(b - a);
    }
    const double area = std::abs(area_twice) * 0.5;
    if (std::abs(perimeter) < 1e-6) {
        return {};
    }
    const double distance = area * static_cast<double>(unclip_ratio) / perimeter;
    if (distance <= 0.0) {
        return {};
    }
    cv::RotatedRect rect = cv::minAreaRect(box);
    rect.size.width += 2.0f * static_cast<float>(distance);
    rect.size.height += 2.0f * static_cast<float>(distance);
    cv::Point2f vertices[4];
    rect.points(vertices);
    return std::vector<cv::Point2f>(vertices, vertices + 4);
}

std::vector<int64_t> ReduceRecognizerShape(std::vector<int64_t> shape) {
    while (shape.size() > 3) {
        auto it = std::find(shape.begin(), shape.end(), 1);
        if (it == shape.end()) {
            break;
        }
        shape.erase(it);
    }
    return shape;
}

const mobilint::NDArray<float>*
PickRecognizerOutputTensor(const std::vector<mobilint::NDArray<float>>& outputs, int batch_size,
                           int dict_size) {
    for (const auto& output : outputs) {
        std::vector<int64_t> shape = ReduceRecognizerShape(output.shape());
        if (shape.size() != 3 || shape[0] != batch_size) {
            continue;
        }
        const int dim1 = static_cast<int>(shape[1]);
        const int dim2 = static_cast<int>(shape[2]);
        const int max_dim = std::max(dim1, dim2);
        const int min_dim = std::min(dim1, dim2);
        if (max_dim >= dict_size / 2 && max_dim <= dict_size + 32 && min_dim > 1) {
            return &output;
        }
    }
    return outputs.empty() ? nullptr : &outputs.front();
}

} // namespace

OcrRuntime::OcrRuntime(std::string det_model_path, std::string rec_model_path,
                       std::string rec_dict_path, ExecMode det_mode,
                       std::vector<int> det_cluster_ids, std::vector<int> det_core_ids,
                       ExecMode rec_mode, std::vector<int> rec_cluster_ids,
                       std::vector<int> rec_core_ids, float drop_score, int recognition_lane_count)
    : det_model_(std::move(det_model_path), det_mode, std::move(det_cluster_ids),
                 std::move(det_core_ids)),
      rec_model_(rec_model_path, rec_mode,
                 RecognitionLaneClusters(rec_cluster_ids, 0, recognition_lane_count), rec_core_ids),
      rec_dict_path_(std::move(rec_dict_path)), drop_score_(drop_score),
      recognition_lane_count_(std::clamp(recognition_lane_count, 1, kDefaultRecognitionLaneCount)) {
    rec_model_extras_.reserve(recognition_lane_count_ - 1);
    rec_buffer_cache_extras_.resize(recognition_lane_count_ - 1);
    for (int lane = 1; lane < recognition_lane_count_; ++lane) {
        rec_model_extras_.push_back(std::make_unique<MxqModel>(
            rec_model_path, rec_mode,
            RecognitionLaneClusters(rec_cluster_ids, lane, recognition_lane_count_), rec_core_ids));
    }
}

OcrRuntime::~OcrRuntime() { Dispose(); }

bool OcrRuntime::Initialize(mobilint::Accelerator* accelerator, std::string* error) {
    if (!det_model_.Load(error) || !rec_model_.Load(error)) {
        return false;
    }
    for (const auto& model : rec_model_extras_) {
        if (model != nullptr && !model->Load(error)) {
            return false;
        }
    }
    if (!det_model_.Launch(accelerator, error) || !rec_model_.Launch(accelerator, error)) {
        return false;
    }
    for (const auto& model : rec_model_extras_) {
        if (model != nullptr && !model->Launch(accelerator, error)) {
            return false;
        }
    }
    if (!det_model_.PrepareBuffers({{1, kDetInputHeight, kDetInputWidth, 3}}, error)) {
        return false;
    }
    if (!LoadDictionary(error)) {
        return false;
    }
    bucket_widths_ = DefaultRecognitionBuckets();
    return true;
}

std::string OcrRuntime::TrimCarriageReturn(std::string value) {
    if (!value.empty() && value.back() == '\r') {
        value.pop_back();
    }
    return value;
}

bool OcrRuntime::LoadDictionary(std::string* error) {
    std::ifstream file(rec_dict_path_);
    if (!file.is_open()) {
        *error = "failed to open recognition dictionary: " + rec_dict_path_;
        return false;
    }
    dict_.clear();
    std::string line;
    while (std::getline(file, line)) {
        dict_.push_back(TrimCarriageReturn(line));
    }
    dict_.push_back(" ");
    return true;
}

std::vector<int> OcrRuntime::DefaultRecognitionBuckets() {
    return {64,  96,  128, 160, 192, 224, 256, 320, 352, 384,
            448, 512, 576, 640, 704, 768, 832, 896, 960};
}

cv::Mat OcrRuntime::LetterboxForDetector(const cv::Mat& image, LetterboxInfo* info) {
    const float ratio = std::min(static_cast<float>(kDetInputHeight) / image.rows,
                                 static_cast<float>(kDetInputWidth) / image.cols);
    const int resized_h = static_cast<int>(std::round(image.rows * ratio));
    const int resized_w = static_cast<int>(std::round(image.cols * ratio));
    const int pad_h = kDetInputHeight - resized_h;
    const int pad_w = kDetInputWidth - resized_w;
    info->src_height = image.rows;
    info->src_width = image.cols;
    info->resized_height = resized_h;
    info->resized_width = resized_w;
    info->pad_top = static_cast<int>(std::round(pad_h / 2.0f - 0.1f));
    info->pad_left = static_cast<int>(std::round(pad_w / 2.0f - 0.1f));

    cv::Mat resized;
    if (resized_h != image.rows || resized_w != image.cols) {
        cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }
    cv::Mat output;
    cv::copyMakeBorder(resized, output, info->pad_top, pad_h - info->pad_top, info->pad_left,
                       pad_w - info->pad_left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return output;
}

void OcrRuntime::FillDetectorInput(const cv::Mat& image, float* dst) {
    const float mean_c0 = 0.485f;
    const float mean_c1 = 0.456f;
    const float mean_c2 = 0.406f;
    const float inv_std_c0 = 1.0f / 0.229f;
    const float inv_std_c1 = 1.0f / 0.224f;
    const float inv_std_c2 = 1.0f / 0.225f;

    const int rows = kDetInputHeight;
    const int cols = kDetInputWidth;

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        const uint8_t* src_row = image.ptr<uint8_t>(y);
        float* dst_row = dst + y * cols * 3;
        for (int x = 0; x < cols; ++x) {
            dst_row[x * 3 + 0] = (static_cast<float>(src_row[x * 3 + 0]) / 255.0f - mean_c0) * inv_std_c0;
            dst_row[x * 3 + 1] = (static_cast<float>(src_row[x * 3 + 1]) / 255.0f - mean_c1) * inv_std_c1;
            dst_row[x * 3 + 2] = (static_cast<float>(src_row[x * 3 + 2]) / 255.0f - mean_c2) * inv_std_c2;
        }
    }
}

std::vector<cv::Point2f> OcrRuntime::OrderPointsClockwise(const std::vector<cv::Point2f>& input) {
    std::vector<cv::Point2f> points = input;
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.x < b.x; });
    const int index_1 = points[1].y > points[0].y ? 0 : 1;
    const int index_4 = points[1].y > points[0].y ? 1 : 0;
    const int index_2 = points[3].y > points[2].y ? 2 : 3;
    const int index_3 = points[3].y > points[2].y ? 3 : 2;
    return {points[index_1], points[index_2], points[index_3], points[index_4]};
}

std::vector<TextBox> OcrRuntime::SortBoxes(const std::vector<TextBox>& boxes) {
    std::vector<TextBox> sorted = boxes;
    std::sort(sorted.begin(), sorted.end(), [](const TextBox& a, const TextBox& b) {
        return std::tie(a.points[0].y, a.points[0].x) < std::tie(b.points[0].y, b.points[0].x);
    });
    for (size_t i = 0; i + 1 < sorted.size(); ++i) {
        for (int j = static_cast<int>(i); j >= 0; --j) {
            if (std::abs(sorted[j + 1].points[0].y - sorted[j].points[0].y) < 10.0f &&
                sorted[j + 1].points[0].x < sorted[j].points[0].x) {
                std::swap(sorted[j], sorted[j + 1]);
            } else {
                break;
            }
        }
    }
    return sorted;
}

std::vector<TextBox> OcrRuntime::DecodeDetectorOutput(const mobilint::NDArray<float>& output,
                                                      const LetterboxInfo& info,
                                                      const cv::Size& source_size) const {
    std::vector<TextBox> grouped;
    if (output.shape().size() != 4) {
        return grouped;
    }
    const int pred_h = static_cast<int>(output.shape()[1]);
    const int pred_w = static_cast<int>(output.shape()[2]);
    const int pred_c = static_cast<int>(output.shape()[3]);
    if (pred_c != 1) {
        return grouped;
    }
    cv::Mat pred(pred_h, pred_w, CV_32FC1, const_cast<float*>(output.data()));
    const int left = std::clamp(info.pad_left, 0, pred_w - 1);
    const int top = std::clamp(info.pad_top, 0, pred_h - 1);
    const int width = std::max(1, std::min(info.resized_width, pred_w - left));
    const int height = std::max(1, std::min(info.resized_height, pred_h - top));
    cv::Mat cropped = pred(cv::Rect(left, top, width, height));

    cv::Mat mask;
    cv::threshold(cropped, mask, 0.3, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour_points : contours) {
        std::vector<cv::Point2f> contour;
        contour.reserve(contour_points.size());
        for (const auto& point : contour_points) {
            contour.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
        }
        const auto mini = GetMiniBox(contour);
        if (mini.second < 3.0f) {
            continue;
        }
        const float score = BoxScoreFast(cropped, mini.first);
        if (score < 0.6f) {
            continue;
        }
        const std::vector<cv::Point2f> unclipped = Unclip(mini.first, 1.5f);
        if (unclipped.empty()) {
            continue;
        }
        const auto expanded = GetMiniBox(unclipped);
        if (expanded.second < 5.0f) {
            continue;
        }
        TextBox box;
        box.score = score;
        for (const auto& point : expanded.first) {
            box.points.emplace_back(
                std::clamp(std::round(point.x / static_cast<float>(cropped.cols) *
                                      static_cast<float>(source_size.width)),
                           0.0f, static_cast<float>(source_size.width)),
                std::clamp(std::round(point.y / static_cast<float>(cropped.rows) *
                                      static_cast<float>(source_size.height)),
                           0.0f, static_cast<float>(source_size.height)));
        }
        box.points = OrderPointsClockwise(box.points);
        grouped.push_back(std::move(box));
    }
    return SortBoxes(grouped);
}

bool OcrRuntime::Detect(const cv::Mat& image, std::vector<TextBox>* boxes, std::string* error) {
    boxes->clear();
    last_timing_.det_preprocess_sec = 0.0;
    last_timing_.det_infer_sec = 0.0;
    last_timing_.det_postprocess_sec = 0.0;
    if (image.empty()) {
        return true;
    }
    const auto preprocess_begin = std::chrono::steady_clock::now();
    LetterboxInfo info;
    cv::Mat input = LetterboxForDetector(image, &info);
    FillDetectorInput(input, det_model_.input_buffers()[0].data());
    const auto infer_begin = std::chrono::steady_clock::now();
    if (!det_model_.InferInto(error)) {
        return false;
    }
    if (det_model_.output_buffers().empty()) {
        *error = "ocr detector returned no output";
        return false;
    }
    const auto postprocess_begin = std::chrono::steady_clock::now();
    *boxes = DecodeDetectorOutput(det_model_.output_buffers()[0], info, image.size());
    const auto end = std::chrono::steady_clock::now();
    last_timing_.det_preprocess_sec =
        std::chrono::duration<double>(infer_begin - preprocess_begin).count();
    last_timing_.det_infer_sec =
        std::chrono::duration<double>(postprocess_begin - infer_begin).count();
    last_timing_.det_postprocess_sec =
        std::chrono::duration<double>(end - postprocess_begin).count();
    det_model_.AdvanceSlot();
    return true;
}

std::vector<cv::Mat>
OcrRuntime::CropRotatedTextCandidates(const cv::Mat& image,
                                      const std::vector<cv::Point2f>& points) {
    if (points.size() < 4) {
        return {};
    }
    const float width = std::max(cv::norm(points[0] - points[1]), cv::norm(points[2] - points[3]));
    const float height = std::max(cv::norm(points[0] - points[3]), cv::norm(points[1] - points[2]));
    if (width < 1.0f || height < 1.0f) {
        return {};
    }
    std::vector<cv::Point2f> target = {
        {0.0f, 0.0f}, {width, 0.0f}, {width, height}, {0.0f, height}};
    const cv::Mat transform = cv::getPerspectiveTransform(points, target);
    cv::Mat warped;
    cv::warpPerspective(image, warped, transform,
                        cv::Size(static_cast<int>(width), static_cast<int>(height)),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    if (warped.empty()) {
        return {};
    }
    if (static_cast<float>(warped.rows) / std::max(warped.cols, 1) < 1.5f) {
        cv::Mat reversed;
        cv::rotate(warped, reversed, cv::ROTATE_180);
        return {warped, reversed};
    }

    cv::Mat counterclockwise;
    cv::Mat clockwise;
    cv::rotate(warped, counterclockwise, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(warped, clockwise, cv::ROTATE_90_CLOCKWISE);
    return {counterclockwise, clockwise};
}

int OcrRuntime::PickBucketWidth(const cv::Mat& image) const {
    const float ratio = static_cast<float>(image.cols) / std::max(image.rows, 1);
    const int target_width = std::max(1, static_cast<int>(std::ceil(kRecHeight * ratio)));
    for (int bucket_width : bucket_widths_) {
        if (target_width <= bucket_width) {
            return bucket_width;
        }
    }
    return bucket_widths_.back();
}

void OcrRuntime::FillRecognizerInput(const std::vector<cv::Mat>& crops, int bucket_width,
                                     float* dst) {
    const int64_t sample_stride = static_cast<int64_t>(kRecHeight) * bucket_width * kRecChannels;
    const int64_t crop_count = static_cast<int64_t>(crops.size());
#pragma omp parallel for if (crops.size() >= 4)
    for (int64_t i = 0; i < crop_count; ++i) {
        float* sample = dst + i * sample_stride;
        std::fill(sample, sample + sample_stride, 0.0f);
        const cv::Mat& image = crops[static_cast<size_t>(i)];
        const float ratio = static_cast<float>(image.cols) / std::max(image.rows, 1);
        const int resized_w =
            std::min(bucket_width, static_cast<int>(std::ceil(kRecHeight * ratio)));
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(resized_w, kRecHeight), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat sample_mat(kRecHeight, bucket_width, CV_32FC3, sample);
        cv::Mat roi = sample_mat(cv::Rect(0, 0, resized_w, kRecHeight));
        resized.convertTo(roi, CV_32FC3, 2.0 / 255.0, -1.0);
    }
}

OcrRuntime::RecognitionBuffers*
OcrRuntime::GetRecognitionBuffers(MxqModel* model,
                                  std::map<RecognitionBufferKey, RecognitionBuffers>* cache,
                                  int batch_size, int bucket_width, std::string* error) {
    if (model == nullptr || cache == nullptr) {
        *error = "recognizer buffer owner is null";
        return nullptr;
    }
    const RecognitionBufferKey key{batch_size, bucket_width};
    if (cache->find(key) == cache->end() && cache->size() >= kMaxRecognitionBufferCacheEntries) {
        cache->clear();
    }
    auto [it, inserted] = cache->try_emplace(key);
    if (inserted) {
        if (!model->MakeBuffers({{batch_size, kRecHeight, bucket_width, kRecChannels}},
                                &it->second.input_buffers, &it->second.output_buffers, error)) {
            cache->erase(it);
            return nullptr;
        }
    }
    return &it->second;
}

bool OcrRuntime::Recognize(const cv::Mat& image, const std::vector<TextBox>& boxes,
                           std::vector<OcrResult>* results, std::string* error) {
    results->clear();
    last_timing_.rec_crop_sec = 0.0;
    last_timing_.rec_preprocess_sec = 0.0;
    last_timing_.rec_infer_sec = 0.0;
    last_timing_.rec_postprocess_sec = 0.0;
    last_timing_.rec_box_count = static_cast<int>(boxes.size());
    last_timing_.rec_crop_count = 0;
    last_timing_.rec_max_bucket_width = 0;
    last_timing_.rec_max_batch_size = 0;
    last_timing_.rec_lane_crop_counts.assign(static_cast<size_t>(recognition_lane_count_), 0);
    last_timing_.rec_lane_batch_sizes.assign(static_cast<size_t>(recognition_lane_count_), 0);
    last_timing_.rec_lane_max_bucket_widths.assign(static_cast<size_t>(recognition_lane_count_), 0);
    last_timing_.rec_lane_infer_secs.assign(static_cast<size_t>(recognition_lane_count_), 0.0);
    if (image.empty() || boxes.empty()) {
        return true;
    }

    const auto crop_begin = std::chrono::steady_clock::now();
    struct RecognitionInputBatch {
        std::vector<cv::Mat> crops;
        std::vector<int> order;
    };
    const bool use_parallel_orientation = recognition_lane_count_ > 1;
    std::vector<RecognitionInputBatch> recognition_inputs(
        use_parallel_orientation ? recognition_lane_count_ : 1);
    for (auto& input : recognition_inputs) {
        input.crops.reserve(boxes.size());
        input.order.reserve(boxes.size());
    }

    std::vector<std::vector<cv::Mat>> crop_candidates(boxes.size());
    const int64_t box_count = static_cast<int64_t>(boxes.size());
#pragma omp parallel for
    for (int64_t i = 0; i < box_count; ++i) {
        const size_t box_index = static_cast<size_t>(i);
        crop_candidates[box_index] = CropRotatedTextCandidates(image, boxes[box_index].points);
    }
    for (size_t i = 0; i < boxes.size(); ++i) {
        std::vector<cv::Mat>& candidates = crop_candidates[i];
        if (candidates.empty()) {
            continue;
        }
        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
            auto& crop = candidates[candidate_index];
            if (!crop.empty()) {
                const size_t lane_index =
                    use_parallel_orientation
                        ? std::min(candidate_index, recognition_inputs.size() - 1)
                        : 0;
                recognition_inputs[lane_index].crops.push_back(std::move(crop));
                recognition_inputs[lane_index].order.push_back(static_cast<int>(i));
            }
        }
    }

    for (size_t lane = 0; lane < recognition_inputs.size(); ++lane) {
        const int lane_crop_count = static_cast<int>(recognition_inputs[lane].crops.size());
        last_timing_.rec_lane_crop_counts[lane] = lane_crop_count;
        last_timing_.rec_crop_count += lane_crop_count;
    }

    const bool has_recognition_input =
        std::any_of(recognition_inputs.begin(), recognition_inputs.end(),
                    [](const RecognitionInputBatch& input) { return !input.crops.empty(); });
    if (!has_recognition_input) {
        return true;
    }

    const auto crop_end = std::chrono::steady_clock::now();
    last_timing_.rec_crop_sec = std::chrono::duration<double>(crop_end - crop_begin).count();

    struct RecognitionGroup {
        std::vector<cv::Mat> crops;
        std::vector<int> order;
    };
    struct RecognizerOutputView {
        const mobilint::NDArray<float>* output = nullptr;
        std::vector<int64_t> raw_shape;
        int steps = 0;
        int classes = 0;
        bool transposed = false;
    };
    struct RecognizerDecodeJob {
        int view_index = 0;
        int batch_index = 0;
        int result_index = 0;
    };
    struct RecognitionRunResult {
        bool ok = true;
        std::string error;
        std::vector<RecognizerOutputView> output_views;
        std::vector<RecognizerDecodeJob> decode_jobs;
        double preprocess_sec = 0.0;
        double infer_sec = 0.0;
        double postprocess_sec = 0.0;
        int crop_count = 0;
        int max_batch_size = 0;
        int max_bucket_width = 0;
    };

    auto run_recognition = [&](RecognitionInputBatch* input, MxqModel* model,
                               std::map<RecognitionBufferKey, RecognitionBuffers>* cache) {
        RecognitionRunResult result;
        if (input == nullptr || input->crops.empty()) {
            return result;
        }
        result.crop_count = static_cast<int>(input->crops.size());
        std::map<int, RecognitionGroup> groups;
        // Preserve the Korean demo's original recognition inputs while caching the exact batch
        // shape.
        int max_bucket_width = bucket_widths_.front();
        for (const cv::Mat& crop : input->crops) {
            max_bucket_width = std::max(max_bucket_width, PickBucketWidth(crop));
        }
        for (size_t i = 0; i < input->crops.size(); ++i) {
            RecognitionGroup& group = groups[max_bucket_width];
            group.crops.push_back(std::move(input->crops[i]));
            group.order.push_back(input->order[i]);
        }

        result.output_views.reserve(groups.size());
        result.decode_jobs.reserve(input->order.size());
        for (auto& bucket_and_group : groups) {
            const int bucket_width = bucket_and_group.first;
            RecognitionGroup& group = bucket_and_group.second;
            result.max_bucket_width = std::max(result.max_bucket_width, bucket_width);
            result.max_batch_size =
                std::max(result.max_batch_size, static_cast<int>(group.crops.size()));

            const auto preprocess_begin = std::chrono::steady_clock::now();
            RecognitionBuffers* buffers = GetRecognitionBuffers(
                model, cache, static_cast<int>(group.crops.size()), bucket_width, &result.error);
            if (buffers == nullptr) {
                result.ok = false;
                return result;
            }
            if (buffers->input_buffers.empty()) {
                result.error = "recognizer input buffer cache is empty";
                result.ok = false;
                return result;
            }
            FillRecognizerInput(group.crops, bucket_width, buffers->input_buffers[0].data());
            const auto infer_begin = std::chrono::steady_clock::now();
            result.preprocess_sec +=
                std::chrono::duration<double>(infer_begin - preprocess_begin).count();

            if (!model->InferIntoBuffers(&buffers->input_buffers, &buffers->output_buffers,
                                         &result.error)) {
                result.ok = false;
                return result;
            }
            const auto postprocess_begin = std::chrono::steady_clock::now();
            result.infer_sec +=
                std::chrono::duration<double>(postprocess_begin - infer_begin).count();

            if (buffers->output_buffers.empty()) {
                result.error = "recognizer returned no output";
                result.ok = false;
                return result;
            }
            const mobilint::NDArray<float>* output = PickRecognizerOutputTensor(
                buffers->output_buffers, static_cast<int>(group.crops.size()),
                static_cast<int>(dict_.size()) + 1);
            if (output == nullptr) {
                result.error = "recognizer returned no usable output tensor";
                result.ok = false;
                return result;
            }

            std::vector<int64_t> shape = ReduceRecognizerShape(output->shape());
            if (shape.size() != 3) {
                result.error = "unexpected recognizer output shape";
                result.ok = false;
                return result;
            }
            int steps = static_cast<int>(shape[1]);
            int classes = static_cast<int>(shape[2]);
            bool transposed = false;
            if (steps > classes) {
                std::swap(steps, classes);
                transposed = true;
            }
            if (static_cast<int>(group.order.size()) != static_cast<int>(shape[0])) {
                result.error = "recognizer output batch size mismatch";
                result.ok = false;
                return result;
            }
            const int view_index = static_cast<int>(result.output_views.size());
            result.output_views.push_back(
                RecognizerOutputView{output, output->shape(), steps, classes, transposed});
            for (size_t batch_index = 0; batch_index < group.order.size(); ++batch_index) {
                result.decode_jobs.push_back(RecognizerDecodeJob{
                    view_index, static_cast<int>(batch_index), group.order[batch_index]});
            }
            const auto postprocess_end = std::chrono::steady_clock::now();
            result.postprocess_sec +=
                std::chrono::duration<double>(postprocess_end - postprocess_begin).count();
        }
        return result;
    };

    std::vector<RecognizerOutputView> output_views;
    std::vector<RecognizerDecodeJob> decode_jobs;
    auto append_run_result = [&](RecognitionRunResult* run_result, bool include_timing) {
        const int view_offset = static_cast<int>(output_views.size());
        output_views.insert(output_views.end(), run_result->output_views.begin(),
                            run_result->output_views.end());
        for (auto job : run_result->decode_jobs) {
            job.view_index += view_offset;
            decode_jobs.push_back(job);
        }
        if (include_timing) {
            last_timing_.rec_preprocess_sec += run_result->preprocess_sec;
            last_timing_.rec_infer_sec += run_result->infer_sec;
            last_timing_.rec_postprocess_sec += run_result->postprocess_sec;
        }
    };

    auto record_lane_metrics = [&](size_t lane, const RecognitionRunResult& run_result) {
        if (lane >= last_timing_.rec_lane_batch_sizes.size()) {
            return;
        }
        last_timing_.rec_lane_batch_sizes[lane] = run_result.max_batch_size;
        last_timing_.rec_lane_max_bucket_widths[lane] = run_result.max_bucket_width;
        last_timing_.rec_lane_infer_secs[lane] = run_result.infer_sec;
        last_timing_.rec_max_batch_size =
            std::max(last_timing_.rec_max_batch_size, run_result.max_batch_size);
        last_timing_.rec_max_bucket_width =
            std::max(last_timing_.rec_max_bucket_width, run_result.max_bucket_width);
    };

    std::vector<MxqModel*> recognition_models;
    std::vector<std::map<RecognitionBufferKey, RecognitionBuffers>*> recognition_caches;
    recognition_models.reserve(recognition_inputs.size());
    recognition_caches.reserve(recognition_inputs.size());
    recognition_models.push_back(&rec_model_);
    recognition_caches.push_back(&rec_buffer_cache_);
    for (size_t i = 0; i + 1 < recognition_inputs.size(); ++i) {
        recognition_models.push_back(rec_model_extras_[i].get());
        recognition_caches.push_back(&rec_buffer_cache_extras_[i]);
    }

    if (use_parallel_orientation && recognition_inputs.size() > 1) {
        const auto parallel_begin = std::chrono::steady_clock::now();
        std::vector<RecognitionRunResult> lane_results(recognition_inputs.size());
        std::vector<std::future<RecognitionRunResult>> futures;
        std::vector<size_t> future_lanes;
        futures.reserve(recognition_inputs.size() - 1);
        future_lanes.reserve(recognition_inputs.size() - 1);
        for (size_t lane = 1; lane < recognition_inputs.size(); ++lane) {
            if (recognition_inputs[lane].crops.empty()) {
                continue;
            }
            future_lanes.push_back(lane);
            futures.push_back(std::async(std::launch::async, [&, lane]() {
                return run_recognition(&recognition_inputs[lane], recognition_models[lane],
                                       recognition_caches[lane]);
            }));
        }
        lane_results[0] =
            run_recognition(&recognition_inputs[0], recognition_models[0], recognition_caches[0]);
        for (size_t i = 0; i < futures.size(); ++i) {
            lane_results[future_lanes[i]] = futures[i].get();
        }
        const auto parallel_end = std::chrono::steady_clock::now();
        for (const auto& lane_result : lane_results) {
            if (!lane_result.ok) {
                *error = lane_result.error;
                return false;
            }
        }
        for (size_t lane = 0; lane < lane_results.size(); ++lane) {
            record_lane_metrics(lane, lane_results[lane]);
        }
        for (auto& lane_result : lane_results) {
            append_run_result(&lane_result, false);
        }
        last_timing_.rec_infer_sec +=
            std::chrono::duration<double>(parallel_end - parallel_begin).count();
    } else {
        RecognitionRunResult primary_result =
            run_recognition(&recognition_inputs[0], recognition_models[0], recognition_caches[0]);
        if (!primary_result.ok) {
            *error = primary_result.error;
            return false;
        }
        record_lane_metrics(0, primary_result);
        append_run_result(&primary_result, true);
    }

    std::vector<OcrResult> results_by_box(boxes.size());
    std::vector<bool> valid_by_box(boxes.size(), false);
    std::vector<OcrResult> local_results(decode_jobs.size());
    std::vector<unsigned char> local_valid(decode_jobs.size(), 0);
    const auto decode_begin = std::chrono::steady_clock::now();
    const int64_t decode_job_count = static_cast<int64_t>(decode_jobs.size());
#pragma omp parallel for if (decode_jobs.size() >= 4)
    for (int64_t job_index = 0; job_index < decode_job_count; ++job_index) {
        const size_t local_job_index = static_cast<size_t>(job_index);
        const RecognizerDecodeJob& job = decode_jobs[local_job_index];
        const RecognizerOutputView& view = output_views[job.view_index];
        const std::vector<int64_t>& raw_shape = view.raw_shape;
        const float* data = view.output->data();
        const int batch_index = job.batch_index;

        std::string text;
        float confidence_sum = 0.0f;
        int confidence_count = 0;
        int prev_idx = -1;
        for (int step = 0; step < view.steps; ++step) {
            const float* step_ptr = nullptr;
            int64_t class_stride = 1;
            if (raw_shape.size() == 4) {
                const int64_t d1 = raw_shape[1];
                const int64_t d2 = raw_shape[2];
                const int64_t d3 = raw_shape[3];
                if (view.transposed) {
                    step_ptr =
                        data + (((static_cast<int64_t>(batch_index) * d1 + 0) * d2) * d3) + step;
                    class_stride = d3;
                } else {
                    step_ptr =
                        data + (((static_cast<int64_t>(batch_index) * d1 + 0) * d2 + step) * d3);
                }
            } else {
                if (view.transposed) {
                    step_ptr = data +
                               static_cast<int64_t>(batch_index) * raw_shape[1] * raw_shape[2] +
                               step;
                    class_stride = raw_shape[2];
                } else {
                    step_ptr = data + ((static_cast<int64_t>(batch_index) * view.steps + step) *
                                       view.classes);
                }
            }

            int best_idx = 0;
            float best_score = -1.0f;
            for (int cls = 0; cls < view.classes; ++cls) {
                const float score = step_ptr[static_cast<int64_t>(cls) * class_stride];
                if (score > best_score) {
                    best_score = score;
                    best_idx = cls;
                }
            }
            if (best_idx != 0 && best_idx != prev_idx) {
                const int char_idx = best_idx - 1;
                if (char_idx >= 0 && char_idx < static_cast<int>(dict_.size())) {
                    text += dict_[char_idx];
                    confidence_sum += best_score;
                    ++confidence_count;
                }
            }
            prev_idx = best_idx;
        }

        const float mean_score =
            confidence_count > 0 ? confidence_sum / static_cast<float>(confidence_count) : 0.0f;
        if (!text.empty() && mean_score >= drop_score_) {
            local_results[local_job_index] = OcrResult{boxes[job.result_index], text, mean_score};
            local_valid[local_job_index] = 1;
        }
    }
    for (size_t job_index = 0; job_index < decode_jobs.size(); ++job_index) {
        if (local_valid[job_index]) {
            const int result_index = decode_jobs[job_index].result_index;
            if (result_index < 0 || result_index >= static_cast<int>(results_by_box.size())) {
                *error = "recognizer result index out of range";
                return false;
            }
            if (!valid_by_box[result_index] ||
                local_results[job_index].score > results_by_box[result_index].score) {
                results_by_box[result_index] = std::move(local_results[job_index]);
                valid_by_box[result_index] = true;
            }
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();
    last_timing_.rec_postprocess_sec +=
        std::chrono::duration<double>(decode_end - decode_begin).count();

    const auto sort_begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < results_by_box.size(); ++i) {
        if (valid_by_box[i]) {
            results->push_back(std::move(results_by_box[i]));
        }
    }
    *results = [&]() {
        std::vector<OcrResult> sorted = *results;
        std::sort(sorted.begin(), sorted.end(), [](const OcrResult& a, const OcrResult& b) {
            return std::tie(a.box.points[0].y, a.box.points[0].x) <
                   std::tie(b.box.points[0].y, b.box.points[0].x);
        });
        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            for (int j = static_cast<int>(i); j >= 0; --j) {
                if (std::abs(sorted[j + 1].box.points[0].y - sorted[j].box.points[0].y) < 10.0f &&
                    sorted[j + 1].box.points[0].x < sorted[j].box.points[0].x) {
                    std::swap(sorted[j], sorted[j + 1]);
                } else {
                    break;
                }
            }
        }
        return sorted;
    }();
    const auto sort_end = std::chrono::steady_clock::now();
    last_timing_.rec_postprocess_sec +=
        std::chrono::duration<double>(sort_end - sort_begin).count();
    return true;
}

void OcrRuntime::Dispose() {
    rec_buffer_cache_.clear();
    rec_buffer_cache_extras_.clear();
    det_model_.Dispose();
    rec_model_.Dispose();
    for (const auto& model : rec_model_extras_) {
        if (model != nullptr) {
            model->Dispose();
        }
    }
}

} // namespace waybill

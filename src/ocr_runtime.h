#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

#include "mxq_model.h"

namespace waybill {

struct TextBox {
    std::vector<cv::Point2f> points;
    float score = 0.0f;
};

struct OcrResult {
    TextBox box;
    std::string text;
    float score = 0.0f;
};

struct OcrTiming {
    double det_preprocess_sec = 0.0;
    double det_infer_sec = 0.0;
    double det_postprocess_sec = 0.0;
    double rec_crop_sec = 0.0;
    double rec_preprocess_sec = 0.0;
    double rec_infer_sec = 0.0;
    double rec_postprocess_sec = 0.0;
    int rec_box_count = 0;
    int rec_crop_count = 0;
    int rec_max_bucket_width = 0;
    int rec_max_batch_size = 0;
    std::vector<int> rec_lane_crop_counts;
    std::vector<int> rec_lane_batch_sizes;
    std::vector<int> rec_lane_max_bucket_widths;
    std::vector<double> rec_lane_infer_secs;
};

class OcrRuntime {
  public:
    OcrRuntime(std::string det_model_path, std::string rec_model_path, std::string rec_dict_path,
               ExecMode det_mode, std::vector<int> det_cluster_ids, std::vector<int> det_core_ids,
               ExecMode rec_mode, std::vector<int> rec_cluster_ids, std::vector<int> rec_core_ids,
               float drop_score, int recognition_lane_count = 2);
    ~OcrRuntime();

    bool Initialize(mobilint::Accelerator* accelerator, std::string* error);
    bool Detect(const cv::Mat& image, std::vector<TextBox>* boxes, std::string* error);
    bool Recognize(const cv::Mat& image, const std::vector<TextBox>& boxes,
                   std::vector<OcrResult>* results, std::string* error);
    void Dispose();

    const OcrTiming& last_timing() const { return last_timing_; }
    MxqModel& det_model() { return det_model_; }
    MxqModel& rec_model() { return rec_model_; }

  private:
    struct LetterboxInfo {
        int src_height = 0;
        int src_width = 0;
        int resized_height = 0;
        int resized_width = 0;
        int pad_top = 0;
        int pad_left = 0;
    };

    struct RecognitionBufferKey {
        int batch_size = 0;
        int bucket_width = 0;
        bool operator<(const RecognitionBufferKey& other) const {
            return std::tie(batch_size, bucket_width) <
                   std::tie(other.batch_size, other.bucket_width);
        }
    };

    struct RecognitionBuffers {
        std::vector<mobilint::NDArray<float>> input_buffers;
        std::vector<mobilint::NDArray<float>> output_buffers;
    };

    static cv::Mat LetterboxForDetector(const cv::Mat& image, LetterboxInfo* info);
    static void FillDetectorInput(const cv::Mat& image, float* dst);
    static void FillRecognizerInput(const std::vector<cv::Mat>& crops, int bucket_width,
                                    float* dst);
    static std::vector<cv::Point2f> OrderPointsClockwise(const std::vector<cv::Point2f>& input);
    static std::vector<TextBox> SortBoxes(const std::vector<TextBox>& boxes);
    static std::vector<cv::Mat>
    CropRotatedTextCandidates(const cv::Mat& image, const std::vector<cv::Point2f>& points);
    static std::string TrimCarriageReturn(std::string value);
    static std::vector<int> DefaultRecognitionBuckets();
    int PickBucketWidth(const cv::Mat& image) const;
    bool LoadDictionary(std::string* error);
    RecognitionBuffers*
    GetRecognitionBuffers(MxqModel* model,
                          std::map<RecognitionBufferKey, RecognitionBuffers>* cache, int batch_size,
                          int bucket_width, std::string* error);
    std::vector<TextBox> DecodeDetectorOutput(const mobilint::NDArray<float>& output,
                                              const LetterboxInfo& info,
                                              const cv::Size& source_size) const;

    MxqModel det_model_;
    MxqModel rec_model_;
    std::vector<std::unique_ptr<MxqModel>> rec_model_extras_;
    std::string rec_dict_path_;
    float drop_score_ = 0.5f;
    std::vector<std::string> dict_;
    std::vector<int> bucket_widths_;
    std::map<RecognitionBufferKey, RecognitionBuffers> rec_buffer_cache_;
    std::vector<std::map<RecognitionBufferKey, RecognitionBuffers>> rec_buffer_cache_extras_;
    int recognition_lane_count_ = 2;
    OcrTiming last_timing_;
};

} // namespace waybill

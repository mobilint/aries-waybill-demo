#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "mxq_model.h"

namespace waybill {

enum class YoloPostprocessType {
    kYolo11,
    kYolo26,
};

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int cls = 0;
};

struct YoloTiming {
    double preprocess_sec = 0.0;
    double infer_sec = 0.0;
    double decode_sec = 0.0;
};

class YoloRuntime {
  public:
    YoloRuntime(std::string model_path, YoloPostprocessType postprocess_type, ExecMode mode,
                std::vector<int> cluster_ids, std::vector<int> core_ids, float conf_thres,
                float iou_thres);
    ~YoloRuntime();

    bool Initialize(mobilint::Accelerator* accelerator, std::string* error);
    bool Infer(const cv::Mat& image, std::vector<Detection>* detections, std::string* error);
    bool InferBest(const cv::Mat& image, Detection* detection, bool* found, std::string* error);
    void Dispose();

    const YoloTiming& last_timing() const { return last_timing_; }
    MxqModel& model() { return model_; }

  private:
    struct LetterboxInfo {
        float gain = 1.0f;
        float pad_x = 0.0f;
        float pad_y = 0.0f;
        int original_h = 0;
        int original_w = 0;
    };

    static void Preprocess(const cv::Mat& image, LetterboxInfo* info, float* dst);
    static void ScaleToOriginal(Detection* detection, const LetterboxInfo& info);
    static float Sigmoid(float x);
    static float IoU(const Detection& a, const Detection& b);
    std::vector<Detection> ApplyNms(std::vector<Detection> detections) const;
    std::vector<Detection> Decode(const std::vector<mobilint::NDArray<float>>& outputs,
                                  const LetterboxInfo& info) const;
    std::vector<Detection> DecodeYolo26(const std::vector<mobilint::NDArray<float>>& outputs,
                                        const LetterboxInfo& info) const;
    std::vector<Detection> DecodeYolo11(const std::vector<mobilint::NDArray<float>>& outputs,
                                        const LetterboxInfo& info) const;
    bool DecodeBest(const std::vector<mobilint::NDArray<float>>& outputs, const LetterboxInfo& info,
                    Detection* detection) const;
    bool DecodeBestYolo26(const std::vector<mobilint::NDArray<float>>& outputs,
                          const LetterboxInfo& info, Detection* detection) const;
    bool DecodeBestYolo11(const std::vector<mobilint::NDArray<float>>& outputs,
                          const LetterboxInfo& info, Detection* detection) const;

    MxqModel model_;
    YoloPostprocessType postprocess_type_ = YoloPostprocessType::kYolo26;
    float conf_thres_ = 0.25f;
    float iou_thres_ = 0.45f;
    YoloTiming last_timing_;
};

YoloPostprocessType InferYoloPostprocessType(const std::string& model_name);

} // namespace waybill

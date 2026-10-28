#pragma once

#include <memory>
#include <string>
#include <vector>

#include <qbruntime/qbruntime.h>

namespace waybill {

enum class ExecMode {
    kSingle,
    kMulti,
    kGlobal4,
    kGlobal8,
};

struct ModelMetadata {
    mobilint::DataType input_dtype = mobilint::DataType::Float32;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};

class MxqModel {
  public:
    MxqModel(std::string model_path, ExecMode mode, std::vector<int> cluster_ids,
             std::vector<int> core_ids = {});
    ~MxqModel();

    bool Load(std::string* error);
    bool Launch(mobilint::Accelerator* accelerator, std::string* error);
    bool PrepareBuffers(const std::vector<std::vector<int64_t>>& input_shapes, std::string* error,
                        int num_slots = 1);
    bool MakeBuffers(const std::vector<std::vector<int64_t>>& input_shapes,
                     std::vector<mobilint::NDArray<float>>* input_buffers,
                     std::vector<mobilint::NDArray<float>>* output_buffers,
                     std::string* error) const;
    bool InferInto(std::string* error);
    bool InferIntoBuffers(std::vector<mobilint::NDArray<float>>* input_buffers,
                          std::vector<mobilint::NDArray<float>>* output_buffers,
                          std::string* error);
    bool InferIntoCHW(std::string* error);
    void Dispose();

    std::vector<mobilint::NDArray<float>>& input_buffers() { return input_buffers_[current_slot_]; }
    std::vector<mobilint::NDArray<float>>& output_buffers() {
        return output_buffers_[current_slot_];
    }
    int num_slots() const { return static_cast<int>(input_buffers_.size()); }
    int current_slot() const { return current_slot_; }
    void AdvanceSlot() {
        if (!input_buffers_.empty()) {
            current_slot_ = (current_slot_ + 1) % input_buffers_.size();
        }
    }

    const ModelMetadata& metadata() const { return metadata_; }
    const std::string& model_path() const { return model_path_; }

  private:
    static mobilint::Cluster ParseCluster(int cluster_id);
    static mobilint::Core ParseCore(int core_id);
    static std::string StatusToString(mobilint::StatusCode status);

    std::string model_path_;
    ExecMode mode_ = ExecMode::kMulti;
    std::vector<int> cluster_ids_;
    std::vector<int> core_ids_;
    std::unique_ptr<mobilint::Model> model_;
    ModelMetadata metadata_;
    bool launched_ = false;
    std::vector<std::vector<mobilint::NDArray<float>>> input_buffers_;
    std::vector<std::vector<mobilint::NDArray<float>>> output_buffers_;
    int current_slot_ = 0;
};

} // namespace waybill

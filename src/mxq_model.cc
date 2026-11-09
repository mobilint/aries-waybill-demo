#include "mxq_model.h"

namespace waybill {

MxqModel::MxqModel(std::string model_path, ExecMode mode, std::vector<int> cluster_ids,
                   std::vector<int> core_ids)
    : model_path_(std::move(model_path)), mode_(mode), cluster_ids_(std::move(cluster_ids)),
      core_ids_(std::move(core_ids)) {}

MxqModel::~MxqModel() { Dispose(); }

mobilint::Cluster MxqModel::ParseCluster(int cluster_id) {
    switch (cluster_id) {
    case 0:
        return mobilint::Cluster::Cluster0;
    case 1:
        return mobilint::Cluster::Cluster1;
    default:
        return mobilint::Cluster::Error;
    }
}

mobilint::Core MxqModel::ParseCore(int core_id) {
    switch (core_id) {
    case 0:
        return mobilint::Core::Core0;
    case 1:
        return mobilint::Core::Core1;
    case 2:
        return mobilint::Core::Core2;
    case 3:
        return mobilint::Core::Core3;
    default:
        return mobilint::Core::Error;
    }
}

std::string MxqModel::StatusToString(mobilint::StatusCode status) {
    return std::to_string(static_cast<int>(status));
}

bool MxqModel::Load(std::string* error) {
    mobilint::ModelConfig config;
    if (mode_ == ExecMode::kSingle) {
        if (cluster_ids_.size() != core_ids_.size() || cluster_ids_.empty()) {
            *error = "single mode requires matching cluster/core ids";
            return false;
        }
        std::vector<mobilint::CoreId> core_ids;
        core_ids.reserve(cluster_ids_.size());
        for (size_t i = 0; i < cluster_ids_.size(); ++i) {
            core_ids.push_back(
                mobilint::CoreId{ParseCluster(cluster_ids_[i]), ParseCore(core_ids_[i])});
        }
        if (!config.setSingleCoreMode(core_ids)) {
            *error = "failed to configure single core mode";
            return false;
        }
    } else {
        std::vector<mobilint::Cluster> clusters;
        for (int cluster_id : cluster_ids_) {
            clusters.push_back(ParseCluster(cluster_id));
        }
        bool ok = false;
        switch (mode_) {
        case ExecMode::kMulti:
            ok = config.setMultiCoreMode(clusters);
            break;
        case ExecMode::kGlobal4:
            ok = config.setGlobal4CoreMode(clusters);
            break;
        case ExecMode::kGlobal8:
            ok = config.setGlobal8CoreMode();
            break;
        case ExecMode::kSingle:
            break;
        }
        if (!ok) {
            *error = "failed to configure execution mode";
            return false;
        }
    }

    mobilint::StatusCode status = mobilint::StatusCode::OK;
    model_ = mobilint::Model::create(model_path_, config, status);
    if (status != mobilint::StatusCode::OK || model_ == nullptr) {
        *error = "failed to load model " + model_path_ + " status=" + StatusToString(status);
        return false;
    }
    metadata_.input_dtype = model_->getModelInputDataType();
    metadata_.input_shapes = model_->getModelInputShape();
    metadata_.output_shapes = model_->getModelOutputShape();
    return true;
}

bool MxqModel::Launch(mobilint::Accelerator* accelerator, std::string* error) {
    if (accelerator == nullptr) {
        *error = "accelerator is null";
        return false;
    }
    if (model_ == nullptr) {
        *error = "model not loaded: " + model_path_;
        return false;
    }
    if (launched_) {
        return true;
    }
    const mobilint::StatusCode status = model_->launch(*accelerator);
    if (status != mobilint::StatusCode::OK) {
        *error = "failed to launch model " + model_path_ + " status=" + StatusToString(status);
        return false;
    }
    launched_ = true;
    return true;
}

bool MxqModel::PrepareBuffers(const std::vector<std::vector<int64_t>>& input_shapes,
                              std::string* error, int num_slots) {
    if (model_ == nullptr) {
        *error = "model not loaded: " + model_path_;
        return false;
    }
    input_buffers_.clear();
    output_buffers_.clear();
    input_buffers_.resize(num_slots);
    output_buffers_.resize(num_slots);
    current_slot_ = 0;

    for (int slot = 0; slot < num_slots; ++slot) {
        if (!MakeBuffers(input_shapes, &input_buffers_[slot], &output_buffers_[slot], error)) {
            return false;
        }
    }
    return true;
}

bool MxqModel::MakeBuffers(const std::vector<std::vector<int64_t>>& input_shapes,
                           std::vector<mobilint::NDArray<float>>* input_buffers,
                           std::vector<mobilint::NDArray<float>>* output_buffers,
                           std::string* error) const {
    if (model_ == nullptr) {
        *error = "model not loaded: " + model_path_;
        return false;
    }
    input_buffers->clear();
    output_buffers->clear();

    mobilint::StatusCode status = mobilint::StatusCode::OK;
    for (const auto& input_shape : input_shapes) {
        input_buffers->emplace_back(input_shape, status);
        if (status != mobilint::StatusCode::OK) {
            *error = "failed to allocate input buffer for " + model_path_;
            return false;
        }
    }
    for (const auto& output_shape : metadata_.output_shapes) {
        std::vector<int64_t> adjusted = output_shape;
        if (!adjusted.empty() && !input_shapes.empty() && !input_shapes[0].empty()) {
            adjusted[0] = input_shapes[0][0];
        }
        output_buffers->emplace_back(adjusted, status);
        if (status != mobilint::StatusCode::OK) {
            *error = "failed to allocate output buffer for " + model_path_;
            return false;
        }
    }
    return true;
}

bool MxqModel::InferInto(std::string* error) {
    if (model_ == nullptr || !launched_) {
        *error = "model not ready: " + model_path_;
        return false;
    }
    const mobilint::StatusCode status =
        model_->infer(input_buffers_[current_slot_], output_buffers_[current_slot_]);
    if (status != mobilint::StatusCode::OK) {
        *error = "inference failed for " + model_path_ + " status=" + StatusToString(status);
        return false;
    }
    return true;
}

bool MxqModel::InferIntoBuffers(std::vector<mobilint::NDArray<float>>* input_buffers,
                                std::vector<mobilint::NDArray<float>>* output_buffers,
                                std::string* error) {
    if (model_ == nullptr || !launched_) {
        *error = "model not ready: " + model_path_;
        return false;
    }
    const mobilint::StatusCode status = model_->infer(*input_buffers, *output_buffers);
    if (status != mobilint::StatusCode::OK) {
        *error = "inference failed for " + model_path_ + " status=" + StatusToString(status);
        return false;
    }
    return true;
}

bool MxqModel::InferIntoCHW(std::string* error) {
    if (model_ == nullptr || !launched_) {
        *error = "model not ready: " + model_path_;
        return false;
    }
    const mobilint::StatusCode status =
        model_->inferCHW(input_buffers_[current_slot_], output_buffers_[current_slot_]);
    if (status != mobilint::StatusCode::OK) {
        *error = "inference failed for " + model_path_ + " status=" + StatusToString(status);
        return false;
    }
    return true;
}

void MxqModel::Dispose() {
    if (model_ != nullptr && launched_) {
        model_->dispose();
    }
    launched_ = false;
    model_.reset();
    input_buffers_.clear();
    output_buffers_.clear();
    current_slot_ = 0;
}

} // namespace waybill

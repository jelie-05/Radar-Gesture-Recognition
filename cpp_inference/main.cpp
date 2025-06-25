#define DMLC_LOG_CUSTOMIZE 1
#define DMLC_USE_CXX11 1
// #define DMLC_USE_LOGGING_LIBRARY 1


#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstring>

#include <data/samples/class0_input.h>
#include <data/samples/class0_softmax.h>
#include <data/samples/class1_input.h>
#include <data/samples/class1_softmax.h>
#include <data/samples/class2_input.h>
#include <data/samples/class2_softmax.h>
#include <data/samples/class3_input.h>
#include <data/samples/class3_softmax.h>

using namespace tvm::runtime;

struct Sample {
    const float* input;
    const float* expected_softmax;
    const char* label;
};

int main() {
    // Load TVM module
    Module mod = Module::LoadFromFile("../tvm_model.so");

    // Load graph
    std::ifstream graph_json_file("../graph.json");
    std::string graph_json((std::istreambuf_iterator<char>(graph_json_file)), std::istreambuf_iterator<char>());

    // Set device
    DLDevice dev{kDLCPU, 0};

    // Create graph executor
    Module gmod = (*Registry::Get("tvm.graph_executor.create"))(graph_json, mod, static_cast<int>(dev.device_type), dev.device_id);

    // Load parameters
    std::ifstream params_in("../tvm_params.params", std::ios::binary);
    std::string param_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    TVMByteArray param_arr{param_data.data(), param_data.size()};
    gmod.GetFunction("load_params")(param_arr);

    // Define input shape
    std::vector<int64_t> input_shape = {1, 32, 10, 2};

    // Create sample array
    std::vector<Sample> samples = {
        {class0_input, class0_softmax, "class0"},
        {class1_input, class1_softmax, "class1"},
        {class2_input, class2_softmax, "class2"},
        {class3_input, class3_softmax, "class3"},
    };

    for (const auto& sample : samples) {
        std::cout << "\n===== Inference for " << sample.label << " =====\n";

        // Create input tensor
        NDArray input_arr = NDArray::Empty(input_shape, {kDLFloat, 32, 1}, dev);
        std::memcpy(input_arr->data, sample.input, sizeof(class0_input));
        gmod.GetFunction("set_input")("serving_default_input:0", input_arr);

        // Run inference
        gmod.GetFunction("run")();
        NDArray output_arr = gmod.GetFunction("get_output")(0);
        float* out_ptr = static_cast<float*>(output_arr->data);

        // Compute output size
        const DLTensor* out_tensor = output_arr.operator->();
        int total_size = 1;
        for (int i = 0; i < out_tensor->ndim; ++i)
            total_size *= out_tensor->shape[i];

        // Print logits
        std::cout << "Logits:\n";
        for (int i = 0; i < total_size; ++i)
            std::cout << std::fixed << std::setprecision(5) << out_ptr[i] << " ";
        std::cout << "\n";

        // Apply softmax
        std::vector<float> probs(total_size);
        float sum_exp = 0.0f;
        for (int i = 0; i < total_size; ++i) {
            probs[i] = std::exp(out_ptr[i]);
            sum_exp += probs[i];
        }
        for (int i = 0; i < total_size; ++i) {
            probs[i] /= sum_exp;
        }

        // Compare to expected softmax
        std::cout << "Softmax Output vs Expected:\n";
        for (int i = 0; i < total_size; ++i) {
            std::cout << "Output[" << i << "] = " << std::fixed << std::setprecision(5)
                      << probs[i] << " | Expected: " << sample.expected_softmax[i] << "\n";
        }
    }

    return 0;
}

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib> 
#include <cmath>

using namespace tvm::runtime;

int main() {
    // Load the compiled model .so
    Module mod = Module::LoadFromFile("/home/swadiryus/projects/Radar-Gesture-Recognition/cpp_inference/tvm_model.so");

    // Load the graph structure (graph.json)
    std::ifstream graph_json_file("graph.json");
    std::string graph_json((std::istreambuf_iterator<char>(graph_json_file)), std::istreambuf_iterator<char>());

    // Set device
    DLDevice dev;
    dev.device_type = kDLCPU;
    dev.device_id = 0;

    // Create graph executor
    int device_type = dev.device_type;
    int device_id = dev.device_id;
    Module gmod = (*Registry::Get("tvm.graph_executor.create"))(graph_json, mod, device_type, device_id);

    // Load parameters .params
    std::ifstream params_in("tvm_params.params", std::ios::binary);
    std::string param_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    TVMByteArray param_arr;
    param_arr.data = param_data.data();
    param_arr.size = param_data.size();
    gmod.GetFunction("load_params")(param_arr);
    // std::cout << "[INFO] Loaded " << param_arr.size << " bytes of parameters.\n";

    // Set dummy input
    std::vector<int64_t> input_shape = {1, 32, 10, 2};
    NDArray input_arr = NDArray::Empty(input_shape, {kDLFloat, 32, 1}, dev);
    float* input_ptr = static_cast<float*>(input_arr->data);

    // Random input data
    for (int i = 0; i < 1 * 32 * 10 * 2; ++i) {
        input_ptr[i] = static_cast<float>(rand()) / RAND_MAX;  // Random dummy input
    }
    // Sinusoidal input data
    // for (int i = 0; i < 1 * 32 * 10 * 2; ++i) {
    //     float val = std::sin(i * 0.1f);        // range: [-1, 1]
    //     input_ptr[i] = (val + 1.0f) / 2.0f;     // normalized to [0, 1]
    // }

    gmod.GetFunction("set_input")("serving_default_input:0", input_arr);

    // Run inference
    std::cout << "[INFO] Running inference...\n";
    gmod.GetFunction("run")();

    // Get output
    NDArray output_arr = gmod.GetFunction("get_output")(0);
    const DLTensor* out_tensor = output_arr.operator->();


    std::cout << "Output shape: [";
    for (int i = 0; i < out_tensor->ndim; ++i) {
        std::cout << out_tensor->shape[i];
        if (i < out_tensor->ndim - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Print output values
    const float* out_ptr = static_cast<const float*>(out_tensor->data);
    int total_size = 1;
    for (int i = 0; i < out_tensor->ndim; ++i)
        total_size *= out_tensor->shape[i];

    std::cout << "Model output:\n";
    for (int i = 0; i < total_size; ++i) {
        std::cout << std::fixed << std::setprecision(3) << out_ptr[i] << " ";
    }
    std::cout << std::endl;

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

    // Print softmax output
    std::cout << "Softmax output:\n";
    for (int i = 0; i < total_size; ++i) {
        std::cout << std::fixed << std::setprecision(3) << probs[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

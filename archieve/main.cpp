#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <vector>

// === TVM runtime headers ===
extern "C" int TVMInitializeRuntime();

#include "tvm_model_c_extracted/codegen/host/include/tvmgen_default.h"

// === Sample input/output headers (replace with your paths) ===
#include "../data/samples/class0_input.h"
#include "../data/samples/class0_softmax.h"
#include "../data/samples/class1_input.h"
#include "../data/samples/class1_softmax.h"
#include "../data/samples/class2_input.h"
#include "../data/samples/class2_softmax.h"
#include "../data/samples/class3_input.h"
#include "../data/samples/class3_softmax.h"

struct Sample {
    const float* input;
    const float* expected_softmax;
    const char* label;
};

int main() {
    // === Initialize TVM runtime ===
    int init_status = TVMInitializeRuntime();
    if (init_status != 0) {
        std::cerr << "Failed to initialize TVM runtime!\n";
        return -1;
    }

    // === Define model input/output buffers ===
    float input_data[1 * 2 * 32 * 10];  // shape: [1, 2, 32, 10]
    float output_data[1 * 4];          // shape: [1, 4]

    // === Create input/output structs ===
    tvmgen_default_inputs inputs;
    tvmgen_default_outputs outputs;

    // === Connect model inputs/outputs ===
    inputs.serving_default_input_0 = input_data; 
    outputs.PartitionedCall_0  = output_data;             

    // === Prepare samples ===
    std::vector<Sample> samples = {
        {class0_input, class0_softmax, "class0"},
        {class1_input, class1_softmax, "class1"},
        {class2_input, class2_softmax, "class2"},
        {class3_input, class3_softmax, "class3"},
    };

    for (const auto& sample : samples) {
        std::cout << "\n===== Inference for " << sample.label << " =====\n";

        std::memcpy(input_data, sample.input, sizeof(input_data));

        int ret = tvmgen_default_run(&inputs, &outputs);
        if (ret != 0) {
            std::cerr << "Error running model for " << sample.label << "\n";
            continue;
        }

        std::cout << "Logits:\n";
        for (int i = 0; i < 4; ++i) {
            std::cout << std::fixed << std::setprecision(5) << output_data[i] << " ";
        }
        std::cout << "\n";

        // Softmax
        float sum_exp = 0.0f;
        std::vector<float> probs(4);
        for (int i = 0; i < 4; ++i) {
            probs[i] = std::exp(output_data[i]);
            sum_exp += probs[i];
        }
        for (int i = 0; i < 4; ++i) {
            probs[i] /= sum_exp;
        }

        std::cout << "Softmax Output vs Expected:\n";
        for (int i = 0; i < 4; ++i) {
            std::cout << "Output[" << i << "] = " << std::fixed << std::setprecision(5)
                      << probs[i] << " | Expected: " << sample.expected_softmax[i] << "\n";
        }
    }

    return 0;
}

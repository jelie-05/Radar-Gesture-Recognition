#define DMLC_LOG_CUSTOMIZE 1
#define DMLC_USE_CXX11 1
// #define DMLC_USE_LOGGING_LIBRARY 1


#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

#include "model-micro/codegen/host/include/tvmgen_default.h"

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

namespace tvm {
namespace runtime {
void InitDSORegistry();  // You will implement this
}
}

int main() {
    // === Allocate input/output buffers ===
    float input_data[1 * 3 * 32 * 10];  // shape: [1, 2, 32, 10]
    float output_data[1 * 5];          // shape: [1, 4]

    // === Define AOT executor structs ===
    tvmgen_default_inputs inputs;
    tvmgen_default_outputs outputs;
    inputs.serving_default_input_0 = input_data;
    outputs.PartitionedCall_0 = output_data;

    // Create sample array
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


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
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>


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

constexpr int PORT = 5005;

bool recv_all(int sock, uint8_t* buffer, size_t size) {
    size_t received = 0;
    while (received < size) {
        ssize_t n = recv(sock, buffer + received, size - received, 0);
        if (n <= 0) return false;
        received += n;
    }
    return true;
}

int main() {
    // // Setting up the listener
    int server_fd, client_fd;
    sockaddr_in address{};
    socklen_t addrlen = sizeof(address);

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("socket failed");
        return 1;
    }

    // Bind socket
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
    if (bind(server_fd, (sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        return 1;
    }

    // Listen for connection
    if (listen(server_fd, 1) < 0) {
        perror("listen");
        return 1;
    }

    std::cout << "[Receiver] Waiting for connection on port " << PORT << "...\n";

    client_fd = accept(server_fd, (sockaddr*)&address, &addrlen);
    if (client_fd < 0) {
        perror("accept");
        return 1;
    }
    std::cout << "[Receiver] Connected to client.\n";

    // // Initializing for prediction
    // === Allocate input/output buffers ===
    float input_data[1 * 2 * 32 * 10];  // shape: [1, 2, 32, 10]
    float output_data[1 * 4];          // shape: [1, 4]

    // === Define AOT executor structs ===
    tvmgen_default_inputs inputs;
    tvmgen_default_outputs outputs;
    inputs.serving_default_input_0 = input_data;
    outputs.PartitionedCall_0 = output_data;

    while (true) {
        uint8_t length_buf[4];
        if (!recv_all(client_fd, length_buf, 4)) break;

        // Convert big-endian to host uint32_t
        uint32_t payload_size = 
              (length_buf[0] << 24) | 
              (length_buf[1] << 16) | 
              (length_buf[2] << 8)  | 
              (length_buf[3]);

        std::vector<uint8_t> payload(payload_size);
        if (!recv_all(client_fd, payload.data(), payload_size)) break;

        std::cout << "[Frame] Received payload of size: " << payload_size << " bytes.\n";

        size_t num_floats = payload_size / sizeof(float);
        float* data = reinterpret_cast<float*>(payload.data());

        constexpr size_t expected_floats = 1 * 2 * 32 * 10;
        if (num_floats != expected_floats) {
            std::cerr << "ERROR: Received frame does not match input size! "
                      << "Expected " << expected_floats << " floats, got " << num_floats << std::endl;
            continue;
        }

        std::memcpy(input_data, data, num_floats * sizeof(float));
        
        int ret = tvmgen_default_run(&inputs, &outputs);

        std::cout << "Logits:\n";
        for (int i = 0; i < 4; ++i) {
            std::cout << std::fixed << std::setprecision(5) << output_data[i] << " ";
        }
        std::cout << "\n";
        
        // // Exporting the output to a file via TCP
        // Define server details (original device IP & port)
        const char* SERVER_IP = "192.168.1.1"; // Raspi IP
        const int SERVER_PORT = 5006;  

        // Create socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            perror("Socket creation error");
            return 1;
        }

        // Prepare server address struct
        sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(SERVER_PORT);
        if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
            std::cerr << "Invalid address/ Address not supported\n";
            return 1;
        }

        // Connect to server (Python listener)
        if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            perror("Connection Failed");
            return 1;
        }

        // Send data length first
        uint32_t data_len = sizeof(output_data); // Number of bytes in the float array
        uint32_t data_len_n = htonl(data_len);
        send(sock, &data_len_n, sizeof(data_len_n), 0);

        // Sending the data
        send(sock, output_data, sizeof(output_data), 0); // Send array directly

        std::cout << "Sent " << data_len << " bytes!\n";

        // Close the socket
        close(sock);
        return 0;

    }

    return 0;
}


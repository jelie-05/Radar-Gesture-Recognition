// tvm target: c -keys=arm_cpu,cpu -mcpu=cortex-a65
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean(float* p0, float* T_divide, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* p0_red_let = (&(global_workspace_15_var[37888]));
  for (int32_t ax0_ax1_fused_ax2_fused_ax3_fused = 0; ax0_ax1_fused_ax2_fused_ax3_fused < 64; ++ax0_ax1_fused_ax2_fused_ax3_fused) {
    ((float*)p0_red_let)[ax0_ax1_fused_ax2_fused_ax3_fused] = 0.000000e+00f;
    for (int32_t k2 = 0; k2 < 24; ++k2) {
      for (int32_t k3 = 0; k3 < 2; ++k3) {
        ((float*)p0_red_let)[ax0_ax1_fused_ax2_fused_ax3_fused] = (((float*)p0_red_let)[ax0_ax1_fused_ax2_fused_ax3_fused] + p0[(((ax0_ax1_fused_ax2_fused_ax3_fused * 48) + (k2 * 2)) + k3)]);
      }
    }
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 64; ++ax0_ax1_fused) {
    T_divide[ax0_ax1_fused] = (((float*)p0_red_let)[ax0_ax1_fused] * 2.083333e-02f);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_nn_relu(float* p0, float* T_relu, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_conv2d_constant_let = (&(global_const_workspace_4_var[108480]));
  void* fused_constant_let = (&(global_const_workspace_4_var[107008]));
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 30; ++ax0_ax1_outer_fused) {
    void* PadInput_let = (&(global_workspace_5_var[34560]));
    void* conv_let = (&(global_workspace_5_var[34304]));
    for (int32_t i1 = 0; i1 < 3; ++i1) {
      for (int32_t i2 = 0; i2 < 10; ++i2) {
        for (int32_t i3 = 0; i3 < 2; ++i3) {
          int32_t cse_var_2 = (i1 * 20);
          int32_t cse_var_1 = (i2 * 2);
          ((float*)PadInput_let)[((cse_var_2 + cse_var_1) + i3)] = p0[(((cse_var_2 + (ax0_ax1_outer_fused * 20)) + cse_var_1) + i3)];
        }
      }
    }
    for (int32_t owi_init = 0; owi_init < 8; ++owi_init) {
      for (int32_t oci_init = 0; oci_init < 8; ++oci_init) {
        ((float*)conv_let)[((owi_init * 8) + oci_init)] = 0.000000e+00f;
      }
    }
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t ic = 0; ic < 2; ++ic) {
          for (int32_t owi = 0; owi < 8; ++owi) {
            for (int32_t oci = 0; oci < 8; ++oci) {
              int32_t cse_var_3 = ((owi * 8) + oci);
              ((float*)conv_let)[cse_var_3] = (((float*)conv_let)[cse_var_3] + (((float*)PadInput_let)[((((kh * 20) + (owi * 2)) + (kw * 2)) + ic)] * ((float*)fused_constant_let)[((((kh * 48) + (kw * 16)) + (ic * 8)) + oci)]));
            }
          }
        }
      }
    }
    for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
      for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
        int32_t cse_var_4 = (ax2_inner * 8);
        float v_ = ((float*)conv_let)[(cse_var_4 + ax3_inner)] + ((float*)fused_nn_conv2d_constant_let)[ax3_inner];
        T_relu[(((ax0_ax1_outer_fused * 64) + cse_var_4) + ax3_inner)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* fused_nn_conv2d_constant_1_let = (&(global_const_workspace_6_var[108416]));
  void* fused_constant_1_let = (&(global_const_workspace_6_var[100352]));
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 28; ++ax0_ax1_outer_fused) {
    void* PadInput_let = (&(global_workspace_7_var[31744]));
    void* conv_let = (&(global_workspace_7_var[32032]));
    for (int32_t ax2_outer = 0; ax2_outer < 6; ++ax2_outer) {
      for (int32_t i1 = 0; i1 < 3; ++i1) {
        for (int32_t i2 = 0; i2 < 3; ++i2) {
          for (int32_t i3 = 0; i3 < 8; ++i3) {
            int32_t cse_var_1 = (i2 * 8);
            ((float*)PadInput_let)[(((i1 * 24) + cse_var_1) + i3)] = p0[(((((i1 * 64) + (ax0_ax1_outer_fused * 64)) + cse_var_1) + (ax2_outer * 8)) + i3)];
          }
        }
      }
      for (int32_t oco = 0; oco < 2; ++oco) {
        for (int32_t oci_init = 0; oci_init < 8; ++oci_init) {
          ((float*)conv_let)[((oco * 8) + oci_init)] = 0.000000e+00f;
        }
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic = 0; ic < 8; ++ic) {
              for (int32_t oci = 0; oci < 8; ++oci) {
                int32_t cse_var_3 = (oco * 8);
                int32_t cse_var_2 = (cse_var_3 + oci);
                ((float*)conv_let)[cse_var_2] = (((float*)conv_let)[cse_var_2] + (((float*)PadInput_let)[(((kh * 24) + (kw * 8)) + ic)] * ((float*)fused_constant_1_let)[(((((kh * 384) + (kw * 128)) + (ic * 16)) + cse_var_3) + oci)]));
              }
            }
          }
        }
      }
      for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
        for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
          int32_t cse_var_5 = (ax3_outer * 8);
          int32_t cse_var_4 = (cse_var_5 + ax3_inner);
          float v_ = ((float*)conv_let)[cse_var_4] + ((float*)fused_nn_conv2d_constant_1_let)[cse_var_4];
          T_relu[((((ax0_ax1_outer_fused * 96) + (ax2_outer * 16)) + cse_var_5) + ax3_inner)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_nn_relu_2(float* p0, float* T_relu, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  void* fused_nn_conv2d_constant_2_let = (&(global_const_workspace_8_var[108224]));
  void* fused_constant_2_let = (&(global_const_workspace_8_var[73728]));
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 26; ++ax0_ax1_outer_fused) {
    void* PadInput_let = (&(global_workspace_9_var[24064]));
    void* conv_let = (&(global_workspace_9_var[25216]));
    for (int32_t i1 = 0; i1 < 3; ++i1) {
      for (int32_t i2 = 0; i2 < 6; ++i2) {
        for (int32_t i3 = 0; i3 < 16; ++i3) {
          int32_t cse_var_2 = (i1 * 96);
          int32_t cse_var_1 = (i2 * 16);
          ((float*)PadInput_let)[((cse_var_2 + cse_var_1) + i3)] = p0[(((cse_var_2 + (ax0_ax1_outer_fused * 96)) + cse_var_1) + i3)];
        }
      }
    }
    for (int32_t oco = 0; oco < 4; ++oco) {
      for (int32_t owi_init = 0; owi_init < 4; ++owi_init) {
        for (int32_t oci_init = 0; oci_init < 8; ++oci_init) {
          ((float*)conv_let)[(((oco * 32) + (owi_init * 8)) + oci_init)] = 0.000000e+00f;
        }
      }
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic = 0; ic < 16; ++ic) {
            for (int32_t owi = 0; owi < 4; ++owi) {
              for (int32_t oci = 0; oci < 8; ++oci) {
                int32_t cse_var_3 = (((oco * 32) + (owi * 8)) + oci);
                ((float*)conv_let)[cse_var_3] = (((float*)conv_let)[cse_var_3] + (((float*)PadInput_let)[((((kh * 96) + (owi * 16)) + (kw * 16)) + ic)] * ((float*)fused_constant_2_let)[(((((kh * 1536) + (kw * 512)) + (ic * 32)) + (oco * 8)) + oci)]));
              }
            }
          }
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 4; ++ax3_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
          int32_t cse_var_4 = (ax3_outer * 8);
          float v_ = ((float*)conv_let)[(((ax3_outer * 32) + (ax2_inner * 8)) + ax3_inner)] + ((float*)fused_nn_conv2d_constant_2_let)[(cse_var_4 + ax3_inner)];
          T_relu[((((ax0_ax1_outer_fused * 128) + (ax2_inner * 32)) + cse_var_4) + ax3_inner)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_nn_relu_3(float* p0, float* T_relu, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  void* fused_nn_conv2d_constant_3_let = (&(global_const_workspace_10_var[107584]));
  void* fused_constant_3_let = (&(global_const_workspace_10_var[0]));
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 24; ++ax0_ax1_outer_fused) {
    void* PadInput_let = (&(global_workspace_11_var[25600]));
    void* conv_let = (&(global_workspace_11_var[26752]));
    for (int32_t ax2_outer = 0; ax2_outer < 2; ++ax2_outer) {
      for (int32_t i1 = 0; i1 < 3; ++i1) {
        for (int32_t i2 = 0; i2 < 3; ++i2) {
          for (int32_t i3 = 0; i3 < 32; ++i3) {
            int32_t cse_var_1 = (i2 * 32);
            ((float*)PadInput_let)[(((i1 * 96) + cse_var_1) + i3)] = p0[(((((i1 * 128) + (ax0_ax1_outer_fused * 128)) + cse_var_1) + (ax2_outer * 32)) + i3)];
          }
        }
      }
      for (int32_t oco = 0; oco < 8; ++oco) {
        for (int32_t oci_init = 0; oci_init < 8; ++oci_init) {
          ((float*)conv_let)[((oco * 8) + oci_init)] = 0.000000e+00f;
        }
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic = 0; ic < 32; ++ic) {
              for (int32_t oci = 0; oci < 8; ++oci) {
                int32_t cse_var_3 = (oco * 8);
                int32_t cse_var_2 = (cse_var_3 + oci);
                ((float*)conv_let)[cse_var_2] = (((float*)conv_let)[cse_var_2] + (((float*)PadInput_let)[(((kh * 96) + (kw * 32)) + ic)] * ((float*)fused_constant_3_let)[(((((kh * 6144) + (kw * 2048)) + (ic * 64)) + cse_var_3) + oci)]));
              }
            }
          }
        }
      }
      for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
        for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
          int32_t cse_var_5 = (ax3_outer * 8);
          int32_t cse_var_4 = (cse_var_5 + ax3_inner);
          float v_ = ((float*)conv_let)[cse_var_4] + ((float*)fused_nn_conv2d_constant_3_let)[cse_var_4];
          T_relu[((((ax0_ax1_outer_fused * 128) + (ax2_outer * 64)) + cse_var_5) + ax3_inner)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_add(float* p0, float* T_add, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* fused_nn_dense_constant_let = (&(global_const_workspace_16_var[108096]));
  void* fused_constant_4_let = (&(global_const_workspace_16_var[92160]));
  void* T_matmul_NT_let = (&(global_workspace_17_var[256]));
  for (int32_t j = 0; j < 32; ++j) {
    ((float*)T_matmul_NT_let)[j] = 0.000000e+00f;
    for (int32_t k = 0; k < 64; ++k) {
      ((float*)T_matmul_NT_let)[j] = (((float*)T_matmul_NT_let)[j] + (p0[k] * ((float*)fused_constant_4_let)[((j * 64) + k)]));
    }
  }
  for (int32_t ax1 = 0; ax1 < 32; ++ax1) {
    T_add[ax1] = (((float*)T_matmul_NT_let)[ax1] + ((float*)fused_nn_dense_constant_let)[ax1]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_add_1(float* p0, float* T_add, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* fused_nn_dense_constant_1_let = (&(global_const_workspace_18_var[108352]));
  void* fused_constant_5_let = (&(global_const_workspace_18_var[104960]));
  void* T_matmul_NT_let = (&(global_workspace_19_var[128]));
  for (int32_t j = 0; j < 16; ++j) {
    ((float*)T_matmul_NT_let)[j] = 0.000000e+00f;
    for (int32_t k = 0; k < 32; ++k) {
      ((float*)T_matmul_NT_let)[j] = (((float*)T_matmul_NT_let)[j] + (p0[k] * ((float*)fused_constant_5_let)[((j * 32) + k)]));
    }
  }
  for (int32_t ax1 = 0; ax1 < 16; ++ax1) {
    T_add[ax1] = (((float*)T_matmul_NT_let)[ax1] + ((float*)fused_nn_dense_constant_1_let)[ax1]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_add_2(float* p0, float* T_add, uint8_t* global_const_workspace_20_var, uint8_t* global_workspace_21_var) {
  void* fused_nn_dense_constant_2_let = (&(global_const_workspace_20_var[108512]));
  void* fused_constant_6_let = (&(global_const_workspace_20_var[107840]));
  void* T_matmul_NT_let = (&(global_workspace_21_var[64]));
  for (int32_t j = 0; j < 4; ++j) {
    ((float*)T_matmul_NT_let)[j] = 0.000000e+00f;
    for (int32_t k = 0; k < 16; ++k) {
      ((float*)T_matmul_NT_let)[j] = (((float*)T_matmul_NT_let)[j] + (p0[k] * ((float*)fused_constant_6_let)[((j * 16) + k)]));
    }
  }
  for (int32_t ax1 = 0; ax1 < 4; ++ax1) {
    T_add[ax1] = (((float*)T_matmul_NT_let)[ax1] + ((float*)fused_nn_dense_constant_2_let)[ax1]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose(float* p0, float* T_transpose, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 320; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_transpose[((ax0_ax1_fused_ax2_fused * 2) + ax3_inner)] = p0[((ax3_inner * 320) + ax0_ax1_fused_ax2_fused)];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_1(float* p0, float* T_transpose, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1536; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_transpose[((ax0_ax1_fused_ax2_fused * 2) + ax3_inner)] = p0[((((ax0_ax1_fused_ax2_fused % 24) * 128) + (ax3_inner * 64)) + (ax0_ax1_fused_ax2_fused / 24))];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* serving_default_input_0_buffer_var, float* PartitionedCall_0_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_5_let = (&(global_workspace_1_var[13312]));
  void* sid_4_let = (&(global_workspace_1_var[0]));
  void* sid_2_let = (&(global_workspace_1_var[24064]));
  void* sid_3_let = (&(global_workspace_1_var[13312]));
  void* sid_1_let = (&(global_workspace_1_var[31744]));
  void* sid_6_let = (&(global_workspace_1_var[25600]));
  void* sid_7_let = (&(global_workspace_1_var[0]));
  void* sid_8_let = (&(global_workspace_1_var[0]));
  void* sid_9_let = (&(global_workspace_1_var[0]));
  if (tvmgen_default_fused_transpose(serving_default_input_0_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu(sid_1_let, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu_1(sid_2_let, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu_2(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu_3(sid_4_let, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_1(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_mean(sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_add(sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_add_1(sid_8_let, sid_9_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_add_2(sid_9_let, PartitionedCall_0_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}


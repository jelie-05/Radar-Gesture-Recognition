// tvm target: c -keys=cpu 
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean(float* p0, float* T_divide, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* p0_red_let = (&(global_workspace_15_var[51200]));
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
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add(float* p0, float* T_add, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* fused_nn_contrib_dense_pack_constant_let = (&(global_const_workspace_16_var[108224]));
  void* fused_constant_4_let = (&(global_const_workspace_16_var[92160]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_global_let = (&(global_workspace_17_var[384]));
    for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
      ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
    }
    for (int32_t k_outer = 0; k_outer < 64; ++k_outer) {
      for (int32_t x_c = 0; x_c < 8; ++x_c) {
        ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_4_let)[(((ax1_outer_ax0_outer_fused * 512) + (k_outer * 8)) + x_c)]));
      }
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 8) + ax1_inner_inner);
      T_add[cse_var_1] = (((float*)compute_global_let)[ax1_inner_inner] + ((float*)fused_nn_contrib_dense_pack_constant_let)[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_1(float* p0, float* T_add, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* fused_nn_contrib_dense_pack_constant_1_let = (&(global_const_workspace_18_var[108416]));
  void* fused_constant_5_let = (&(global_const_workspace_18_var[104960]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 2; ++ax1_outer_ax0_outer_fused) {
    void* compute_global_let = (&(global_workspace_19_var[448]));
    for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
      ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
    }
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      for (int32_t x_c = 0; x_c < 8; ++x_c) {
        ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_5_let)[(((ax1_outer_ax0_outer_fused * 256) + (k_outer * 8)) + x_c)]));
      }
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 8) + ax1_inner_inner);
      T_add[cse_var_1] = (((float*)compute_global_let)[ax1_inner_inner] + ((float*)fused_nn_contrib_dense_pack_constant_1_let)[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_2(float* p0, float* T_add, uint8_t* global_const_workspace_20_var, uint8_t* global_workspace_21_var) {
  void* fused_nn_contrib_dense_pack_constant_2_let = (&(global_const_workspace_20_var[108512]));
  void* fused_constant_6_let = (&(global_const_workspace_20_var[107840]));
  void* compute_global_let = (&(global_workspace_21_var[448]));
  for (int32_t x_c_init = 0; x_c_init < 4; ++x_c_init) {
    ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
  }
  for (int32_t k_outer = 0; k_outer < 16; ++k_outer) {
    for (int32_t x_c = 0; x_c < 4; ++x_c) {
      ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_6_let)[((k_outer * 4) + x_c)]));
    }
  }
  for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
    T_add[ax1_inner_inner] = (((float*)compute_global_let)[ax1_inner_inner] + ((float*)fused_nn_contrib_dense_pack_constant_2_let)[ax1_inner_inner]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_nn_relu(float* p0, float* T_relu, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_conv2d_constant_let = (&(global_const_workspace_4_var[108480]));
  void* fused_constant_let = (&(global_const_workspace_4_var[107008]));
  void* pad_temp_let = (&(global_workspace_5_var[7680]));
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 32; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 10; ++i2) {
      for (int32_t i3 = 0; i3 < 2; ++i3) {
        int32_t cse_var_1 = (((i0_i1_fused * 20) + (i2 * 2)) + i3);
        ((float*)pad_temp_let)[cse_var_1] = p0[cse_var_1];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 240; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_nhwc_let = (&(global_workspace_5_var[10240]));
    for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
      ((float*)conv2d_nhwc_let)[0] = 0.000000e+00f;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 2; ++rc) {
            ((float*)conv2d_nhwc_let)[0] = (((float*)conv2d_nhwc_let)[0] + (((float*)pad_temp_let)[((((((ax0_ax1_fused_ax2_fused >> 3) * 20) + (ry * 20)) + (rx * 2)) + ((ax0_ax1_fused_ax2_fused & 7) * 2)) + rc)] * ((float*)fused_constant_let)[((((ry * 48) + (rx * 16)) + (rc * 8)) + ax3)]));
          }
        }
      }
      float v_ = ((float*)conv2d_nhwc_let)[0] + ((float*)fused_nn_conv2d_constant_let)[ax3];
      T_relu[((ax0_ax1_fused_ax2_fused * 8) + ax3)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* fused_nn_conv2d_constant_1_let = (&(global_const_workspace_6_var[108352]));
  void* fused_constant_1_let = (&(global_const_workspace_6_var[100352]));
  void* pad_temp_let = (&(global_workspace_7_var[10752]));
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 30; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 8; ++i2) {
      for (int32_t i3 = 0; i3 < 8; ++i3) {
        int32_t cse_var_1 = (((i0_i1_fused * 64) + (i2 * 8)) + i3);
        ((float*)pad_temp_let)[cse_var_1] = p0[cse_var_1];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 168; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_nhwc_let = (&(global_workspace_7_var[18432]));
    for (int32_t ax3 = 0; ax3 < 16; ++ax3) {
      ((float*)conv2d_nhwc_let)[0] = 0.000000e+00f;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 8; ++rc) {
            ((float*)conv2d_nhwc_let)[0] = (((float*)conv2d_nhwc_let)[0] + (((float*)pad_temp_let)[((((((ax0_ax1_fused_ax2_fused / 6) * 64) + (ry * 64)) + (rx * 8)) + ((ax0_ax1_fused_ax2_fused % 6) * 8)) + rc)] * ((float*)fused_constant_1_let)[((((ry * 384) + (rx * 128)) + (rc * 16)) + ax3)]));
          }
        }
      }
      float v_ = ((float*)conv2d_nhwc_let)[0] + ((float*)fused_nn_conv2d_constant_1_let)[ax3];
      T_relu[((ax0_ax1_fused_ax2_fused * 16) + ax3)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_nn_relu_2(float* p0, float* T_relu, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  void* fused_nn_conv2d_constant_2_let = (&(global_const_workspace_8_var[108096]));
  void* fused_constant_2_let = (&(global_const_workspace_8_var[73728]));
  void* pad_temp_let = (&(global_workspace_9_var[13312]));
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 28; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 6; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        int32_t cse_var_1 = (((i0_i1_fused * 96) + (i2 * 16)) + i3);
        ((float*)pad_temp_let)[cse_var_1] = p0[cse_var_1];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 104; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_nhwc_let = (&(global_workspace_9_var[24064]));
    for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
      ((float*)conv2d_nhwc_let)[0] = 0.000000e+00f;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 16; ++rc) {
            ((float*)conv2d_nhwc_let)[0] = (((float*)conv2d_nhwc_let)[0] + (((float*)pad_temp_let)[((((((ax0_ax1_fused_ax2_fused >> 2) * 96) + (ry * 96)) + (rx * 16)) + ((ax0_ax1_fused_ax2_fused & 3) * 16)) + rc)] * ((float*)fused_constant_2_let)[((((ry * 1536) + (rx * 512)) + (rc * 32)) + ax3)]));
          }
        }
      }
      float v_ = ((float*)conv2d_nhwc_let)[0] + ((float*)fused_nn_conv2d_constant_2_let)[ax3];
      T_relu[((ax0_ax1_fused_ax2_fused * 32) + ax3)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
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
  void* pad_temp_let = (&(global_workspace_11_var[13312]));
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 26; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 4; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        int32_t cse_var_1 = (((i0_i1_fused * 128) + (i2 * 32)) + i3);
        ((float*)pad_temp_let)[cse_var_1] = p0[cse_var_1];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 48; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_nhwc_let = (&(global_workspace_11_var[38912]));
    for (int32_t ff = 0; ff < 64; ++ff) {
      ((float*)conv2d_nhwc_let)[ff] = 0.000000e+00f;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 32; ++rc) {
            ((float*)conv2d_nhwc_let)[ff] = (((float*)conv2d_nhwc_let)[ff] + (((float*)pad_temp_let)[((((((ax0_ax1_fused_ax2_fused >> 1) * 128) + (ry * 128)) + (rx * 32)) + ((ax0_ax1_fused_ax2_fused & 1) * 32)) + rc)] * ((float*)fused_constant_3_let)[((((ry * 6144) + (rx * 2048)) + (rc * 64)) + ff)]));
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 64; ++ax3_inner) {
      float v_ = ((float*)conv2d_nhwc_let)[ax3_inner] + ((float*)fused_nn_conv2d_constant_3_let)[ax3_inner];
      T_relu[((ax0_ax1_fused_ax2_fused * 64) + ax3_inner)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose(float* p0, float* T_transpose, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 32; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
        T_transpose[(((ax0_ax1_fused * 20) + (ax2 * 2)) + ax3_inner)] = p0[(((ax3_inner * 320) + (ax0_ax1_fused * 10)) + ax2)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_1(float* p0, float* T_transpose, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 64; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 24; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
        T_transpose[(((ax0_ax1_fused * 48) + (ax2 * 2)) + ax3_inner)] = p0[(((ax2 * 128) + (ax3_inner * 64)) + ax0_ax1_fused)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* serving_default_input_0_buffer_var, float* PartitionedCall_0_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_6_let = (&(global_workspace_1_var[38912]));
  void* sid_3_let = (&(global_workspace_1_var[0]));
  void* sid_2_let = (&(global_workspace_1_var[0]));
  void* sid_1_let = (&(global_workspace_1_var[10240]));
  void* sid_5_let = (&(global_workspace_1_var[26624]));
  void* sid_4_let = (&(global_workspace_1_var[0]));
  void* sid_7_let = (&(global_workspace_1_var[0]));
  void* sid_8_let = (&(global_workspace_1_var[256]));
  void* sid_9_let = (&(global_workspace_1_var[384]));
  if (tvmgen_default_fused_transpose(serving_default_input_0_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu(sid_1_let, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu_1(sid_2_let, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu_2(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_nn_relu_3(sid_4_let, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_1(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_mean(sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add(sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_1(sid_8_let, sid_9_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_2(sid_9_let, PartitionedCall_0_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}


#pragma once
#include <torch/extension.h>

torch::Tensor gemv_mxq_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _kernel_last,
    // torch::Tensor _scales, // 4b
    // torch::Tensor _zeros,  
    torch::Tensor _zeros_and_scales,
    torch::Tensor _scales_2nd, // fp16
    torch::Tensor _zeros_2nd,
    torch::Tensor _scales_4b, // fp16
    torch::Tensor _zeros_4b, // 2b
    int split_k_iters);

/*

*/

#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "gemv_mxq_cuda.h"
#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define PACK_FACTOR_2b 16
#define WARP_SIZE 32
#define DEBUG_THY 1023


// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
    #pragma unroll
    for(int i = 4; i >= 0; i--){
        sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
    }
    /*
    // Equivalent to the following tree reduction implementation:
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    */
    return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
    return (c + divisor - 1) / divisor;
}

// Use shared memory to load 1st zeros and 1st scales
__global__ void gemv_mxq_kernel_g16_v0(
    const double4* _inputs, const uint32_t* weight, const uint32_t* weight_last, 
    // const uint32_t* zeros, const uint32_t* scales, 
    const uint32_t* zeros_and_scales,
    const half* scales_2nd, const uint32_t* zeros_2nd, const half* scale_4b, uint32_t* zero_4b, half* _outputs, 
    const int IC, const int OC){
    const int group_size = 16;
    const int second_order_group_size = 4; // == blockDim.y
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const double4 *inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half *outputs = _outputs + batch_idx * OC;

    // const int num_groups_packed = make_divisible(IC, (64*WARP_SIZE)); // loop_num = 2
    const int weight_w = IC / (16 * 3 + 8 * 2) * 4;      // 4096/64*4 = 256
    const int weight_last_w = IC / (16 * 3 + 8 * 2) * 1; // 64
    // const int zeros_w_1st = make_divisible(IC*0.75 / group_size, PACK_FACTOR_2b); // 4096*0.75/16/16=12 uint32 zeros_1st(192 2b zeros_1st)
    // const int sf_w_1st = make_divisible(IC*0.75 / group_size, PACK_FACTOR); // 4096*0.75/16/8=24 uint32 scales_1st(192 4b scales_ist)
    const int zeros_and_scales_w_1st = make_divisible(IC / group_size, PACK_FACTOR_2b); // 4096*0.75/16/16=12 uint32 zeros_1st(192 2b zeros_1st)
    const int zeros_w_2nd = 32;
    // const int sf_w_1st = make_divisible(IC / group_size, PACK_FACTOR_2b); // 4096*0.75/16/8=24 uint32 scales_1st(192 4b scales_ist)
    const float scaling_factor_4b = __half2float(scale_4b[oc_idx]);
    const int zeros_4b = (zero_4b[oc_idx / 8] >> (oc_idx % 8 * 4)) & 0xF;

    // ################ shared memory definition #####################
    // ################ packed 1st scales, 1st zeros and 2nd zeros ##############################
    __shared__ u_int32_t shared_packed_1st_zeros_and_scales[32][4]; // include 1st zeros and 1st scales for 4 OC channels in this thread block
    __shared__ u_int32_t shared_packed_2nd_zeros[32];

    shared_packed_1st_zeros_and_scales[threadIdx.x][threadIdx.y] = zeros_and_scales[oc_idx * zeros_and_scales_w_1st * 2 + threadIdx.x];
    shared_packed_2nd_zeros[threadIdx.x] = zeros_2nd[oc_idx / second_order_group_size * zeros_w_2nd + threadIdx.x];

    // __shared__ double4 shared_packed_input[32];
    // shared_packed_input[blockDim.x * threadIdx.y + threadIdx.x] = inputs[blockDim.x * threadIdx.y + threadIdx.x];
    // shared_packed_input[(blockDim.x * threadIdx.y + threadIdx.x) + 128] = inputs[(blockDim.x * threadIdx.y + threadIdx.x) + 128];
    // shared_packed_input[threadIdx.x] = inputs[threadIdx.x];
    // __syncthreads();
    // ###############################################################

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w);
    //  tile size: 4 OC x 1024 IC per iter

    uint32_t packed_zeros_and_scales_1st; // Including two loops all 1st zeros and scales for this OC channel
    packed_zeros_and_scales_1st = shared_packed_1st_zeros_and_scales[threadIdx.x][threadIdx.y];

    uint32_t packed_zeros_2nd;
    packed_zeros_2nd = shared_packed_2nd_zeros[threadIdx.x];

    __shared__ double4 shared_packed_inputs[256];
    shared_packed_inputs[threadIdx.x * 4 + threadIdx.y] = inputs[threadIdx.x * 4 + threadIdx.y];
    shared_packed_inputs[128 + threadIdx.x * 4 + threadIdx.y] = inputs[128 + threadIdx.x * 4 + threadIdx.y];
    __syncthreads();

    for (int packed_group_idx = 0; packed_group_idx < 2; packed_group_idx++)
    {
        // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
        // Each loop, 32 threads(1 warp) load 6*int32 zeros_1st(96*2b), load 12*int32 scales_1st(96*4b), load 12*int32 zeros_2nd(96*4b), load

        uint32_t loop_packed_zeros_1st = packed_zeros_and_scales_1st >> (16 * packed_group_idx + 0) & 0xFF;
        uint32_t loop_packed_scales_1st = packed_zeros_and_scales_1st >> (16 * packed_group_idx + 8) & 0xFF;
        uint32_t loop_packed_zeros_2nd = packed_zeros_2nd >> (8 * packed_group_idx) & 0xFF;

        // Then, 32 threads(1 warp) load 2048(32*64) elements, 32*3*16=1536 2b and 32*1*16=512 4b
        uint32_t packed_weights[4];
        uint32_t packed_weights_last;
        // use float4 to load weights, each thread first load 3*16*2b and 1*8*4b, then load 1*8*4b, totally 64 elements
        *((float4 *)(packed_weights)) = *((float4 *)(weight + oc_idx * weight_w + packed_group_idx * weight_w / 2 + threadIdx.x * 4));
        packed_weights_last = weight_last[oc_idx * weight_last_w + packed_group_idx * weight_last_w / 2 + threadIdx.x];

        // int inputs_ptr_delta = packed_group_idx * 0 + threadIdx.x * 1; // 0, 2048/2, ...
        // const double4 *inputs_ptr = shared_packed_input + inputs_ptr_delta;

        // __shared__ double4 shared_packed_inputs[128];
        // shared_packed_inputs[threadIdx.x * 4] = inputs[packed_group_idx * 128  + threadIdx.x * 4];
        // __syncthreads();

#pragma unroll
        for (int ii = 0; ii < 3; ii++)
        { // Get first 3 uint32 packed weights(total 3*16*2b)
            // Then, get scales and zeros for each thread, need dequantize 2nd scales, each thread mapping
            // Here is the first 32 threads operation, we need another 32 threads operation for 16 2b and 16 4b, totally need 32 loops
            float current_1st_zeros;
            float current_2nd_zeros;
            float current_1st_scales;

            current_1st_zeros = (float)((loop_packed_zeros_1st >> (ii * 2)) & 0x3);
            current_2nd_zeros = (float)((loop_packed_zeros_2nd >> (ii * 2)) & 0x3);
            current_1st_scales = (float)((loop_packed_scales_1st >> (ii * 2)) & 0x3);

            half scaling_factor_2nd = scales_2nd[oc_idx / second_order_group_size * 192 + packed_group_idx * 192/2 + threadIdx.x * 3 + ii];
            float scaling_factor = __half2float(scaling_factor_2nd) * (current_1st_scales - current_2nd_zeros);
            // if(blockIdx.x == 0 && blockIdx.y == DEBUG_THY && threadIdx.x == 0 && threadIdx.y == 0)// && packed_group_idx == 0)
            // printf("[Loop %0d]: scale=%f, 1st_scale=%f, 2nd_zero=%f, 2nd_scale=%0f\n", ii, scaling_factor, current_1st_scales, current_2nd_zeros, scaling_factor_2nd);
            // __shared__ double4 shared_packed_inputs[32];
            // shared_packed_inputs[threadIdx.x] = inputs[packed_group_idx * 128  + threadIdx.x * 4];

            // __syncthreads();

            uint32_t current_packed_weight = packed_weights[ii];
            half packed_inputs[16];
            *((double4 *)packed_inputs) = *(shared_packed_inputs + ii);

#pragma unroll
            for (int jj = 0; jj < 16; jj++)
            {
                float current_single_weight_fp = (float)(current_packed_weight & 0x3);
                float dequantized_weight = scaling_factor * (current_single_weight_fp - current_1st_zeros);
                // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && packed_group_idx == 0)
                // printf("[%0d,%0d]: dq_weight=%f, input=%f, current_weight=%f, scale=%f, zero=%f, %X\n", ii, jj, dequantized_weight, packed_inputs[jj], current_single_weight_fp, scaling_factor, current_1st_zeros, current_packed_weight);
                psum += dequantized_weight * __half2float(packed_inputs[jj]);
                // if(blockIdx.x == 0 && blockIdx.y == DEBUG_THY && threadIdx.x == 0 && threadIdx.y == 0)// && packed_group_idx == 0)
                // printf("[%0d,%0d]: psum=%f, dq_weight=%f, input=%f, current_weight=%f, scale=%f, zero=%f, %X\n", ii, jj, psum, dequantized_weight, packed_inputs[jj], current_single_weight_fp, scaling_factor, current_1st_zeros, current_packed_weight);
                current_packed_weight = current_packed_weight >> 2;
            }
        }
        // calculate 16 4b weight and activations
        {
            uint32_t current_packed_weight = packed_weights[3];
            uint32_t current_packed_weight_last = packed_weights_last;

            // __shared__ double4 shared_packed_inputs[32];
            // shared_packed_inputs[threadIdx.x] = inputs[packed_group_idx * 128  + threadIdx.x * 4];

            // __syncthreads();
            half packed_inputs[16];
            *((double4 *)packed_inputs) = *(shared_packed_inputs + 3);

#pragma unroll
            for (int jj = 0; jj < 8; jj++)
            {
                float current_single_weight_fp = (float)(current_packed_weight & 0xF);
                float dequantized_weight = scaling_factor_4b * (current_single_weight_fp - zeros_4b);
                // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0)
                // printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
                psum += dequantized_weight * __half2float(packed_inputs[jj]);
                // if(blockIdx.x == 0 && blockIdx.y == DEBUG_THY && threadIdx.x == 0 && threadIdx.y == 0)// && packed_group_idx == 0)
                // printf("[3,%0d]: psum=%f, dq_weight=%f, input=%f, current_weight=%f, scale=%f, zero=%f, %X\n", jj, psum, dequantized_weight, packed_inputs[jj], current_single_weight_fp, scaling_factor_4b, zeros_4b, current_packed_weight);
                current_packed_weight = current_packed_weight >> 4;
            }

#pragma unroll
            for (int jj = 0; jj < 8; jj++)
            {
                float current_single_weight_fp = (float)(current_packed_weight_last & 0xF);
                float dequantized_weight = scaling_factor_4b * (current_single_weight_fp - zeros_4b);
                // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0)
                // printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
                psum += dequantized_weight * __half2float(packed_inputs[jj + 8]);
                // if(blockIdx.x == 0 && blockIdx.y == DEBUG_THY && threadIdx.x == 0 && threadIdx.y == 0)// && packed_group_idx == 0)
                // printf("[3,%0d]: psum=%f, dq_weight=%f, input=%f, current_weight=%f, scale=%f, zero=%f, %X\n", (jj+8), psum, dequantized_weight, packed_inputs[jj+8], current_single_weight_fp, scaling_factor_4b, zeros_4b, current_packed_weight);
                current_packed_weight_last = current_packed_weight_last >> 4;
            }
        }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0)
    {
        outputs[oc_idx] = __float2half(psum);
    }

}


/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC, IC // 8];
  _zeros: int tensor of shape [OC, IC // G // 8];
  _scaling_factors: tensor of shape [OC, IC // G];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;

Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_mxq_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _kernel_last,
    // torch::Tensor _scaling_factors,
    // torch::Tensor _scales, // 2b
    // torch::Tensor _zeros,  
    torch::Tensor _zeros_and_scales,
    torch::Tensor _scales_2nd, // fp16
    torch::Tensor _zeros_2nd, // 2b
    torch::Tensor _scales_4b, // fp16
    torch::Tensor _zeros_4b, // 2b
    int group_size)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<double4*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto kernel_last = reinterpret_cast<uint32_t*>(_kernel_last.data_ptr<int>());
    // auto scales = reinterpret_cast<uint32_t*>(_scales.data_ptr<int>());
    // auto zeros = reinterpret_cast<uint32_t*>(_zeros.data_ptr<int>());
    auto zeros_and_scales = reinterpret_cast<uint32_t*>(_zeros_and_scales.data_ptr<int>());
    auto scales_2nd = reinterpret_cast<half*>(_scales_2nd.data_ptr<at::Half>());
    auto zeros_2nd = reinterpret_cast<uint32_t*>(_zeros_2nd.data_ptr<int>());
    auto scales_4b = reinterpret_cast<half*>(_scales_4b.data_ptr<at::Half>());
    auto zeros_4b = reinterpret_cast<uint32_t*>(_zeros_4b.data_ptr<int>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [OC, IC]
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(0)}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    int blockDim_z = num_out_feats;
    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
    dim3 num_threads(32, 4);
    if (group_size == 16)
    {
        gemv_mxq_kernel_g16_v0<<<num_blocks, num_threads>>>(
            // pointers
            in_feats, kernel, kernel_last, zeros_and_scales, scales_2nd, zeros_2nd, scales_4b, zeros_4b, out_feats,
            // constants
            num_in_channels, num_out_channels
        );
    }
    return _out_feats;
;}


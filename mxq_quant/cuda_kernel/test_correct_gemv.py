import torch
import torch.nn as nn

import mxq_inference_engine


DEV = torch.device('cuda')

M = 1
N = 4096
K = 4096

DTYPE = torch.half

groupsize_1st = 16
groupsize_2nd = 4
pack_num_2b = 16

zeros_and_scales_1st_val = 0xAA55_AA55 # 1010_1010_0101_0101_1010_1010_0101_0101  {8b_scale}_{8b_zero}_{8b_scale}_{8b_zero}
zeros_2nd_val = 0x5555_5555 # 0101
scales_2nd_val = 1

scales_and_zeros_1st = torch.ones((N, int(K/groupsize_1st/pack_num_2b) * 2), device=DEV, dtype=torch.int) * zeros_and_scales_1st_val
zeros_2nd = torch.ones((int(N/groupsize_2nd), int(K/groupsize_1st/pack_num_2b * 2)), device=DEV, dtype=torch.int) * zeros_2nd_val
scales_2nd = torch.ones((int(N/groupsize_2nd), int(K/groupsize_1st)), device=DEV, dtype=DTYPE) * scales_2nd_val

weight_val_2b = 0xAAAA_AAAA
weight_val_4b = 0xAAAA_AAAA

weight = torch.ones((N, 256), device=DEV, dtype=torch.int) * weight_val_2b
weight_last = torch.ones((N, 64), device=DEV, dtype=torch.int) * weight_val_4b


zeros_4b_val = 0x9999_9999
scales_4b_val = 1
scales_4b = torch.ones(N, device=DEV, dtype=DTYPE) * scales_4b_val
zeros_4b = torch.ones(int(N/8), device=DEV, dtype=torch.int) * zeros_4b_val

##########################################

input_X = torch.ones((M, K), device=DEV, dtype=DTYPE)

C = torch.zeros((M, N), device=DEV, dtype=DTYPE)

##########################################
# torch.matmul(input_X, B, out=C) 
# torch.cuda.synchronize()

C = mxq_inference_engine.gemv_mxq_forward_cuda(input_X, weight, weight_last, scales_and_zeros_1st, scales_2nd, zeros_2nd, scales_4b, zeros_4b, groupsize_1st)
torch.cuda.synchronize()

C_cmp = torch.ones((M, N), device=DEV, dtype=torch.int) * 4096
assert torch.allclose(C.int(), C_cmp)
print('------------------------------------')
print('------------------------------------')
print('------------ Check Pass! -----------')
print('------------------------------------')
print('------------------------------------')
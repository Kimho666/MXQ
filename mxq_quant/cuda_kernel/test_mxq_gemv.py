import torch
import torch.nn as nn

import mxq_inference_engine

#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False


DEV = torch.device('cuda')

# M = 12288 * 4
# N = 12288
M = 1
N = 4096
K = 4096


DTYPE = torch.half
B = torch.randn((K, N), device=DEV, dtype=DTYPE)
A = torch.randn((M, K), device=DEV, dtype=DTYPE)
C = torch.zeros((M, N), device=DEV, dtype=DTYPE)

COUNT = 100000
import time
torch.matmul(A, B, out=C) 
torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    torch.matmul(A, B, out=C) 
    torch.cuda.synchronize()
print('FP16:', (time.time() - tick) / COUNT *1000,'ms')
t_fp16 = (time.time() - tick) / COUNT


DTYPE = torch.half
group_size = 128
pack_num = 8 # int32 = 4b * 8
B = torch.randn((N, int(K/pack_num)), device=DEV, dtype=DTYPE)
A = torch.randn((M, K), device=DEV, dtype=DTYPE)
C = torch.zeros((M, N), device=DEV, dtype=DTYPE)
DTYPE = torch.half
B = B.to(torch.int)
A = A.to(DTYPE)
C = C.to(DTYPE)


B = torch.randint(-1000000000, 1000000000, (N, int(K/pack_num)), device=DEV, dtype=torch.int)
scales = torch.randn((N, int(K/group_size)), device=DEV, dtype=DTYPE)
# zeros = torch.randn(N, device=DEV, dtype=torch.int)
zeros = torch.ones((N, int(K/group_size/pack_num)), device=DEV, dtype=torch.int)


import time
tick = time.time()
for _ in range(COUNT):
    C = mxq_inference_engine.gemv_forward_cuda(A, B, scales, zeros, group_size)
    torch.cuda.synchronize()
print('awq_4bit:', (time.time() - tick) / COUNT *1000,'ms')
t_awq = (time.time() - tick) / COUNT
print(f'speedup with fp16 {t_fp16/t_awq:.4f}X')

group_size = 16
groupsize_2nd = 4
pack_num = 16
B = torch.randint(-1000000000, 1000000000, (N, int(K/pack_num)), device=DEV, dtype=torch.int)
scales_1nd = torch.ones((N, int(K/group_size/pack_num) * 2), device=DEV, dtype=torch.int)
scales_2nd = torch.randn((int(N/groupsize_2nd), int(K/group_size)), device=DEV, dtype=DTYPE)
zeros_1nd = zeros = torch.ones((N, int(K/group_size/pack_num)), device=DEV, dtype=torch.int)
zeros_2nd = zeros = torch.ones((int(N/groupsize_2nd), int(K/group_size/pack_num)), device=DEV, dtype=torch.int)

scales_4b = torch.randn(N, device=DEV, dtype=DTYPE)
zeros_4b = zeros = torch.ones(int(N/8), device=DEV, dtype=torch.int)

import time
tick = time.time()
for _ in range(COUNT):
    C = mxq_inference_engine.gemv_mxq_forward_cuda(A, B, B, scales_1nd, scales_2nd, zeros_2nd, scales_4b, zeros_4b, group_size)
    torch.cuda.synchronize()
print('mxq_2.8bit:', (time.time() - tick) / COUNT *1000,'ms')
t_mxq = (time.time() - tick) / COUNT
print(f'speedup with fp16 {t_fp16/t_mxq:.4f}X')


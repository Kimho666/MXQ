import math
import time

import torch
import torch.nn as nn
import transformers
from .quantizer import Quantizer
from .weight_permutation import get_permutation_order

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# RTN
class MXQGPT0:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.save_quant_dict = {}

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        #print(f"debug:{self.nsamples},{tmp},{self.H},{inp.shape}")
        self.save_quant_dict['inp'] = inp
        self.save_quant_dict['nsamples'] = self.nsamples
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        


    def fasterquant(
        self, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0


        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            W1 = W[:, i1:i2].clone()
            # print(f'debug:{W1}')
            quantizer = Quantizer()
            quantizer.configure(bits=2, perchannel=True, sym=False, qq_scale_bits=4)
            quantizer.find_params(W1, weight=True)
            W1 = quantizer.quantize_dequantize(W1)
            W[:, i1:i2] = W1
            # print(f'debug_after:{W1}')

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# SpRTN
class MXQGPT1:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.outlier_num = 0
        self.ol_ratio = .0
        self.nsamples = 0
        self.unstructured_outlier_mask = torch.zeros_like(W, dtype=torch.bool)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        #print(f"debug:{self.nsamples},{tmp},{self.H.shape},{inp.shape}")
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


    def fasterquant(
        self, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        #print(damp)
        diag = torch.arange(self.columns, device=self.dev)
        #print(diag)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None
        bit = 1

        outlier_relative_threshold = 0.2 * 3
        outlier_scale = (W.var(dim=0) / torch.diag(Hinv).square()).mean().item()
        unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale


        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            W1_without_outliers = torch.zeros_like(W1)
            if bit == 1:
                ol_threshold = 1.1
                W1_mean = W1.abs().sum(dim=1,keepdim=True).expand_as(W1) / blocksize
                #print(W1_mean)
                # print(f'deBBBBBAbbb:{W1}')
                # print(f'deAAAAAAbbb:{W1_mean}')
                likely_unstructured_outlier_mask = (W1 > (ol_threshold * W1_mean)).float() + (W1 < (ol_threshold * (-W1_mean))).float()
                assert likely_unstructured_outlier_mask.max() != 2
                #print(likely_unstructured_outlier_mask.sum(dim=1,keepdim=True))
                likely_unstructured_outlier_mask = (likely_unstructured_outlier_mask.sum(dim=1,keepdim=True) > 4).float().expand_as(likely_unstructured_outlier_mask)

                # W1_avg = (
                #     (W1 * (1-likely_unstructured_outlier_mask)).abs().sum(dim=1,keepdim=True) / 
                #     (1-likely_unstructured_outlier_mask).sum(dim=1,keepdim=True)
                # )
                # # print(f'debbb:{W1_avg}')

                # W1_without_outliers = W1 * (1-likely_unstructured_outlier_mask)  + W1_avg * likely_unstructured_outlier_mask
                self.unstructured_outlier_mask[:,i1:i2] = likely_unstructured_outlier_mask
                
                quantizer = Quantizer()
                quantizer.configure(bits=bit, perchannel=True, sym=False)
                quantizer.find_params(W1, weight=True)

                is_outlier = self.unstructured_outlier_mask[:,i1:i2].float()
                is_not_outlier = (1 - is_outlier)
                W[:, i1:i2] = quantizer.quantize_dequantize(W1) * is_not_outlier + W1 * is_outlier
                # print(f'debugCCCCCCC:{W[:,i1:i2]}')
            else:
                loo_quantization_error_sq = get_leave_one_out_error(
                    W1, torch.diag(Hinv1), bits=bit, sym=False
                )
                likely_unstructured_outlier_mask = (
                    loo_quantization_error_sq > unstructured_outlier_threshold
                ).float()
                non_outlier_mask = 1 - likely_unstructured_outlier_mask
                mean_over_non_outliers = torch.sum(
                    W1 * non_outlier_mask, dim=1, keepdim=True
                ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                W1_without_outliers = (
                    W1 * non_outlier_mask + mean_over_non_outliers * likely_unstructured_outlier_mask
                )
                # if bit==1:
                #     all_zero_w_mask = (torch.sum(W1_without_outliers.abs(), dim=1, keepdim=True).expand_as(W1) == 0).float()
                #     W1_without_outliers = W1_without_outliers + torch.sum(W1.abs(), dim=1, keepdim=True) / blocksize * all_zero_w_mask


                quantizer = Quantizer()
                quantizer.configure(bits=bit, perchannel=True, sym=False)
                quantizer.find_params(W1_without_outliers, weight=True)

                W_quant = quantizer.quantize_dequantize(W1).reshape_as(W1)

                W_quant_err = (W1 - W_quant) / torch.diag(Hinv1)
                self.unstructured_outlier_mask[:,i1:i2] = (
                    W_quant_err.square() > unstructured_outlier_threshold
                )

                is_outlier = self.unstructured_outlier_mask[:,i1:i2].float()
                is_not_outlier = (1 - is_outlier)
                W[:, i1:i2] = quantizer.quantize_dequantize(W1) * is_not_outlier + W1 * is_outlier

            # for i in range(count):
            #     w = W1[:, i]
            #     d = Hinv1[i, i]

            #     q = w.clone()
            #     #w_quant = quantizer.quantize_dequantize(w)
            #     w_quant = torch.zeros_like(w)
            #     w_quant = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)

            #     quant_err = (w - w_quant) / d
            #     self.unstructured_outlier_mask[:,i+i1] = (
            #         quant_err.square() > unstructured_outlier_threshold
            #     )
            #     is_outlier = self.unstructured_outlier_mask[:,i+i1].float()
            #     is_not_outlier = (1 - is_outlier)
            #     q = (
            #         (quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)*is_not_outlier + 
            #          is_outlier * w
            #         )
            #     )
            #     Q1[:, i] = q
            # W[:, i1:i2] = Q1
        
        self.ol_ratio = self.unstructured_outlier_mask.float().sum() / (self.rows*self.columns)
        print(f"[Info]:outlier ratio is {self.ol_ratio*100:.4f}%")

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# NAS-QAT-RTN
class MXQGPT2:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.save_quant_dict = {}

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        #print(f"debug:{self.nsamples},{tmp},{self.H},{inp.shape}")
        self.save_quant_dict['inp'] = inp
        self.save_quant_dict['nsamples'] = self.nsamples
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        


    def fasterquant(
        self, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        ratio_2b = 0.6
        ratio_4b = 1 - ratio_2b


        start_4b = 0
        for i in range(0, int(self.columns * ratio_2b), blocksize):
            i1 = i
            i2 = min(i + blocksize, self.columns)
            start_4b = i2
            W1 = W[:, i1:i2].clone()
            # print(f'debug:{W1}')
            quantizer = Quantizer()
            quantizer.configure(bits=2, perchannel=True, sym=False, qq_scale_bits=4)
            quantizer.find_params(W1, weight=True)
            W1 = quantizer.quantize_dequantize(W1)
            W[:, i1:i2] = W1
     
        i1 = start_4b
        i2 = self.columns

        quantizer_4b = Quantizer()
        quantizer_4b.configure(bits=4, perchannel=True, sym=False, qq_scale_bits=4)
        quantizer_4b.find_params(W[:,i1:i2], weight=True)
        W1_4b = quantizer_4b.quantize_dequantize(W[:,i1:i2])
        W[:, i1:i2] = W1_4b

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# NAS-QAT-RTN
class MXQGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.save_quant_dict = {}

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        #print(f"debug:{self.nsamples},{tmp},{self.H},{inp.shape}")
        self.save_quant_dict['inp'] = inp
        self.save_quant_dict['nsamples'] = self.nsamples
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        


    def fasterquant(
        self, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        ratio_2b = 6/8
        ratio_4b = 1 - ratio_2b
        num_4b = 64 - int(64*ratio_2b)

        i1 = 0
        i2 = int(self.columns * ratio_4b)
        W_another = W.clone()

        W_4b = torch.zeros((self.rows,int(self.columns/64 * num_4b)),device=self.dev)
        for ii in range(0, self.columns, 64):
            
            ii1 = ii
            ii_4b = int(ii1 / 64)
            ii2 = min(ii + int(64 * ratio_2b), self.columns)
            ii3 = min(ii + 64, self.columns)
            # print(f'OUT LOOP:start={ii1}, 2b_end={ii2}, 4b_end={ii3}, ii_4b={ii_4b}')
            for jj in range(ii1, ii2, blocksize): # 2b part
                i1 = jj
                i2 = min(jj + blocksize, ii2)
                W1 = W[:, i1:i2].clone()
                quantizer = Quantizer()
                quantizer.configure(bits=2, perchannel=True, sym=False, qq_scale_bits=4)
                quantizer.find_params(W1, weight=True)
                W1 = quantizer.quantize_dequantize(W1)
                W[:, i1:i2] = W1
                #print(f'INN LOOP:start={i1}, 2b_end={i2}')

            W_4b[:,ii_4b*num_4b:(ii_4b+1)*num_4b] = W[:,ii2:ii3].clone()

        quantizer_4b = Quantizer()
        quantizer_4b.configure(bits=4, perchannel=True, sym=False, qq_scale_bits=4)
        quantizer_4b.find_params(W_4b, weight=True)
        W_4b_quant = quantizer_4b.quantize_dequantize(W_4b)

        for ii in range(0, self.columns, 64):
            ii1 = ii
            ii_4b = int(ii1 / 64)
            ii2 = min(ii + int(64 * ratio_2b), self.columns)
            ii3 = min(ii + 64, self.columns)
            W[:,ii2:ii3] = W_4b_quant[:,ii_4b*num_4b:(ii_4b+1)*num_4b]
        
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

def get_leave_one_out_error(group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, *, bits, sym):
    """EXPERIMENTAL! BEWARE - for each weight, fit quantizer without this_one_weight and return this one weight's reconstruction"""

    assert group_weight.ndim == 2
    loo_indices = torch.arange(group_weight.shape[1], device=group_weight.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
    groupwise_loo_data = group_weight[:, loo_indices]  # [num_groups, num_loo = groupsize, groupsize - 1]
    fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
    fast_quantizer.configure(bits, perchannel=True, sym=sym)
    fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

    # compute error improvement from not quantizing each one weight
    # to do so, we shall first train quantizer on leave-one-out data (which can be done faster since not all data affects quantization)
    loo_groupwise_reconstructed_weights = fast_quantizer.quantize_dequantize(
        groupwise_loo_data.flatten(0, 1)
    ).reshape_as(groupwise_loo_data)
    loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]  # [num_loo = groupsize, groupsize - 1]
    assert group_diag_hessian_inv_cho.ndim == 1

    # total quantization error consists of hessian-weighted mse on all remaining weights except for the one that's left out
    # -- this is because the left-out weights will not be quantized, and therefore, has zero quantization error
    loo_errors_sq = (
        ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
    )
    assert loo_errors_sq.shape == group_weight.shape  # [num_groups, num_loo = groupsize]

    # as a baseline error, quantize data normally without outliers
    base_quantizer = Quantizer(shape=group_weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(group_weight, weight=True)
    baseline_reconstructed_weights = base_quantizer.quantize_dequantize(group_weight)
    baseline_errors_sq = (
        ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
    )

    # outlier's usefulness = how much does mse decrease from treating this weight as an outlier
    reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
    return reduction_in_squared_error


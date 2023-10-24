import math
import time

import torch
import torch.nn as nn
import transformers
from .quantizer import Quantizer
from .weight_permutation import get_permutation_order

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
class SparseGPT0:

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

#sparsegpt with permutation
class SparseGPT1_0:

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, reorder = 0
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if reorder==1:
            perm = get_permutation_order(self.H, W, "sparse_act_order")
        else:
            perm = get_permutation_order(self.H, W, "identity")
        W = W[:, perm]

        H = self.H
        H = H[perm][:, perm]
        #H = self.H
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        invperm = torch.argsort(perm)
        W = W[:, invperm]
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

#sparsegpt with 2-order permutation
class SparseGPT1:

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()
        #order_v2_size = 1024
        order_v2_size = self.rows
        for row_idx_1 in range(0, self.rows, order_v2_size):
            row_idx_2 = min(row_idx_1 + order_v2_size, self.rows)
            row_idx_count = row_idx_2 - row_idx_1
            W_block = W[row_idx_1:row_idx_2,:]
            print(f'Row_idx={row_idx_1},size={row_idx_count}')
            perm = get_permutation_order(self.H, W_block, "sparse_act_order")
            W_block = W_block[:, perm]

            H = self.H
            H = H[perm][:, perm]
            #H = self.H
            #del self.H
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W_block[:, dead] = 0

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

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W_block[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prune_n == 0: 
                    if mask is not None:
                        mask1 = mask[:, i1:i2]
                    else:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        mask1 = tmp <= thresh
                else:
                    mask1 = torch.zeros_like(W1) == 1

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if prune_n != 0 and i % prune_m == 0:
                        tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                    q = w.clone()
                    q[mask1[:, i]] = 0

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d 
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W_block[:, i1:i2] = Q1
                #Losses += torch.sum(Losses1, 1) / 2

                W_block[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            invperm = torch.argsort(perm)
            W[row_idx_1:row_idx_2,:] = W_block[:, invperm]
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# 1+16
class SparseGPT2:

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            
            mask1_pos = mask1 & (W1>0)
            mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            W1_total = (W1 * mask1.float()).abs().sum(dim=1)
            W1_numel = (mask1.float()).sum(dim=1) + 1e-9
            W1_avg = W1_total / W1_numel

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                q[mask1_pos[:, i]] = W1_avg[mask1_pos[:, i]]
                q[mask1_neg[:, i]] = (-1)*W1_avg[mask1_neg[:, i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# 1+4 1 scale
class SparseGPT3:

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            mask1_pos = mask1 & (W1>0)
            mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            W1_total = (W1 * mask1.float()).abs().sum(dim=1)
            W1_numel = (mask1.float()).sum(dim=1) + 1e-9
            W1_avg = W1_total / W1_numel

            # find other higher weight and quantize to 4b
            mask1_4b = ~ mask1
            quantizer = Quantizer()
            quantizer.configure(bits=4, perchannel=True, sym=False)
            W1_without_1b = W1 * mask1_4b.float()
            quantizer.find_params(W1_without_1b, weight=True)

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                q[mask1_pos[:, i]] = W1_avg[mask1_pos[:, i]]
                q[mask1_neg[:, i]] = (-1)*W1_avg[mask1_neg[:, i]]
                #w_quant = quantizer.quantize_dequantize(w)
                q[mask1_4b[:, i]] = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1_4b[:, i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# 1+4 2 scales
class SparseGPT4:

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            mask1_pos = mask1 & (W1>=0)
            mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            W1_pos_total = (W1 * mask1_pos.float()).abs().sum(dim=1)
            W1_pos_numel = (mask1_pos.float()).sum(dim=1) + 1e-9
            W1_pos_avg = W1_pos_total / W1_pos_numel
            W1_neg_total = (W1 * mask1_neg.float()).abs().sum(dim=1)
            W1_neg_numel = (mask1_neg.float()).sum(dim=1) + 1e-9
            W1_neg_avg = W1_neg_total / W1_neg_numel

            # find other higher weight and quantize to 4b
            mask1_4b = ~ mask1
            quantizer = Quantizer()
            quantizer.configure(bits=4, perchannel=True, sym=False)
            W1_without_1b = W1 * mask1_4b.float()
            quantizer.find_params(W1_without_1b, weight=True)

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                q[mask1_pos[:, i]] = W1_pos_avg[mask1_pos[:, i]]
                q[mask1_neg[:, i]] = (-1)*W1_neg_avg[mask1_neg[:, i]]
                #w_quant = quantizer.quantize_dequantize(w)
                q[mask1_4b[:, i]] = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1_4b[:, i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# RTN
class SparseGPT5:

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
        


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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
class SparseGPT5:

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

# our method 1+4+16 1 scale
class SparseGPT6:

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
        self.outlier_num = 0
        self.ol_ratio = .0
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
        #print(f"debug:{self.nsamples},{tmp},{self.H.shape},{inp.shape}")
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        mask_1_4_16 = torch.zeros_like(W)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if prune_n != 0 and i % prune_m == 0:
                        tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            #mask1 = (mask1.float() - mask1_outlier_1b.float()).bool()
            # mask1_pos = mask1 & (W1>0)
            # mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            
            W1_total = (W1 * mask1.float()).abs().sum(dim=1)
            W1_numel = (mask1.float()).sum(dim=1) + 1e-9
            W1_avg = W1_total / W1_numel

            # find other higher weight and quantize to 4b
            mask1_4b = ~ mask1

            # detect outlier with fixed threshold from 1b matrix
            threshold = 3
            mask1_outlier = (W1.abs() > (threshold * W1_avg.unsqueeze(1).repeat(1,count))) & mask1
            self.outlier_num += mask1_outlier.float().sum()
            mask1 = (mask1.float() - mask1_outlier.float()).bool()
            mask1_pos = mask1 & (W1>=0)
            mask1_neg = mask1 & (W1<0)
    
            # detect outlier with fixed threshold from 4b matrix
            # W1_avg_per_block = (W1.abs().sum(dim=1) / count).unsqueeze(1)
            # threshold = 10
            # mask1_outlier = (W1.abs() > (threshold * W1_avg_per_block.repeat(1,count))) & mask1_4b
            # self.outlier_num += mask1_outlier.float().sum()
            # mask1_4b = (mask1_4b.float() - mask1_outlier.float()).bool()

            quantizer = Quantizer()
            quantizer.configure(bits=4, perchannel=True, sym=False)
            W1_without_1b = W1 * mask1_4b.float()
            quantizer.find_params(W1_without_1b, weight=True)

            mask_1_4_16[:, i1:i2] = mask1.float() + 4 * mask1_4b.float() + 16 * mask1_outlier.float()

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                q[mask1_pos[:, i]] = W1_avg[mask1_pos[:, i]]
                q[mask1_neg[:, i]] = (-1)*W1_avg[mask1_neg[:, i]]
                #w_quant = quantizer.quantize_dequantize(w)
                q[mask1_4b[:, i]] = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1_4b[:,i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.ol_ratio = self.outlier_num / (self.rows*self.columns)
        print(f"outlier ratio is {self.ol_ratio:.10f}")
        self.save_quant_dict["quant_mask"] = mask_1_4_16
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# our method 1+4+16 2 scale
class SparseGPT7:

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
        self.outlier_num = 0
        self.ol_ratio = .0

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            #mask1 = (mask1.float() - mask1_outlier_1b.float()).bool()
            mask1_pos = mask1 & (W1>=0)
            mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            
            # W1_total = (W1 * mask1.float()).abs().sum(dim=1)
            # W1_numel = (mask1.float()).sum(dim=1) + 1e-9
            # W1_avg = W1_total / W1_numel

            W1_pos_total = (W1 * mask1_pos.float()).abs().sum(dim=1)
            W1_pos_numel = (mask1_pos.float()).sum(dim=1) + 1e-9
            W1_pos_avg = W1_pos_total / W1_pos_numel
            W1_neg_total = (W1 * mask1_neg.float()).abs().sum(dim=1)
            W1_neg_numel = (mask1_neg.float()).sum(dim=1) + 1e-9
            W1_neg_avg = W1_neg_total / W1_neg_numel

            # find other higher weight and quantize to 4b
            mask1_4b = ~ mask1

            # detect outlier with fixed threshold from 1b matrix
            threshold = 3
            mask1_pos_outlier = (W1.abs() > (threshold * W1_pos_avg.unsqueeze(1).repeat(1,count))) & mask1_pos
            self.outlier_num += mask1_pos_outlier.float().sum()
            mask1_neg_outlier = (W1.abs() > (threshold * W1_neg_avg.unsqueeze(1).repeat(1,count))) & mask1_neg
            self.outlier_num += mask1_neg_outlier.float().sum()

            mask1 = (mask1.float() - mask1_pos_outlier.float() - mask1_pos_outlier.float()).bool()
            # mask1_pos = mask1 & (W1>0)
            # mask1_neg = mask1 & (W1<0)
    
            # detect outlier with fixed threshold from 4b matrix
            # W1_avg_per_block = (W1.abs().sum(dim=1) / count).unsqueeze(1)
            # threshold = 10
            # mask1_outlier = (W1.abs() > (threshold * W1_avg_per_block.repeat(1,count))) & mask1_4b
            # self.outlier_num += mask1_outlier.float().sum()
            # mask1_4b = (mask1_4b.float() - mask1_outlier.float()).bool()

            quantizer = Quantizer()
            quantizer.configure(bits=4, perchannel=True, sym=False)
            W1_without_1b = W1 * mask1_4b.float()
            quantizer.find_params(W1_without_1b, weight=True)

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                q[mask1_pos[:, i]] = W1_pos_avg[mask1_pos[:, i]]
                q[mask1_neg[:, i]] = (-1)*W1_neg_avg[mask1_neg[:, i]]
                #w_quant = quantizer.quantize_dequantize(w)
                q[mask1_4b[:, i]] = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1_4b[:,i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.ol_ratio = self.outlier_num / (self.rows*self.columns)
        print(f"outlier ratio is {self.ol_ratio:.10f}")
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# our method 1+4+16 1 scale not 0 offset
class SparseGPT8:

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
        self.outlier_num = 0
        self.ol_ratio = .0

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            #mask1 = (mask1.float() - mask1_outlier_1b.float()).bool()
            # mask1_pos = mask1 & (W1>0)
            # mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            
            W1_total = (W1 * mask1.float()).abs().sum(dim=1)
            W1_numel = (mask1.float()).sum(dim=1) + 1e-9
            W1_avg = W1_total / W1_numel

            # find other higher weight and quantize to 4b
            mask1_4b = ~ mask1

            # detect outlier with fixed threshold from 1b matrix
            threshold = 3
            mask1_outlier = (W1.abs() > (threshold * W1_avg.unsqueeze(1).repeat(1,count))) & mask1
            self.outlier_num += mask1_outlier.float().sum()
            mask1 = (mask1.float() - mask1_outlier.float()).bool()
            mask1_pos = mask1 & (W1>0)
            mask1_neg = mask1 & (W1<0)
    
            # detect outlier with fixed threshold from 4b matrix
            # W1_avg_per_block = (W1.abs().sum(dim=1) / count).unsqueeze(1)
            # threshold = 10
            # mask1_outlier = (W1.abs() > (threshold * W1_avg_per_block.repeat(1,count))) & mask1_4b
            # self.outlier_num += mask1_outlier.float().sum()
            # mask1_4b = (mask1_4b.float() - mask1_outlier.float()).bool()

            quantizer = Quantizer()
            quantizer.configure(bits=4, perchannel=True, sym=False)
            W1_without_1b = W1 * mask1_4b.float()
            quantizer.find_params(W1_without_1b, weight=True)

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                q[mask1_pos[:, i]] = W1_avg[mask1_pos[:, i]]
                q[mask1_neg[:, i]] = (-1)*W1_avg[mask1_neg[:, i]]
                #w_quant = quantizer.quantize_dequantize(w)
                q[mask1_4b[:, i]] = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1_4b[:,i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.ol_ratio = self.outlier_num / (self.rows*self.columns)
        print(f"outlier ratio is {self.ol_ratio:.10f}")
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# our method 2+4+16
class SparseGPT9:

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
        self.outlier_num = 0
        self.ol_ratio = .0

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            #mask1 = (mask1.float() - mask1_outlier_1b.float()).bool()
            # mask1_pos = mask1 & (W1>0)
            # mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            
            W1_total = (W1 * mask1.float()).abs().sum(dim=1)
            W1_numel = (mask1.float()).sum(dim=1) + 1e-9
            W1_avg = W1_total / W1_numel

            # find other higher weight and quantize to 4b
            mask1_4b = ~ mask1

            # detect outlier with fixed threshold from 1b matrix
            threshold = 3
            mask1_outlier = (W1.abs() > (threshold * W1_avg.unsqueeze(1).repeat(1,count))) & mask1
            self.outlier_num += mask1_outlier.float().sum()
            mask1 = (mask1.float() - mask1_outlier.float()).bool()
            #mask1_pos = mask1 & (W1>0)
            #mask1_neg = mask1 & (W1<0)
    
            # detect outlier with fixed threshold from 4b matrix
            # W1_avg_per_block = (W1.abs().sum(dim=1) / count).unsqueeze(1)
            # threshold = 10
            # mask1_outlier = (W1.abs() > (threshold * W1_avg_per_block.repeat(1,count))) & mask1_4b
            # self.outlier_num += mask1_outlier.float().sum()
            # mask1_4b = (mask1_4b.float() - mask1_outlier.float()).bool()

            quantizer = Quantizer()
            quantizer.configure(bits=4, perchannel=True, sym=False)
            W1_without_1b = W1 * mask1_4b.float()
            quantizer.find_params(W1_without_1b, weight=True)

            quantizer_2 = Quantizer()
            quantizer_2.configure(bits=2, perchannel=True, sym=False)
            W1_1b = W1 * mask1.float()
            quantizer_2.find_params(W1_1b, weight=True)

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                #q[mask1_pos[:, i]] = W1_avg[mask1_pos[:, i]]
                #q[mask1_neg[:, i]] = (-1)*W1_avg[mask1_neg[:, i]]
                q[mask1[:,i]] = quantizer_2.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1[:,i]]
                #w_quant = quantizer.quantize_dequantize(w)
                q[mask1_4b[:, i]] = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1_4b[:,i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.ol_ratio = self.outlier_num / (self.rows*self.columns)
        print(f"outlier ratio is {self.ol_ratio:.10f}")
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# our method 1+4+16 1 scale with permutation
class SparseGPT10:

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
        self.outlier_num = 0
        self.ol_ratio = .0

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


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        perm = get_permutation_order(self.H, W, "sparse_act_order")
        W = W[:, perm]

        H = self.H
        H = H[perm][:, perm]
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

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            #mask1 = (mask1.float() - mask1_outlier_1b.float()).bool()
            # mask1_pos = mask1 & (W1>0)
            # mask1_neg = mask1 & (W1<0)
            #W1_max = W1[mask1].max(dim=1)[0]
            #W1_min = W1[mask1].min(dim=1)[0]
            
            W1_total = (W1 * mask1.float()).abs().sum(dim=1)
            W1_numel = (mask1.float()).sum(dim=1) + 1e-9
            W1_avg = W1_total / W1_numel

            # find other higher weight and quantize to 4b
            mask1_4b = ~ mask1

            # detect outlier with fixed threshold from 1b matrix
            threshold = 3
            mask1_outlier = (W1.abs() > (threshold * W1_avg.unsqueeze(1).repeat(1,count))) & mask1
            self.outlier_num += mask1_outlier.float().sum()
            mask1 = (mask1.float() - mask1_outlier.float()).bool()
            mask1_pos = mask1 & (W1>0)
            mask1_neg = mask1 & (W1<0)
    
            # detect outlier with fixed threshold from 4b matrix
            # W1_avg_per_block = (W1.abs().sum(dim=1) / count).unsqueeze(1)
            # threshold = 10
            # mask1_outlier = (W1.abs() > (threshold * W1_avg_per_block.repeat(1,count))) & mask1_4b
            # self.outlier_num += mask1_outlier.float().sum()
            # mask1_4b = (mask1_4b.float() - mask1_outlier.float()).bool()

            quantizer = Quantizer()
            quantizer.configure(bits=4, perchannel=True, sym=False)
            W1_without_1b = W1 * mask1_4b.float()
            quantizer.find_params(W1_without_1b, weight=True)

            # print(W1_avg.shape)
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if prune_n != 0 and i % prune_m == 0:
                #     tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                #q[mask1[:, i]] = 0
                q[mask1_pos[:, i]] = W1_avg[mask1_pos[:, i]]
                q[mask1_neg[:, i]] = (-1)*W1_avg[mask1_neg[:, i]]
                #w_quant = quantizer.quantize_dequantize(w)
                q[mask1_4b[:, i]] = quantizer.quantize_dequantize(w.unsqueeze(1)).reshape_as(w)[mask1_4b[:,i]]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        invperm = torch.argsort(perm)
        W = W[:, invperm]
        self.ol_ratio = self.outlier_num / (self.rows*self.columns)
        print(f"outlier ratio is {self.ol_ratio:.10f}")
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# NAS-QAT-RTN
class SparseGPT14:

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
        


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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
        # print(f'before:{W}')


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

            # print(f'debug_after:{W1}')
        # print(f'after:{W}')
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

# NAS-QAT-RTN
class SparseGPT:

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
        


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
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
        ratio_2b = 6/10
        ratio_4b = 1 - ratio_2b
        num_4b = 64 - int(64*ratio_2b)
        # print(f'before:{W}')
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
            
            # W_4b = W[:,ii2:ii3].clone()
            # quantizer_4b = Quantizer()
            # quantizer_4b.configure(bits=4, perchannel=True, sym=False, qq_scale_bits=4)
            # quantizer_4b.find_params(W_4b, weight=True)
            # W_4b = quantizer_4b.quantize_dequantize(W_4b)
            # W[:, ii2:ii3] = W_4b

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

        # print(f'origin:{W_another[0,32:48]}')
        # print(f'quant:{W[0,32:48]}')
        # print(f'w_4b_quant:{W_4b_quant[0,16:32]}')  
        # print(f'origin:{W_another[0,48+64:64+64]}')
        # print(f'quant:{W[0,48+64:64+64]}')
        # print(f'w_4b_quant:{W_4b_quant[0,16:32]}')
        

            # print(f'debug_after:{W1}')
        # print(f'after:{W}')
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


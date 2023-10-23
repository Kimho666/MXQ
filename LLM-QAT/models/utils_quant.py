# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn


class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0].cuda(), clip_val[1].cuda())
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                # groupwise
                max_input = torch.zeros_like(input).detach()
                groupsize = 128
                dim_group = int(input.shape[-1] / groupsize)
                for i in range(dim_group):
                    i1 = i*groupsize
                    i2 = min((i+1)*groupsize, input.shape[-1])
                    max_input[:, i1:i2] = (
                         torch.max(torch.abs(input[:, i1:i2]), dim=-1, keepdim=True)[0]
                         .expand_as(input[:, i1:i2])
                         .detach()
                    )
                # groupwise
                #max_input = (
                #    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                #    .expand_as(input)
                #    .detach()
                #)
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        output = torch.round(input * s).div(s + 1e-6)
        #print(f'DEBUG:num_bits={num_bits},input={input},s={s},output={output}')
        #print(f'quant_error={output-input}')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                # groupwise
                alpha = torch.zeros_like(input).detach()
                beta = torch.zeros_like(input).detach()
                groupsize = 8
                dim_group = int(input.shape[-1] / groupsize)
                for i in range(dim_group):
                    i1 = i*groupsize
                    i2 = min((i+1)*groupsize, input.shape[-1])
                    alpha[:, i1:i2] = (
                        (
                            input[:, i1:i2].max(dim=-1, keepdim=True)[0]
                            - input[:, i1:i2].min(dim=-1, keepdim=True)[0]
                        )
                        .expand_as(input[:, i1:i2])
                        .detach()
                    )
                    beta[:, i1:i2] = input[:, i1:i2].min(dim=-1, keepdim=True)[0].expand_as(input[:, i1:i2]).detach()
                # groupwise
                # gptq groupwise

                # gptq groupwise
                # alpha = (
                #     (
                #         input.max(dim=-1, keepdim=True)[0]
                #         - input.min(dim=-1, keepdim=True)[0]
                #     )
                #     .expand_as(input)
                #     .detach()
                # )
                # beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        #quant_input = torch.round(torch.round(input_normalized * s).div(s))
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

# NOT Use
class VecAsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                # groupwise
                alpha = torch.zeros_like(input).detach()
                beta = torch.zeros_like(input).detach()
                groupsize = 8
                vector_size = 2
                dim_group = int(input.shape[-1] / groupsize)
                output_tmp = torch.zeros_like(input)
                for i in range(dim_group):
                    i1 = i*groupsize
                    i2 = min((i+1)*groupsize, input.shape[-1])
                    alpha[:, i1:i2] = (
                        (
                            input[:, i1:i2].max(dim=-1, keepdim=True)[0]
                            - input[:, i1:i2].min(dim=-1, keepdim=True)[0]
                        )
                        .expand_as(input[:, i1:i2])
                        .detach()
                    )
                    beta[:, i1:i2] = input[:, i1:i2].min(dim=-1, keepdim=True)[0].expand_as(input[:, i1:i2]).detach()
                    for v_i in range(vector_size):
                        start = i1 + v_i * vector_size
                        end = i1 + (v_i + 1) * vector_size
                        output_tmp[:,start:end] = (
                            (
                                input[:,start:end].sum(dim=-1,keepdim=True) / vector_size
                            )
                            .expand_as(input[:,start:end])
                        )
                # groupwise
                # gptq groupwise

                # gptq groupwise
                # alpha = (
                #     (
                #         input.max(dim=-1, keepdim=True)[0]
                #         - input.min(dim=-1, keepdim=True)[0]
                #     )
                #     .expand_as(input)
                #     .detach()
                # )
                # beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (output_tmp - beta) / (alpha + 1e-8)
        # input_normalized = (input - beta) / (alpha + 1e-8)
        s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        #quant_input = torch.round(torch.round(input_normalized * s).div(s))
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

class MXAsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                # groupwise
                alpha = torch.zeros_like(input).detach()
                beta = torch.zeros_like(input).detach()
                s = torch.zeros_like(input).detach()
                groupsize = 16
                ratio_2b = 0.7
                ratio_2b = 6/8
                num_4b = 64 - int(64*ratio_2b)
                ratio_4b = 1 - ratio_2b
                #############################################
                # 2nd hessain 4b + 2b
                W_4b = torch.zeros((input.shape[0],int(input.shape[-1]/64 * num_4b)),device=alpha.device)
                a_4b = torch.zeros_like(W_4b).detach()
                for ii in range(0, input.shape[-1], 64):
                    ii1 = ii
                    ii_4b = int(ii1/64)
                    ii2 = min(ii + int(64 * ratio_2b), input.shape[-1])
                    ii3 = min(ii + 64, input.shape[-1])
                    for jj in range(ii1, ii2, groupsize): # 2b part
                        i1 = jj
                        i2 = min(jj + groupsize,ii2)
                        alpha[:, i1:i2] = (
                            (
                                input[:, i1:i2].max(dim=-1, keepdim=True)[0]
                                - input[:, i1:i2].min(dim=-1, keepdim=True)[0]
                            )
                            .expand_as(input[:, i1:i2])
                            .detach()
                        )
                        beta[:, i1:i2] = input[:, i1:i2].min(dim=-1, keepdim=True)[0].expand_as(input[:, i1:i2]).detach()
                        s[:, i1:i2] = 2**num_bits - 1
                    # print(f'{ii2},{ii3}')
                    W_4b[:,ii_4b*num_4b:(ii_4b+1)*num_4b] = input[:,ii2:ii3].clone()
                alpha_4b = (
                    (
                        W_4b.max(dim=-1, keepdim=True)[0]
                        - W_4b.min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(W_4b)
                    .detach()
                )
                beta_4b = W_4b.min(dim=-1,keepdim=True)[0].expand_as(W_4b).detach()
                for ii in range(0, input.shape[-1], 64):
                    ii1 = ii
                    ii_4b = int(ii1/64)
                    ii2 = min(ii + int(64 * ratio_2b), input.shape[-1])
                    ii3 = min(ii + 64, input.shape[-1])
                    alpha[:,ii2:ii3] = alpha_4b[:,ii_4b*num_4b:(ii_4b+1)*num_4b]
                    beta[:,ii2:ii3] = beta_4b[:,ii_4b*num_4b:(ii_4b+1)*num_4b]
                    s[:,ii2:ii3] = 2**4-1

                #############################################
                #############################################
                #############################################
                # front 2b + end 4b
                #############################################
                # 2b part
                # start_4b = 0
                # for i in range(0, int(input.shape[-1] * ratio_2b), groupsize):
                #     i1 = i
                #     i2 = min(i+groupsize, input.shape[-1])
                #     start_4b = i2
                #     alpha[:, i1:i2] = (
                #         (
                #             input[:, i1:i2].max(dim=-1, keepdim=True)[0]
                #             - input[:, i1:i2].min(dim=-1, keepdim=True)[0]
                #         )
                #         .expand_as(input[:, i1:i2])
                #         .detach()
                #     )
                #     beta[:, i1:i2] = input[:, i1:i2].min(dim=-1, keepdim=True)[0].expand_as(input[:, i1:i2]).detach()
                #     s[:, i1:i2] = 2**num_bits - 1
                # # s[:, 0:dim_group*groupsize] = 2**num_bits - 1

                # # 40% 4b layer wise
                # i1 = start_4b
                # i2 = input.shape[-1]
                # s[:, i1:i2] = 2**4 - 1
                # alpha[:, i1:i2] = (
                #     (
                #         input[:, i1:i2].max(dim=-1, keepdim=True)[0]
                #         - input[:, i1:i2].min(dim=-1, keepdim=True)[0]
                #     )
                #     .expand_as(input[:, i1:i2])
                #     .detach()
                # )
                # beta[:, i1:i2] = input[:, i1:i2].min(dim=-1, keepdim=True)[0].expand_as(input[:, i1:i2]).detach()
                ##################################
                # groupwise
                # gptq groupwise

                # gptq groupwise
                # alpha = (
                #     (
                #         input.max(dim=-1, keepdim=True)[0]
                #         - input.min(dim=-1, keepdim=True)[0]
                #     )
                #     .expand_as(input)
                #     .detach()
                # )
                # beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        #s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        #quant_input = torch.round(torch.round(input_normalized * s).div(s))
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

class MX1AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                # groupwise
                alpha = torch.zeros_like(input).detach()
                beta = torch.zeros_like(input).detach()
                s = torch.zeros_like(input).detach()
                groupsize = 32
                ratio_2b = 0.6
                #ratio_2b = 6/8
                ratio_4b = 1 - ratio_2b
                #############################################
                #############################################
                # front 2b + end 4b
                #############################################
                # 2b part
                start_4b = 0
                for i in range(0, int(input.shape[-1] * ratio_2b), groupsize):
                    i1 = i
                    i2 = min(i+groupsize, input.shape[-1])
                    start_4b = i2
                    alpha[:, i1:i2] = (
                        (
                            input[:, i1:i2].max(dim=-1, keepdim=True)[0]
                            - input[:, i1:i2].min(dim=-1, keepdim=True)[0]
                        )
                        .expand_as(input[:, i1:i2])
                        .detach()
                    )
                    beta[:, i1:i2] = input[:, i1:i2].min(dim=-1, keepdim=True)[0].expand_as(input[:, i1:i2]).detach()
                    s[:, i1:i2] = 2**num_bits - 1
                # # s[:, 0:dim_group*groupsize] = 2**num_bits - 1

                # # 40% 4b layer wise
                i1 = start_4b
                i2 = input.shape[-1]
                s[:, i1:i2] = 2**4 - 1
                alpha[:, i1:i2] = (
                    (
                        input[:, i1:i2].max(dim=-1, keepdim=True)[0]
                        - input[:, i1:i2].min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(input[:, i1:i2])
                    .detach()
                )
                beta[:, i1:i2] = input[:, i1:i2].min(dim=-1, keepdim=True)[0].expand_as(input[:, i1:i2]).detach()
                ##################################
                # groupwise
                # gptq groupwise

                # gptq groupwise
                # alpha = (
                #     (
                #         input.max(dim=-1, keepdim=True)[0]
                #         - input.min(dim=-1, keepdim=True)[0]
                #     )
                #     .expand_as(input)
                #     .detach()
                # )
                # beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        #s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        #quant_input = torch.round(torch.round(input_normalized * s).div(s))
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=32,
        a_bits=32,
        act_layerwise=False,
        weight_layerwise=False,
        is_qk=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise
        self.is_qk = is_qk
        # params for weight quant
        # if self.w_bits < 32:
        #     self.weight_clip_val = Parameter(torch.tensor([-2.0, 2.0]), requires_grad=False)
        if self.a_bits < 32 and self.a_bits > 2:
            if symmetric:
                self.act_quantizer = SymQuantizer
            else:
                self.act_quantizer = AsymQuantizer

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 32:
            weight = self.weight
        elif self.w_bits >= 2:
            weight_clip_val = torch.tensor([-2.0, 2.0])
            weight = MXAsymQuantizer.apply(
                real_weights, weight_clip_val, self.w_bits, self.weight_layerwise
            )
            # if self.is_qk:
            #     weight = MXAsymQuantizer.apply(
            #         real_weights, weight_clip_val, self.w_bits, self.weight_layerwise
            #     )
            # else:
            #     weight = MX1AsymQuantizer.apply(
            #         real_weights, weight_clip_val, self.w_bits, self.weight_layerwise
            #     )
        else:
            if self.w_bits == 1:
                if self.weight_layerwise:
                    scaling_factor = torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = torch.zeros_like(real_weights).detach()
                    groupsize = 8
                    vector_size = 2
                    dim_group = int(real_weights.shape[-1] / groupsize)
                    for i in range(dim_group):
                        i1 = i*groupsize
                        i2 = min((i+1)*groupsize, real_weights.shape[-1])
                        scaling_factor[:, i1:i2] = torch.mean(
                            abs(real_weights[:, i1:i2]), dim=1, keepdim=True
                        ).detach()
                    # for i in range(dim_group):
                    #     i1 = i*groupsize
                    #     i2 = min((i+1)*groupsize, real_weights.shape[-1])
                    #     even_idx = (
                    #         i1 + 
                    #         torch.arange(0,int(groupsize/vector_size),device=scaling_factor.device)
                    #     )
                    #     odd_idx = even_idx + 1
                    #     scaling_factor[:, even_idx] = torch.mean(
                    #         abs(real_weights[:, even_idx]), dim=1, keepdim=True
                    #     ).detach()
                    #     scaling_factor[:, odd_idx] = torch.mean(
                    #         abs(real_weights[:, odd_idx]), dim=1, keepdim=True
                    #     ).detach()
                    
                    

                    # scaling_factor = torch.mean(
                    #     abs(real_weights), dim=1, keepdim=True
                    # ).detach()
                quan_weights_no_grad = scaling_factor * (
                    torch.sign(real_weights / scaling_factor)
                )
            # elif self.w_bits == 2:
            #     scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            #     quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
            else:
                num_bits = 2 ** (self.w_bits - 1)
                clip_val = 1 - 1e-2
                if self.weight_layerwise:
                    scaling_factor = 2 * torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = (
                        2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
                    )
                quan_weights_no_grad = (
                    scaling_factor
                    * (
                        torch.round(
                            torch.clamp(
                                real_weights / scaling_factor, -clip_val, clip_val
                            )
                            * num_bits
                            - 0.5
                        )
                        + 0.5
                    )
                    / num_bits
                )

            weight = (
                quan_weights_no_grad.detach() - real_weights.detach() + real_weights
            )
        # Quantize inputs
        if self.a_bits < 32 and self.a_bits > 2:
            act_clip_val = torch.tensor([-2.0, 2.0])
            input_ = self.act_quantizer.apply(
                input_, act_clip_val, self.a_bits, self.act_layerwise
            )

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

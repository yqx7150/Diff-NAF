import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend

class _hash_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)

    def forward(ctx, inputs, embeddings, offsets, base_resolution, calc_grad_inputs=False):





        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous().to(inputs.device)

        B, D = inputs.shape
        L = offsets.shape[0] - 1
        C = embeddings.shape[1]
        H = base_resolution


        outputs = torch.zeros(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.zeros(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, H, calc_grad_inputs, dy_dx)


        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, H]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):


        grad = grad.contiguous()

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, H, calc_grad_inputs, dy_dx, grad_inputs)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None
        else:
            return None, grad_embeddings, None, None, None


hash_encode = _hash_encode.apply


class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19):
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')


        self.offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = base_resolution * 2 ** i
            params_in_level = min(self.max_params, (resolution + 1) ** input_dim)

            self.offsets.append(offset)
            offset += params_in_level
        self.offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(self.offsets, dtype=np.int32))
        
        self.n_params = self.offsets[-1] * level_dim


        self.embeddings = nn.Parameter(torch.zeros(offset, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} H={self.base_resolution} params={self.embeddings.shape}"
    
    def forward(self, inputs, size=1):



        if inputs.min().item() < -size or inputs.max().item() > size:
            raise ValueError(f'HashGrid encoder: inputs range [{inputs.min().item()}, {inputs.max().item()}] not in [{-size}, {size}]!')

        inputs = (inputs + size) / (2 * size)
        


        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = hash_encode(inputs, self.embeddings, self.offsets, self.base_resolution, inputs.requires_grad)
        outputs = outputs.view(prefix_shape + [self.output_dim])



        return outputs